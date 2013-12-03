/*
 * Accelerated k-means and mean-shift algorithms via OpenCL
 * Authors: Martin Simon & Pavel Sirucek
 *
 * OpenCL kernels
 */


/* Implementace K-means
 * Prozatim v jednom kernelu (vyhoda je ze se nemusi sznchronizovat postup zpracovani kernelu,
                              ani si nejsem jisty jak to udelat, "nevyhodou" je uzke hrdlo pri pre
							  pocitavani stredu, kdy pracuje K vlaken a ostatni cekaji, tohle se
							  vetsim poctem kernelu nevyresi, protoze by stejne vlakna musely
							  pockat na prepocitani stredu)
 * Postup zpracovani (jak by mel byt, nejsem si jisty zda to tak je):
 *
 * Kazdy pixel se priradi do clusteru (k nejblizsimu stredu).
 * Pomoci barrier() se pocka na vsechna vlakna, pote se K vlaken
 * postara o prepocitani stredu. Pocka se na vsechna prepocitani a jede se znovu.
 */
 /* TODO - velke obrazky, zpracuje se cast..problem bude nekde u maximalniho poctu vlaken nebo tak
         - artefakty u nejakych obrazku, pripadne jednou to zpracuje dobre, podruhe podivne
		 - doba zpracovani (?)
		 - zadani parametru K z cmdl, omezeni na max napr 16.
		 - doladit (nekonceny cyklus napr, detekce zmeny clsuteru apod.) (?)
 */
__kernel void kmeans(__global uchar4* input, __global uchar4* output, __global uchar4* centroids, __global uint* pixels, uint width, uint height, uint K)
{
	uint gidX = get_global_id(0);
	uint gidY = get_global_id(1);
	bool center_change = true; // detekce zmeny stredu (prepocitani)

	uint pixel_index = gidX + width * gidY;
	float min_dist; // nejmensi vzdalenost

	uint last_cluster; // minuly cluster nebo stred
	float last_dist = 0.0f; // minula vzdalenost

	while (true)
	{
		for (uint i = 0; i < K; i++)
		{
			// spocteni vzdalenosti pixelu od stredu
			float dist = 0.0f;
			float3 distxyz;
			distxyz.x = convert_float(centroids[i].x) - convert_float(input[pixel_index].x);
			distxyz.x *= distxyz.x;
			distxyz.y = convert_float(centroids[i].y) - convert_float(input[pixel_index].y);
			distxyz.y *= distxyz.y;
			distxyz.z = convert_float(centroids[i].z) - convert_float(input[pixel_index].z);
			distxyz.z *= distxyz.z;
			dist = distxyz.x + distxyz.y + distxyz.z;

			if (i == 0)
			{
				min_dist = dist;
				pixels[pixel_index] = i;
			}
			else
			{
				if (dist < min_dist)
				{ // prirazeni pixelu do noveho clusteru
					min_dist = dist;
					pixels[pixel_index] = i;
				}
			}

		}

		// konec pokud se pixel nepresunul, vzdalenost zustala stejna (tedy zadna zmena) a pokud se neprepocitavaji stredy
		if (last_cluster == pixels[pixel_index] && last_dist == min_dist && !center_change)
		{
			output[pixel_index] = centroids[pixels[pixel_index]];
			output[pixel_index].w = 255;
			return;
		}

		last_cluster = pixels[pixel_index];
		last_dist = min_dist;
		// pockame na zpracovani vsech pixelu
		barrier(CLK_GLOBAL_MEM_FENCE);

		// poslani K vlaken na clustery -> prepocitani stredu
		if (gidX < K && center_change)
		{
			// prumer hodnot
			float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
			uint num = 0;

			for (uint i = 0; i < width * height; i++)
			{
				if (pixels[i] == gidX)
				{
					sum += convert_float4(input[i]);
					num++;
				}
			}
			// novy stred
			uchar4 newCenter = convert_uchar4(sum / convert_float(num));

			if (centroids[gidX].x != newCenter.x || centroids[gidX].y != newCenter.y || centroids[gidX].z != newCenter.z)
			{
				centroids[gidX] = newCenter;
				centroids[gidX].w = 255;
			}
			else // stred se nezmenil
			{
				center_change = false;
			}
		}
		else // toto vlakno neprepocitava stredy
		{
			center_change = false;
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}


__kernel void meanshift(__global uchar4* input, __global uchar4* output, __global uint* peaks, __global uint* counts, uint width, uint height, uint winsize, __global int* peakcount, __global uint* uniquepeaks, __global uchar4* colors)
{
int x = get_global_id(0);
int y = get_global_id(1);

//reconstruct window by given param
float actx = convert_float(x);
float acty = convert_float(y);
int wymax = acty + ((winsize-1) / 2) + 1;
int wymin = acty - ((winsize-1) / 2);
int wxmax = actx + ((winsize-1) / 2) + 1;
int wxmin = actx - ((winsize-1) / 2);

float oldx = convert_float(x);
float oldy = convert_float(y);

//cycle control value
bool cont = true;
int iter = 0;
//mean structures
float newX, newY;

//begincycle - until the step is bigger than relevant
do {
    newX = 0;
    newY = 0;
    for (int wy = wymin; wy < wymax; wy++)
    {
        for (int wx = wxmin; wx < wxmax; wx++)
        {
            if (wx < 0 || wy < 0 || wx >= width|| wy >= height)
            {
                //out of original window
                //set 0 to the position of the x in local cache
                continue;
            }
            else if (wx == convert_int(actx) && wy == convert_int(acty))
            {
                //set 1 to the position of the x
                //to [actx-wxmin,acty-wymin] set 1

                newX += (wx - actx);
                newY += (wy - acty);
            }
            else
            {
                //compute lengths for [wx,wy] - [x,y]
                //if the length is bigger the maxlength, set 0 there
                //otherwise set the length
                int gid = convert_int_rte(actx) + convert_int_rte(acty)*width;

                float normalXDiff = actx/(width-1) - wx/(width-1);
                float normalYDiff = acty/(height-1) - wx/(height-1);
                float normalRDiff = convert_float(input[gid].s0)/255 - convert_float(input[wx + wy*width].s0)/255;
                float normalGDiff = convert_float(input[gid].s1)/255 - convert_float(input[wx + wy*width].s1)/255;
                float normalBDiff = convert_float(input[gid].s2)/255 - convert_float(input[wx + wy*width].s2)/255;

                float length = 1-sqrt(
                    normalXDiff*normalXDiff +
                    normalYDiff*normalYDiff +
                    normalRDiff*normalRDiff +
                    normalGDiff*normalGDiff +
                    normalBDiff*normalBDiff);

//printf("(%f) + (%f) + (%f) + (%f)+ (%f) = %f \n", normalXDiff, normalYDiff, normalRDiff, normalGDiff, normalBDiff,length);

                newX += (wx - actx) * length;
                newY += (wy - acty) * length;
            }
        }
    }
//printf("[%d:%d]- old:[%f,%f], new diff:[%f,%f]\n", x,y,actx,acty,newX,newY);
    //recompute mean of the window
    oldx = actx;
    oldy = acty;

    actx = actx + newX;
    acty = acty + newY;

    //shift the window to the mean be in the center
    wymax = convert_int_rte(acty) + ((winsize-1) / 2) + 1;
    wymin = convert_int_rte(acty) - ((winsize-1) / 2);
    wxmax = convert_int_rte(actx) + ((winsize-1) / 2) + 1;
    wxmin = convert_int_rte(actx) - ((winsize-1) / 2);

    if(fabs(newX) <= 0.2 && fabs(newY) <= 0.2)
    {
        //the step is lower the 0.5pix
        cont = false;
    }


    //endcycle
    iter++;
} while (cont && iter < 1000);

actx = convert_float(max(convert_int_rte(actx),0));
acty = convert_float(max(convert_int_rte(acty),0));

actx = convert_float(min(convert_int_rte(actx),convert_int_rte(width)-1));
acty = convert_float(min(convert_int_rte(acty),convert_int_rte(height)-1));

int gid = x + y*width;

//store the peak position to peaks array
peaks[gid] = convert_int_rte(actx) + convert_int_rte(acty)*width;

//increment value on peak position in counts array
//counts[convert_int_rte(actx) + convert_int_rte(acty)*width] = counts[convert_int_rte(actx) + convert_int_rte(acty)*width] + 1;

barrier(CLK_GLOBAL_MEM_FENCE);

int peak = peaks[gid];

for (int i=0; i<100;i++)
{
    if(peaks[peaks[peak]] == peaks[peak])
    {
        break;
    }
    peaks[gid] = peaks[peak];
    peak = peaks[peaks[peak]];
}

barrier(CLK_GLOBAL_MEM_FENCE);

if (peakcount[0] == -1)
{
    peakcount[0] = 0; //lock the critical section

    //begin of critical section
    //printf("[%d,%d] critic section - begin\n",x,y);
    for (int yy=0; yy<height; yy++)
    {
        for (int xx=0; xx<width; xx++)
        {
            int actpeak = peaks[xx + width*yy];

            /* Search the array for actpeak */
            int i;
            for (i=0; i<peakcount[0];i++)
            {
                if (uniquepeaks[i] == actpeak)
                {
                    break;
                }
            }

            /* If went through all the elements = haven't found */
            if (i == peakcount[0])
            {
                uniquepeaks[peakcount[0]++] = actpeak;
            }
        }
    }

    //printf("pocet vrcholu: %d\n", peakcount[0]);
    int colorStep = (256*256*256)/peakcount[0];
//printf("%d - [%d,%d,%d]\n", colorStep, (colorStep / 65536) % 256, (colorStep / 265) % 256, colorStep % 256);
    for (int n =0; n<peakcount[0]; n++)
    {

        int actstep = n*colorStep;
        colors[uniquepeaks[n]].s0 = (actstep / 65536) % 256;
        colors[uniquepeaks[n]].s1 = (actstep / 256) % 256;
        colors[uniquepeaks[n]].s2 = actstep % 256;
        colors[uniquepeaks[n]].s3 = 255;
        //printf("[%d,%d,%d]\n",colors[uniquepeaks[n]].s0,colors[uniquepeaks[n]].s1,colors[uniquepeaks[n]].s2);
        //TODO tady se barvy spravne ulozi
    }

    //printf("[%d,%d] critic section - end\n",x,y);
}
printf("11111111111\n");
barrier(CLK_GLOBAL_MEM_FENCE); //opravdu funguje???
printf("00000000000\n");
gid = x + y*width;
//printf("[%d,%d,%d]\n",colors[peaks[x + y*width]].s0,colors[peaks[x + y*width]].s1,colors[peaks[x + y*width]].s2);

//peak = peaks[gid];
output[gid] = colors[peaks[gid]];
//TODO tady je vetsina barev spatne zobrazena.. jakto?!
//printf("[%d,%d,%d]\n", output[gid].s0, output[gid].s1, output[gid].s2);
//printf("%v4\n",output[gid]);

}

__kernel void ms_begin(__global uchar4* input, __global uint* peaks, uint width, uint height, uint winsize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    float h = 1;

    //reconstruct window by given param
    float actx = convert_float(x);
    float acty = convert_float(y);
    int wymax = acty + ((winsize-1) / 2) + 1;
    int wymin = acty - ((winsize-1) / 2);
    int wxmax = actx + ((winsize-1) / 2) + 1;
    int wxmin = actx - ((winsize-1) / 2);

    float oldx = convert_float(x);
    float oldy = convert_float(y);

    //mean-shift
    double numX, numY, den;
    double ecko;
    float hinv = 1/h;

    //cycle control value
    bool cont = true;
    int iter = 0;
    //mean structures
    //float newX, newY;

    //begincycle - until the step is bigger than relevant
    do {
        //newX = 0;
        //newY = 0;

        numX = numY = den =0;

        for (int wy = wymin; wy < wymax; wy++)
        {
            for (int wx = wxmin; wx < wxmax; wx++)
            {
                if (wx < 0 || wy < 0 || wx >= width|| wy >= height)
                {
                    //out of original window
                    //set 0 to the position of the x in local cache
                    continue;
                }
//                else if (wx == convert_int(actx) && wy == convert_int(acty))
//                {
//                    //set 1 to the position of the x
//                    //to [actx-wxmin,acty-wymin] set 1
//
//                    newX += (wx - actx);
//                    newY += (wy - acty);
//                }
                else
                {
                    //compute lengths for [wx,wy] - [x,y]
                    //if the length is bigger the maxlength, set 0 there
                    //otherwise set the length
                    int gid = convert_int_rte(actx) + convert_int_rte(acty)*width;

                    float normalXDiff = actx/(width-1) - wx/(width-1);
                    float normalYDiff = acty/(height-1) - wy/(height-1);
                    float normalRDiff = convert_float(input[gid].s0)/255 - convert_float(input[wx + wy*width].s0)/255;
                    float normalGDiff = convert_float(input[gid].s1)/255 - convert_float(input[wx + wy*width].s1)/255;
                    float normalBDiff = convert_float(input[gid].s2)/255 - convert_float(input[wx + wy*width].s2)/255;

                    //float length = 1-sqrt(

                    /* ||act - w||^2 */
                    float length =
                        normalXDiff * normalXDiff +
                        normalYDiff * normalYDiff +
                        normalRDiff * normalRDiff +
                        normalGDiff * normalGDiff +
                        normalBDiff * normalBDiff;

                    /* e^((-length)/h) */
                    ecko = exp(-hinv * length);

                    /* sum of X coordinate - numerator */
                    numX += wx * ecko;

                    /* sum of Y coordinate - numerator */
                    numY += wy * ecko;

                    /* sum of denominator */
                    den += ecko;

                   // newX += (wx - actx) * length;
                    //newY += (wy - acty) * length;
                }
            }
        }


        //recompute mean of the window
        oldx = actx;
        oldy = acty;

        //actx = actx + newX;
        actx = numX/den;
        acty = numY/den;
        //acty = acty + newY;
//printf("[%d:%d](%d)- old:[%f,%f], new diff:[%f,%f]\n", x,y,iter,oldx,oldy,actx,acty);
        //shift the window to the mean be in the center
        wymax = convert_int_rte(acty) + ((winsize-1) / 2) + 1;
        wymin = convert_int_rte(acty) - ((winsize-1) / 2);
        wxmax = convert_int_rte(actx) + ((winsize-1) / 2) + 1;
        wxmin = convert_int_rte(actx) - ((winsize-1) / 2);

        //if(fabs(newX) <= 0.1 && fabs(newY) <= 0.1)
        if(fabs(oldx - actx) <= 0.1 && fabs(oldy - acty) <= 0.1)
        {
            //the step is lower the 0.5pix
            cont = false;
        }


        //endcycle
        iter++;
    } while (cont && iter < 1000);

    actx = convert_float(max(convert_int_rte(actx),0));
    acty = convert_float(max(convert_int_rte(acty),0));

    actx = convert_float(min(convert_int_rte(actx),convert_int_rte(width)-1));
    acty = convert_float(min(convert_int_rte(acty),convert_int_rte(height)-1));


    //store the peak position to peaks array
    peaks[x + y*width] = convert_int_rte(actx) + convert_int_rte(acty)*width;
}




__kernel void ms_optimize(__global uint* peaks, uint width)
{
//    int gid = get_global_id(0) + width * get_global_id(1);
//
//    int peak = peaks[gid];
//
//    for (int i=0; i<100;i++)
//    {
//        if(peaks[peaks[peak]] == peaks[peak])
//        {
//            break;
//        }
//        printf("neco");
//        peaks[gid] = peaks[peak];
//        peak = peaks[peaks[peak]];
//    }
}




__kernel void ms_color(__global uchar4* output, __global uchar* colors, __global uint* peaks, uint width)
{
    int gid = get_global_id(0) + width * get_global_id(1);

    output[gid] = colors[peaks[gid]];
}