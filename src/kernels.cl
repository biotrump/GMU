/*
 * Accelerated k-means and mean-shift algorithms via OpenCL
 * Authors: Martin Simon & Pavel Sirucek
 *
 * OpenCL kernels
 */


 /*
 * Prirazeni pixelu ke stredum.
 */
__kernel void assignCentroids(__global uchar4* input, __global uchar4* output, __global uchar4* centroids, __global uint* pixels, uint width, uint height, uint K)
{
	uint gidX = get_global_id(0);
	uint gidY = get_global_id(1);

	uint pixel_index = gidX + width * gidY;
	float min_dist = 1000000.0f; // nejmensi vzdalenost

	for (uint i = 0; i < K; i++)
	{	// spocteni vzdalenosti pixelu od stredu
		float dist = 0.0f;
		float4 distxyz;

		distxyz = convert_float4(centroids[i]) - convert_float4(input[pixel_index]);
		distxyz = distxyz * distxyz;
		dist = distxyz.x + distxyz.y + distxyz.z;

		// mensi vzdalenost? - priradime stred
		if (dist < min_dist)
		{
			min_dist = dist;
			pixels[pixel_index] = i;
		}
	}

	output[pixel_index] = centroids[pixels[pixel_index]];
	output[pixel_index].w = 255;
}

/*
 * Prepocitani stredu shluku.
 */
__kernel void recomputeCenters(__global uchar4* input, __global uchar4* centroids, __global uint* pixels, uint width, uint height, uint K)
{
	uint center = get_global_id(0);

	if (center < K)
	{
		float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
		uint num = 0;

		// zprumerovani stredu pixelu a vytvoreni noveho stredu
		for (uint i = 0; i < width * height; i++)
		{
			if (pixels[i] == center)
			{
				sum += convert_float4(input[i]);
				num++;
			}
		}

		uchar4 newCenter = convert_uchar4(sum / convert_float(num));

		centroids[center] = newCenter;
		centroids[center].w = 255;
	}
}


__kernel void meanshift(__global uchar4* input, uint width, uint height, uint winsize, __global uchar4* output)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    float h = convert_float(winsize);

    //reconstruct window by given param
    float actx = convert_float(x);
    float acty = convert_float(y);
    int wymax = acty + ((winsize-1) / 2) + 1;
    int wymin = acty - ((winsize-1) / 2);
    int wxmax = actx + ((winsize-1) / 2) + 1;
    int wxmin = actx - ((winsize-1) / 2);
    uint limit = max(width,height);

    float oldx = actx;
    float oldy = acty;

    //mean-shift
    float numX, numY, den;
    float ecko;
    float hinv = 1/h;

    //cycle control value
    int iter = 0;

    int gid;
    float length;

    float normalXDiff, normalYDiff, normalRDiff, normalGDiff, normalBDiff;

    //begincycle - until the step is bigger than relevant
    do {
        numX = numY = den = 0;

        for (int wy = wymin; wy < wymax; wy++)
        {
            for (int wx = wxmin; wx < wxmax; wx++)
            {
                if (wx < 0 || wy < 0 || wx >= width|| wy >= height)
                {
                    //out of original window
                    continue;
                }
                else
                {
                    //compute lengths for [wx,wy] - [x,y]
                    gid = convert_int_rte(actx) + convert_int_rte(acty)*width;

                    normalXDiff = actx - wx;
                    normalYDiff = acty - wy;
                    normalRDiff = convert_float(input[gid].s0) - convert_float(input[wx + wy*width].s0);
                    normalGDiff = convert_float(input[gid].s1) - convert_float(input[wx + wy*width].s1);
                    normalBDiff = convert_float(input[gid].s2) - convert_float(input[wx + wy*width].s2);

                    /* ||act - w||^2 */
                    length =
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
                }
            }
        }

        //recompute mean of the window
        oldx = actx;
        oldy = acty;

        actx = numX/den;
        acty = numY/den;

        //shift the window to the mean be in the center
        wymax = convert_int_rte(acty) + ((winsize-1) / 2) + 1;
        wymin = convert_int_rte(acty) - ((winsize-1) / 2);
        wxmax = convert_int_rte(actx) + ((winsize-1) / 2) + 1;
        wxmin = convert_int_rte(actx) - ((winsize-1) / 2);

        if(fabs(oldx - actx) < 0.1)
        {
            if(fabs(oldy - acty) < 0.1)
            {
                //the step is lower set
                break;
            }
        }

        //endcycle
        iter++;
    } while (iter < limit);

    actx = convert_float(max(convert_int_rte(actx),0));
    acty = convert_float(max(convert_int_rte(acty),0));

    actx = convert_float(min(convert_int_rte(actx),convert_int_rte(width)-1));
    acty = convert_float(min(convert_int_rte(acty),convert_int_rte(height)-1));

    //set result color
    output[x + y*width] = input[convert_int_rte(actx) + convert_int_rte(acty)*width];
}

