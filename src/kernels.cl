/*
 * Accelerated k-means and mean-shift algorithms via OpenCL
 * Authors: Martin Simon & Pavel Sirucek
 *
 * OpenCL kernels
 */


__kernel void k_means()
{
}

__kernel void mean_shift(__global uchar4* input, __global uint* peaks, __global uint* counts,
__local float* cache, uint width, uint height, uint winsize, float maxlength)
{
int x = get_global_id(0);
int y = get_global_id(1);


//reconstruct window by given param
int actx = x;
int acty = y;
int wymax = acty + ((winsize-1) / 2) + 1;
int wymin = acty - ((winsize-1) / 2);
int wxmax = actx + ((winsize-1) / 2) + 1;
int wxmin = actx - ((winsize-1) / 2);
int oldx = x;
int oldy = y;

//cycle control value
bool cont = true;

//mean structures
float denominatorX = 0;
float numeratorX = 0;
float denominatorY = 0;
float numeratorY = 0;
float fNumerator;

//begincycle - until the step is bigger than relevant
do {
    for (int wy = wymin; wy < wymax; wy++)
    {
        for (int wx = wxmin; wx < wxmax; wx++)
        {
            if (wx < 0 || wy < 0 || wx > width-1|| wy > height-1)
            {
                //out of original window
                //set 0 to the position of the x in local cache
                cache[wx - wxmin + (wy-wymin)*winsize] = convert_float(0);
            }
            else if (wx == actx && wy == acty)
            {
                //set 1 to the position of the x
                //to [actx-wxmin,acty-wymin] set 1
                cache[wx - wxmin + (wy-wymin)*winsize] = convert_float(1);
            }
            else
            {
                //compute lengths for [wx,wy] - [x,y]
                //if the length is bigger the maxlength, set 0 there
                //otherwise set the length
                float length = 1 - 1/sqrt(
                    (actx-wx)*(actx-wx) +
                    (acty-wy)*(acty-wy) +
                    (input[actx + acty*width].s0 - input[wx + wy*width].s0)*(input[actx + acty*width].s0 - input[wx + wy*width].s0) +
                    (input[actx + acty*width].s1 - input[wx + wy*width].s1)*(input[actx + acty*width].s1 - input[wx + wy*width].s1) +
                    (input[actx + acty*width].s2 - input[wx + wy*width].s2)*(input[actx + acty*width].s2 - input[wx + wy*width].s2));
                if (length > maxlength)
                {
                    cache[wx - wxmin + (wy-wymin)*winsize] = convert_float(0);
                }
                else
                {
                    //store the value to its place
                    cache[wx - wxmin + (wy-wymin)*winsize] = convert_float(1-length);

                    //add numerator and denominator for the X mean computation
                    fNumerator = sqrt((wx-actx)*(wx-actx));
                    numeratorX += fNumerator * (1-length);
                    denominatorX += fNumerator;

                    //add numerator and denominator for the Y mean computation
                    fNumerator = sqrt((wy-acty)*(wy-acty));
                    numeratorY += fNumerator * (1-length);
                    denominatorY += fNumerator;
                }
            }
        }
    }

    //recompute mean of the window
    actx = convert_int(numeratorX/denominatorX);
    acty = convert_int(numeratorY/denominatorY);

    //shift the window to the mean be in the center
    wymax = acty + ((winsize-1) / 2) + 1;
    wymin = acty - ((winsize-1) / 2);
    wxmax = actx + ((winsize-1) / 2) + 1;
    wxmin = actx - ((winsize-1) / 2);

    if (actx == oldx && acty == oldy)
    {
        //the step is lower the 0.5pix
        cont = false;
    }

    //endcycle
} while (cont);

//store the peak position to peaks array
peaks[x+y*width] = actx + acty*width;

//increment value on peak position in counts array
counts[actx + acty*width] = counts[actx + acty*width] + 1;

}

__kernel void mean_shift_result(__global uint* counts, __global uint* peaks, __global uchar4* output, uint width, uint height)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if(x < width && y < height)
	{
		int gid = x + y*width;
		output[gid] = add_sat(sobel_x[gid], sobel_y[gid]);
		output[gid].s3 = 255;
	}
}

//TODO delete following

__kernel void edge_x( __global uchar4* input, __global uchar4* output, uint width, uint height, __local float4* cache)
{
	int x = get_global_id(0); //defines column
	int y = get_global_id(1); //defines row

	int lx = get_local_id(0);
	int lw = get_local_size(0);

	if(lx == 0)
	{
		int startI = y*width;
		int endI = y*width + height - 1;
		output[startI] = (uchar4)(0, 0, 0, 0);
		output[endI] = (uchar4)(0, 0, 0, 0);
	}

	int kw = 0;
	while(kw < width)
	{
		int gindex = y*width + lx + kw;

		//cache in the row
		if(lx + kw < width)
		{
			cache[lx] = convert_float4(input[gindex]);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(lx + kw < width-1 && lx > 0 && lx < lw - 1)
		{
			output[gindex] = convert_uchar4(fabs(cache[lx - 1] - cache[lx + 1]));
		}

		kw += lw - 1;
	}
}

__kernel void edge_y( __global uchar4* input, __global uchar4* output, uint width, uint height, __local float4* cache)
{
	int x = get_global_id(0); //defines column
	int y = get_global_id(1); //defines row

	int ly = get_local_id(1);
	int lh = get_local_size(1);

	if(ly == 0)
	{
		int startI = x;
		int endI = (height-1)*width + x;
		output[startI] = (uchar4)(0, 0, 0, 0);
		output[endI] = (uchar4)(0, 0, 0, 0);
	}

	int kh = 0;
	while(kh < height)
	{
		int gindex = (ly + kh)*width + x;

		//cache in the column
		if(ly + kh < height)
		{
			cache[ly] = convert_float4(input[gindex]);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(ly + kh < height-1 && ly > 0 && ly < lh - 1)
		{
			output[gindex] = convert_uchar4_sat_rte(fabs(cache[ly - 1] - cache[ly + 1]));
		}

		kh += lh - 1;
	}
}


__kernel void edge_result(__global uchar4* sobel_x, __global uchar4* sobel_y, __global uchar4* output, uint width, uint height)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if(x < width && y < height)
	{
		int gid = x + y*width;
		output[gid] = add_sat(sobel_x[gid], sobel_y[gid]);
		output[gid].s3 = 255;
	}
}
