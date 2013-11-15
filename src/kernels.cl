/*
 * Accelerated k-means and mean-shift algorithms via OpenCL
 * Authors: Martin Simon & Pavel Sirucek
 *
 * OpenCL kernels
 */


__kernel void k_means()
{
}

__kernel void mean_shift()
{
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
