#define DEBUG

#include "sdlwrapper.h"
#include "error.h"
#include <stdio.h>
#include <CL/opencl.h>
#include <stdlib.h>


#define MIN(a, b) ((a) > (b) ? (b) : (a))
//global variables

SDL_Surface *screen;

cl_uchar4* h_inputImageData = NULL;
cl_uchar4* h_outputImageData = NULL;

//width and height of the image
int width = 0, height = 0;

cl_uint pixelSize = 32; //rgba 8bits per channel

//opencl stuff
cl_context context;
cl_command_queue commandQueue;
cl_kernel edgeXKernel, edgeYKernel, edgeResultKernel;
cl_program program;


/** CL memory buffer for images */
cl_mem d_inputImageBuffer = NULL;
cl_mem d_edgeXImageBuffer = NULL;
cl_mem d_edgeYImageBuffer = NULL;
cl_mem d_outputImageBuffer = NULL;

//the size of our blocks

/**
 * TODO preferovana velikost
 */
size_t kernelWorkGroupSize = 1024;
size_t blockSizeX = 1024;
size_t blockSizeY = 1;

int printTiming(cl_event event, const char* title){
    cl_ulong startTime;
    cl_ulong endTime;
    /* Display proiling info */
    cl_int status = clGetEventProfilingInfo(event,
                                     CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong),
                                     &startTime,
                                     0);
    CheckOpenCLError(status, "clGetEventProfilingInfo.(startTime)");
       

    status = clGetEventProfilingInfo(event,
                                     CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong),
                                     &endTime,
                                     0);
    
    CheckOpenCLError(status, "clGetEventProfilingInfo.(stopTime)");

    cl_double elapsedTime = (endTime - startTime) * 1e-6;

    printf("%s elapsedTime %.3lf ms\n", title, elapsedTime);

    return 0;
}

char* loadProgSource(const char* cFilename)
{
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;

	pFileStream = fopen(cFilename, "rb");
	if(pFileStream == 0) 
	{       
		return NULL;
	}


    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + 1); 
    if (fread(cSourceString, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    cSourceString[szSourceLength] = '\0';

    return cSourceString;
}

/**
 * Draw the output image to sdl surface
 */
int drawOutputImage(SDL_Surface *screen){

	
    SDL_Surface *temp = SDL_CreateRGBSurfaceFrom(h_outputImageData,
        width, height, pixelSize, width*4, 
        0x0000ff, 0x00ff00, 0xff0000, 0xff000000);
    SDL_Rect rec;
 
    rec.x = 0;
    rec.y = 0;
    rec.w = width;
    rec.h = height;
    
    SDL_Surface *output = SDL_DisplayFormatAlpha(temp);
    SDL_BlitSurface(output, &rec, screen, &rec);
    SDL_FreeSurface(temp);
    SDL_FreeSurface(output);
    return 0;
}

/**
 * Inicialize stuff on the client side
 */
int setupHost(const char *inputImageName)
{
	SDL_Surface *inputImage;

	if(readImage(inputImageName, &inputImage) < 0)
	{
		return -1;
	}

	width = inputImage->w;
	height = inputImage->h;

	h_inputImageData = (cl_uchar4*) malloc(width * height * sizeof(cl_uchar4));

	if(h_inputImageData == NULL)
	{
		logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory.");
		return -1;
	}

	memcpy(h_inputImageData, inputImage->pixels, width * height * sizeof(cl_uchar4));

	//allocate output image

	h_outputImageData = (cl_uchar4 *) malloc(width * height * sizeof(cl_uchar4));
	
	if(h_outputImageData == NULL)
	{
		logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory.");
		return -1;
	}

	memset(h_outputImageData, 0, width * height * sizeof(cl_uchar4));

	SDL_FreeSurface(inputImage);
	return 0;
}

/**
 * Initialize host and opencl device
 */
int setupCL()
{
	cl_int ciErr = CL_SUCCESS;

	// Get Platform
	cl_platform_id *cpPlatforms;
	cl_uint cuiPlatformsCount;
	ciErr = clGetPlatformIDs(0, NULL, &cuiPlatformsCount); CheckOpenCLError( ciErr, "clGetPlatformIDs: cuiPlatformsNum=%i", cuiPlatformsCount );
	cpPlatforms = (cl_platform_id*)malloc(cuiPlatformsCount*sizeof(cl_platform_id));
	ciErr = clGetPlatformIDs(cuiPlatformsCount, cpPlatforms, NULL); CheckOpenCLError( ciErr, "clGetPlatformIDs" );
	
	cl_platform_id platform = 0;

	const unsigned int TMP_BUFFER_SIZE = 1024;
	char sTmp[TMP_BUFFER_SIZE];

	for(unsigned int f0=0; f0<cuiPlatformsCount; f0++){
		bool shouldBrake = false;
		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_PROFILE, TMP_BUFFER_SIZE, sTmp, NULL);    CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_PROFILE=%s",f0, sTmp);
		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_VERSION, TMP_BUFFER_SIZE, sTmp, NULL);    CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_VERSION=%s",f0, sTmp);
		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_NAME, TMP_BUFFER_SIZE, sTmp, NULL);       CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_NAME=%s",f0, sTmp);
		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_VENDOR, TMP_BUFFER_SIZE, sTmp, NULL);     CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_VENDOR=%s",f0, sTmp);

		//prioritize AMD and CUDA platforms
		
		if ((strcmp(sTmp, "Advanced Micro Devices, Inc.") == 0) || (strcmp(sTmp, "NVIDIA Corporation") == 0)) {
			platform = cpPlatforms[f0];
		}
		
		//prioritize Intel
		/*if ((strcmp(sTmp, "Intel(R) Corporation") == 0)) {
			platform = cpPlatforms[f0];
		}*/

		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_EXTENSIONS, TMP_BUFFER_SIZE, sTmp, NULL); CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_EXTENSIONS=%s",f0, sTmp);
		printf("\n");
	}

	if(platform == 0)
	{ //no prioritized found
		if(cuiPlatformsCount > 0)
		{ 
			platform = cpPlatforms[0];
		} else {
			logMessage(DEBUG_LEVEL_ERROR, "No device was found");
			return -1;
		}
	}
	// Get Devices
	cl_uint cuiDevicesCount;
	ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &cuiDevicesCount); CheckOpenCLError( ciErr, "clGetDeviceIDs: cuiDevicesCount=%i", cuiDevicesCount );
	cl_device_id *cdDevices = (cl_device_id*)malloc(cuiDevicesCount*sizeof(cl_device_id));
	ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, cuiDevicesCount, cdDevices, NULL); CheckOpenCLError( ciErr, "clGetDeviceIDs" );
	
	unsigned int deviceIndex = 0;

	for(unsigned int f0=0; f0<cuiDevicesCount; f0++){
		cl_device_type cdtTmp;
		size_t iDim[3];

		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_TYPE, sizeof(cdtTmp), &cdtTmp, NULL);
		CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_TYPE=%s%s%s%s",f0, cdtTmp&CL_DEVICE_TYPE_CPU?"CPU,":"", 
			cdtTmp&CL_DEVICE_TYPE_GPU?"GPU,":"", 
			cdtTmp&CL_DEVICE_TYPE_ACCELERATOR?"ACCELERATOR,":"", 
			cdtTmp&CL_DEVICE_TYPE_DEFAULT?"DEFAULT,":"");

		if(cdtTmp & CL_DEVICE_TYPE_CPU)
		{ //prioritize gpu if both cpu and gpu are available
			deviceIndex = f0;
		}

		cl_bool bTmp;
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_AVAILABLE, sizeof(bTmp), &bTmp, NULL);             CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_AVAILABLE=%s",f0, bTmp?"YES":"NO");
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_NAME, TMP_BUFFER_SIZE, sTmp, NULL);                CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_NAME=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_VENDOR, TMP_BUFFER_SIZE, sTmp, NULL);              CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_VENDOR=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DRIVER_VERSION, TMP_BUFFER_SIZE, sTmp, NULL);             CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DRIVER_VERSION=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_PROFILE, TMP_BUFFER_SIZE, sTmp, NULL);             CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_PROFILE=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_VERSION, TMP_BUFFER_SIZE, sTmp, NULL);             CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_VERSION=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(iDim), iDim, NULL);    CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_MAX_WORK_ITEM_SIZES=%ix%ix%i",f0, iDim[0], iDim[1], iDim[2]);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), iDim, NULL);  CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_MAX_WORK_GROUP_SIZE=%i",f0, iDim[0]);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_EXTENSIONS, TMP_BUFFER_SIZE, sTmp, NULL);          CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_EXTENSIONS=%s",f0, sTmp);
		printf("\n");
	}

	cl_context_properties cps[3] = 
    { 
        CL_CONTEXT_PLATFORM, 
        (cl_context_properties)platform, 
        0 
    };

	

	//create context
	context = clCreateContext(cps, 1, &cdDevices[deviceIndex], NULL, NULL, &ciErr);  CheckOpenCLError( ciErr, "clCreateContext" );
	//may use clCreateContextFromType than choose a device based on the returned devices
	
	printf("here");
	//create a command queue
	commandQueue = clCreateCommandQueue(context, cdDevices[deviceIndex], 
			CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ciErr); 
	CheckOpenCLError( ciErr, "clCreateCommandQueue" );

	//==================================================================================
	//allocate and initialize memory buffers

	//we are only going to read from this
	d_inputImageBuffer = clCreateBuffer(context,
										CL_MEM_READ_ONLY,
										width * height * pixelSize,
										0,
										&ciErr);
	CheckOpenCLError(ciErr, "CreateBuffer inputImage");

	//write our image to the buffer
	// Write Data to inputImageBuffer - blocking write
    ciErr = clEnqueueWriteBuffer(commandQueue,
                                  d_inputImageBuffer,
                                  CL_TRUE, //blocking write
                                  0,
                                  width * height * sizeof(cl_uchar4),
                                  h_inputImageData,
                                  0,
                                  0,
                                  0);

	CheckOpenCLError(ciErr, "Copy input image data");

	//create mid buffers
	d_edgeXImageBuffer = clCreateBuffer(context,
										CL_MEM_READ_WRITE,
										width * height * pixelSize,
										0, &ciErr);
	CheckOpenCLError(ciErr, "CreateBuffer edgeX");

	d_edgeYImageBuffer = clCreateBuffer(context,
										CL_MEM_READ_WRITE,
										width * height * pixelSize,
										0, &ciErr);
	CheckOpenCLError(ciErr, "CreateBuffer edgeY");


	//output image buffer - write only
	d_outputImageBuffer = clCreateBuffer(context,
										CL_MEM_WRITE_ONLY,
										width * height * pixelSize,
										0,
										&ciErr);
	CheckOpenCLError(ciErr, "Allocate output buffer");

	

	//=================================================================================
	// Create and compile and openCL program

	char *cSourceCL = loadProgSource("../edge.cl");
	
	program = clCreateProgramWithSource(context, 1, (const char **)&cSourceCL, NULL, &ciErr);  CheckOpenCLError( ciErr, "clCreateProgramWithSource" );
	free(cSourceCL);

	ciErr = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	
	cl_int logStatus;

	//build log
    char *buildLog = NULL;
    size_t buildLogSize = 0;
    logStatus = clGetProgramBuildInfo( program, 
										cdDevices[deviceIndex], 
										CL_PROGRAM_BUILD_LOG, 
										buildLogSize, 
										buildLog, 
										&buildLogSize);
	
	CheckOpenCLError(logStatus, "clGetProgramBuildInfo.");

    buildLog = (char*)malloc(buildLogSize);
    if(buildLog == NULL)
    {
        printf("Failed to allocate host memory. (buildLog)");
        return -1;
    }
    memset(buildLog, 0, buildLogSize);

    logStatus = clGetProgramBuildInfo (program, 
										cdDevices[deviceIndex], 
										CL_PROGRAM_BUILD_LOG, 
										buildLogSize, 
										buildLog, 
										NULL);
	CheckOpenCLError(logStatus, "clGetProgramBuildInfo.");

    printf(" \n\t\t\tBUILD LOG\n");
    printf(" ************************************************\n");
    printf("%s", buildLog);
    printf(" ************************************************\n");
    free(buildLog);
	
	CheckOpenCLError( ciErr, "clBuildProgram" );

	//==========================================================================
	// kernels
	edgeResultKernel = clCreateKernel(program, "edge_result", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel edge_result" );
	edgeXKernel = clCreateKernel(program, "edge_x", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel edge_x" );
	edgeYKernel = clCreateKernel(program, "edge_y", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel edge_y" );

	// Check group size against group size returned by kernel
	size_t tempKernelWorkGroupSize;
    ciErr = clGetKernelWorkGroupInfo(edgeResultKernel,
									cdDevices[deviceIndex],
									CL_KERNEL_WORK_GROUP_SIZE,
									sizeof(size_t),
									&tempKernelWorkGroupSize,
									0);
	CheckOpenCLError(ciErr, "clGetKernelInfo");
    kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);

    ciErr = clGetKernelWorkGroupInfo(edgeXKernel,
									cdDevices[deviceIndex],
									CL_KERNEL_WORK_GROUP_SIZE,
									sizeof(size_t),
									&tempKernelWorkGroupSize,
									0);
	CheckOpenCLError(ciErr, "clGetKernelInfo");
    kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);

    ciErr = clGetKernelWorkGroupInfo(edgeYKernel,
									cdDevices[deviceIndex],
									CL_KERNEL_WORK_GROUP_SIZE,
									sizeof(size_t),
									&tempKernelWorkGroupSize,
									0);
	CheckOpenCLError(ciErr, "clGetKernelInfo");
    kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);

	if((blockSizeX * blockSizeY) > kernelWorkGroupSize)
    {
        printf("Out of Resources!\n");
        printf("Group Size specified: %i\n", blockSizeX * blockSizeY);
        printf("Max Group Size supported on the kernel: %i\n", kernelWorkGroupSize);
        printf("Falling back to %i.\n", kernelWorkGroupSize);

        if(blockSizeX > kernelWorkGroupSize)
        {
            blockSizeX = kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }

	free(cdDevices);

	return 0;
}


int checkWorkgroupSize(cl_kernel &pkernel)
{
	cl_int ciErr = CL_SUCCESS;
	size_t tempKernelWorkGroupSize;
	ciErr = clGetKernelWorkGroupInfo(pkernel,
									NULL, //this only workes if single device
									CL_KERNEL_WORK_GROUP_SIZE,
									sizeof(size_t),
									&tempKernelWorkGroupSize,
									0);
	CheckOpenCLError(ciErr, "clGetKernelInfo %u", tempKernelWorkGroupSize);
    kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);

	if((blockSizeX * blockSizeY) > kernelWorkGroupSize)
    {
        printf("Out of Resources!\n");
        printf("Group Size specified: %i\n", blockSizeX * blockSizeY);
        printf("Max Group Size supported on the kernel: %i\n", kernelWorkGroupSize);
        printf("Falling back to %i.\n", kernelWorkGroupSize);

        if(blockSizeX > kernelWorkGroupSize)
        {
            blockSizeX = kernelWorkGroupSize;
            blockSizeY = 1;
			return EXIT_FAILURE;
        }
    }
	return EXIT_SUCCESS;	
}

int runKernels(){
    int status;
    cl_event event_result, event_edgeX, event_edgeY;

    /* Setup arguments to the kernel */

//////////////////////////////////////////////////////////////////////////////////////////////////
// EDGE X

 /* input buffer */
    status = clSetKernelArg(edgeXKernel, 
                            0, 
                            sizeof(cl_mem), 
                            &d_inputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

    /* output buffer */
    status = clSetKernelArg(edgeXKernel, 
                            1, 
                            sizeof(cl_mem), 
                            &d_edgeXImageBuffer);
	
	CheckOpenCLError(status, "clSetKernelArg. (edgeXImage)");

    /* image width */
    status = clSetKernelArg(edgeXKernel, 
                            2, 
                            sizeof(cl_uint), 
                            &width);

	CheckOpenCLError(status, "clSetKernelArg. (width)");

	/* image height */
    status = clSetKernelArg(edgeXKernel, 
                            3, 
                            sizeof(cl_uint), 
                            &height);

	CheckOpenCLError(status, "clSetKernelArg. (height)");
    

    /* local memory */
	int cache_size = (blockSizeX) * (blockSizeY);
	status = clSetKernelArg(edgeXKernel, 
	                        4, 
	                        sizeof(cl_float4)*cache_size, 
	                        0);

	CheckOpenCLError(status, "clSetKernelArg. (local cache) %u", sizeof(cl_float4)*cache_size);

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
    size_t globalThreadsEdgeX[] = 
    {
        blockSizeX,
        height
    };
    size_t localThreadsEdgeX[] = {blockSizeX, 1};

    status = clEnqueueNDRangeKernel(commandQueue,
                                    edgeXKernel,
                                    2,
                                    NULL, //offset
                                    globalThreadsEdgeX,
                                    localThreadsEdgeX,
                                    0,
                                    NULL,
                                    &event_edgeX);

    CheckOpenCLError(status, "clEnqueueNDRangeKernel edgeX.");


   status = clWaitForEvents(1, &event_edgeX);
   CheckOpenCLError(status, "clWaitForEvents edgeX.");
//////////////////////////////////////////////////////////////////////////////////////////////////
// EDGE Y

 /* input buffer */
    status = clSetKernelArg(edgeYKernel, 
                            0, 
                            sizeof(cl_mem), 
                            &d_inputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

    /* output buffer */
    status = clSetKernelArg(edgeYKernel, 
                            1, 
                            sizeof(cl_mem), 
                            &d_edgeYImageBuffer);
	
	CheckOpenCLError(status, "clSetKernelArg. (edgeYImage)");

    /* image width */
    status = clSetKernelArg(edgeYKernel, 
                            2, 
                            sizeof(cl_uint), 
                            &width);

	CheckOpenCLError(status, "clSetKernelArg. (width)");

	/* image height */
    status = clSetKernelArg(edgeYKernel, 
                            3, 
                            sizeof(cl_uint), 
                            &height);

	CheckOpenCLError(status, "clSetKernelArg. (height)");
    

    /* local memory */
    cache_size = (blockSizeX) * (blockSizeY);
    status = clSetKernelArg(edgeYKernel, 
                            4, 
                            sizeof(cl_float4)*cache_size, 
                            0);

	CheckOpenCLError(status, "clSetKernelArg. (local cache)");
    

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
    size_t globalThreadsEdgeY[] = 
    {
        width,
        blockSizeX
    };
    size_t localThreadsEdgeY[] = {1, blockSizeX};

    status = clEnqueueNDRangeKernel(commandQueue,
                                    edgeYKernel,
                                    2,
                                    NULL, //offset
                                    globalThreadsEdgeY,
                                    localThreadsEdgeY,
                                    0,
                                    NULL,
                                    &event_edgeY);

    CheckOpenCLError(status, "clEnqueueNDRangeKernel edgeY.");

   status = clWaitForEvents(1, &event_edgeY);
   CheckOpenCLError(status, "clWaitForEvents edgeY.");

//////////////////////////////////////////////////////////////////////////////////////////////////
// RESULT KERNEL
    /* input buffer X*/
    status = clSetKernelArg(edgeResultKernel, 
                            0, 
                            sizeof(cl_mem), 
                            &d_edgeXImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (inputImage)");


    /* input buffer Y*/
    status = clSetKernelArg(edgeResultKernel, 
                            1, 
                            sizeof(cl_mem), 
                            &d_edgeYImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

    /* output buffer */
    status = clSetKernelArg(edgeResultKernel, 
                            2, 
                            sizeof(cl_mem), 
                            &d_outputImageBuffer);
	
	CheckOpenCLError(status, "clSetKernelArg. (outputImage)");

    /* image width */
    status = clSetKernelArg(edgeResultKernel, 
                            3, 
                            sizeof(cl_uint), 
                            &width);

	CheckOpenCLError(status, "clSetKernelArg. (width)");

	/* image height */
    status = clSetKernelArg(edgeResultKernel, 
                            4, 
                            sizeof(cl_uint), 
                            &height);

	CheckOpenCLError(status, "clSetKernelArg. (height)");
    
    
	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
    size_t globalThreadsResult[] = 
    {
        ((width + blockSizeX - 1)/blockSizeX) * blockSizeX,
        ((height + blockSizeY - 1)/blockSizeY) * blockSizeY
    };
    size_t localThreadsResult[] = {blockSizeX, blockSizeY};

	cl_event wait_events[] = { event_edgeX, event_edgeY };

	
    status = clEnqueueNDRangeKernel(commandQueue,
                                    edgeResultKernel,
                                    2,
                                    NULL, //offset
                                    globalThreadsResult,
                                    localThreadsResult,
                                    2,
                                    wait_events,
                                    &event_result);

    CheckOpenCLError(status, "clEnqueueNDRangeKernel.");

    status = clWaitForEvents(1, &event_result);
    CheckOpenCLError(status, "clWaitForEvents.");

	printTiming(event_edgeX, "Edge X: ");
	printTiming(event_edgeX, "Edge Y: ");
    printTiming(event_result, "Edge Result: ");

	//Read back the image - if textures were used for showing this wouldn't be necessary
	//blocking read
	status = clEnqueueReadBuffer(commandQueue,
                                d_outputImageBuffer,
                                CL_TRUE,
                                0,
                                width * height * sizeof(cl_uchar4),
                                h_outputImageData,
                                0,
                                0,
                                0);
	
    CheckOpenCLError(status, "read output.");

    return 0;
}

int cleanup()
{
	/* Releases OpenCL resources (Context, Memory etc.) */
    cl_int status;

    status = clReleaseKernel(edgeResultKernel);
    CheckOpenCLError(status, "clReleaseKernel edgeResult.");
    status = clReleaseKernel(edgeXKernel);
    CheckOpenCLError(status, "clReleaseKernel edgeX.");
    status = clReleaseKernel(edgeYKernel);
    CheckOpenCLError(status, "clReleaseKernel edgeY.");

    status = clReleaseProgram(program);
    CheckOpenCLError(status, "clReleaseProgram.");

    status = clReleaseMemObject(d_inputImageBuffer);
    CheckOpenCLError(status, "clReleaseMemObject input");
    
    status = clReleaseMemObject(d_edgeXImageBuffer);
    CheckOpenCLError(status, "clReleaseMemObject edgeX");

    status = clReleaseMemObject(d_edgeYImageBuffer);
    CheckOpenCLError(status, "clReleaseMemObject edgeY");
	
    status = clReleaseMemObject(d_outputImageBuffer);
    CheckOpenCLError(status, "clReleaseMemObject output");

    status = clReleaseCommandQueue(commandQueue);
    CheckOpenCLError(status, "clReleaseCommandQueue.");

    status = clReleaseContext(context);
    CheckOpenCLError(status, "clReleaseContext.");

    /* release program resources (input memory etc.) */
    if(h_inputImageData) 
        free(h_inputImageData);
        
    if(h_outputImageData)
        free(h_outputImageData);

    return 0;
}


int main(int argc, char* argv[])
{
	if(argc != 2)
	{
		return 1;
	}

	// Init SDL - only video subsystem will be used
    if(SDL_Init(SDL_INIT_VIDEO) < 0) throw SDL_Exception();
    // Shutdown SDL when program ends
    atexit(SDL_Quit);

#if SDL_IMAGE_PATCHLEVEL >= 10
	// load support for the JPG and PNG image formats
	int flags=IMG_INIT_JPG|IMG_INIT_PNG;
	int initted=IMG_Init(flags);
	if((initted&flags) != flags) {
		logMessage(DEBUG_LEVEL_ERROR, "IMG_Init: Failed to init required jpg and png support!");
		logMessage(DEBUG_LEVEL_ERROR, IMG_GetError());
		throw SDL_Exception();
	}
	atexit(IMG_Quit);
#endif

	//load image
	if(setupHost(argv[1]) != 0)
	{
		cleanup();
		return 1;
	}

	screen = initScreen(width, height, 24);

	mainLoop(screen);

	cleanup();

	return 0;
}


/**
 * Called after context was created
 */
void onInit()
{
	if(setupCL() != 0)
		return;
    if(runKernels() != 0)
		return;
	drawOutputImage(screen);
}

/**
 * Called when the window should be redrawn
 */
void onWindowRedraw()
{
	drawOutputImage(screen);
    SDL_UpdateRect(screen, 0, 0, 0, 0);
}

/**
 * Called when the window was resized
 * @param width The new width
 * @param height The new height
 */
void onWindowResized(int width, int height)
{
	   onWindowRedraw();
}

/**
 * Called when the key was pressed
 * @param key The key that was pressed
 * @param mod Modifiers
 */
void onKeyDown(SDLKey key, SDLMod mod)
{}

/**
 * Called when the key was released
 * @param key The key that was released
 * @param mod Modifiers
 */
void onKeyUp(SDLKey key, SDLMod mod)
{}

/**
 * Called when the mouse moves over the window
 * @param x The new x position
 * @param y The new y position
 * @param xrel Relative move from last x position
 * @param yrel Relative move from last y position
 * @param buttons Mask of the buttons that are pressed
 */
void onMouseMove(unsigned x, unsigned y, int xrel, int yrel, Uint8 buttons)
{}

/**
 * Called when a mouse button was pressed
 * @param button The button that was pressed
 * @param x The x position where the mouse was clicked
 * @param y The y position where the mouse was clicked
 */
void onMouseDown(Uint8 button, unsigned x, unsigned y)
{}

/**
 * Called when a mouse button was released
 * @param button The button that was released
 * @param x The x position where the mouse was clicked
 * @param y The y position where the mouse was clicked
 */
void onMouseUp(Uint8 button, unsigned x, unsigned y)
{}
