#define DEBUG

#include "sdlwrapper.h"
#include "error.h"
#include <stdio.h>
#include <CL/opencl.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>

#pragma comment( lib, "OpenCL" )
#pragma comment( lib, "SDL" )
#pragma comment( lib, "SDLmain" )
#pragma comment( lib, "SDL_image" )


using namespace std;

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define B_KMEANS 1
#define B_MEANSHIFT 2

//global variables

SDL_Surface *screen;

cl_uchar4* h_inputImageData = NULL;
cl_uchar4* h_outputImageData = NULL;

int algorithm; //used algorithm

//width and height of the image
int width = 0, height = 0;
int K = 4;

cl_uint pixelSize = 32; //rgba 8bits per channel

//opencl stuff
cl_context context;
cl_command_queue commandQueue;
cl_kernel kmeans;
cl_kernel meanshiftBeginKernel, meanshiftPeaksKernel, meanshiftResultKernel, meanshift;
cl_kernel msBegin, msOptimize, msColor;
//cl_kernel edgeXKernel, edgeYKernel, kmeansResultKernel;
//cl_kernel meanshiftKernel, meanshiftPeaksKernel, meanshiftResultKernel;
//cl_kernel msKernel, msResult;
cl_program program;


/** CL memory buffer for images */
cl_mem d_inputImageBuffer = NULL;
cl_mem d_outputImageBuffer = NULL;


/* k-means memory buffers */
cl_mem d_pixels = NULL;
cl_mem d_centroids = NULL;

/* mean-shift memory buffers */
cl_mem d_countsBuffer = NULL;
cl_mem d_peaksBuffer = NULL;
cl_mem d_colors = NULL;
cl_mem d_peakCount = NULL;
cl_mem d_uniquePeaks = NULL;


//the size of our blocks

/**
 * TODO preferovana velikost
 */
size_t kernelWorkGroupSize = 1024;
size_t blockSizeX = 1024;
size_t blockSizeY = 1;

/* Size of mean-shift window */
int msWinSize = 5;

/* Maximal length of mean-shift */
float msMaxLength = 10.0f;

// nahodne zvoleni K stredu

void generateCenters(int K, cl_uchar4* centers)
{
    for (int i = 0; i < K; i++)
    {
        centers->s[0] = rand() % 256;
        centers->s[1] = rand() % 256;
        centers->s[2] = rand() % 256;
        centers->s[3] = 255;
    }
}


int printTiming(cl_event event, const char* title)
{
    cl_ulong startTime;
    cl_ulong endTime;
    /* Display proiling info */
    cl_int status = clGetEventProfilingInfo(event,
                                            CL_PROFILING_COMMAND_START,
                                            sizeof (cl_ulong),
                                            &startTime,
                                            0);
    CheckOpenCLError(status, "clGetEventProfilingInfo.(startTime)");


    status = clGetEventProfilingInfo(event,
                                     CL_PROFILING_COMMAND_END,
                                     sizeof (cl_ulong),
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
    if (pFileStream == 0)
    {
        return NULL;
    }


    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END);
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET);

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *) malloc(szSourceLength + 1);
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
int drawOutputImage(SDL_Surface *screen)
{


    SDL_Surface *temp = SDL_CreateRGBSurfaceFrom(h_outputImageData,
                                                 width, height, pixelSize, width * 4,
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

    if (readImage(inputImageName, &inputImage) < 0)
    {
        return -1;
    }

    width = inputImage->w;
    height = inputImage->h;

    h_inputImageData = (cl_uchar4*) malloc(width * height * sizeof (cl_uchar4));

    if (h_inputImageData == NULL)
    {
        logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory.");
        return -1;
    }

    memcpy(h_inputImageData, inputImage->pixels, width * height * sizeof (cl_uchar4));

    //allocate output image

    h_outputImageData = (cl_uchar4 *) malloc(width * height * sizeof (cl_uchar4));

    if (h_outputImageData == NULL)
    {
        logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory.");
        return -1;
    }

    memset(h_outputImageData, 0, width * height * sizeof (cl_uchar4));

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
    ciErr = clGetPlatformIDs(0, NULL, &cuiPlatformsCount);
    CheckOpenCLError(ciErr, "clGetPlatformIDs: cuiPlatformsNum=%i", cuiPlatformsCount);
    cpPlatforms = (cl_platform_id*) malloc(cuiPlatformsCount * sizeof (cl_platform_id));
    ciErr = clGetPlatformIDs(cuiPlatformsCount, cpPlatforms, NULL);
    CheckOpenCLError(ciErr, "clGetPlatformIDs");

    cl_platform_id platform = 0;

    const unsigned int TMP_BUFFER_SIZE = 1024;
    char sTmp[TMP_BUFFER_SIZE];

    for (unsigned int f0 = 0; f0 < cuiPlatformsCount; f0++)
    {
        //bool shouldBrake = false;
        ciErr = clGetPlatformInfo(cpPlatforms[f0], CL_PLATFORM_PROFILE, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_PROFILE=%s", f0, sTmp);
        ciErr = clGetPlatformInfo(cpPlatforms[f0], CL_PLATFORM_VERSION, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_VERSION=%s", f0, sTmp);
        ciErr = clGetPlatformInfo(cpPlatforms[f0], CL_PLATFORM_NAME, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_NAME=%s", f0, sTmp);
        ciErr = clGetPlatformInfo(cpPlatforms[f0], CL_PLATFORM_VENDOR, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_VENDOR=%s", f0, sTmp);

        //prioritize AMD and CUDA platforms

        if ((strcmp(sTmp, "Advanced Micro Devices, Inc.") == 0) || (strcmp(sTmp, "NVIDIA Corporation") == 0))
        {
            platform = cpPlatforms[f0];
        }

        //prioritize Intel
        /*if ((strcmp(sTmp, "Intel(R) Corporation") == 0)) {
            platform = cpPlatforms[f0];
        }*/

        ciErr = clGetPlatformInfo(cpPlatforms[f0], CL_PLATFORM_EXTENSIONS, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_EXTENSIONS=%s", f0, sTmp);
        printf("\n");
    }

    if (platform == 0)
    { //no prioritized found
        if (cuiPlatformsCount > 0)
        {
            platform = cpPlatforms[0];
        }
        else
        {
            logMessage(DEBUG_LEVEL_ERROR, "No device was found");
            return -1;
        }
    }
    // Get Devices
    cl_uint cuiDevicesCount;
    ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &cuiDevicesCount);
    CheckOpenCLError(ciErr, "clGetDeviceIDs: cuiDevicesCount=%i", cuiDevicesCount);
    cl_device_id *cdDevices = (cl_device_id*) malloc(cuiDevicesCount * sizeof (cl_device_id));
    ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, cuiDevicesCount, cdDevices, NULL);
    CheckOpenCLError(ciErr, "clGetDeviceIDs");

    unsigned int deviceIndex = 0;

    for (unsigned int f0 = 0; f0 < cuiDevicesCount; f0++)
    {
        cl_device_type cdtTmp;
        size_t iDim[3];

        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DEVICE_TYPE, sizeof (cdtTmp), &cdtTmp, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_TYPE=%s%s%s%s", f0, cdtTmp & CL_DEVICE_TYPE_CPU ? "CPU," : "",
                         cdtTmp & CL_DEVICE_TYPE_GPU ? "GPU," : "",
                         cdtTmp & CL_DEVICE_TYPE_ACCELERATOR ? "ACCELERATOR," : "",
                         cdtTmp & CL_DEVICE_TYPE_DEFAULT ? "DEFAULT," : "");

        if (cdtTmp & CL_DEVICE_TYPE_CPU)
        { //prioritize gpu if both cpu and gpu are available
            deviceIndex = f0;
        }

        cl_bool bTmp;
        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DEVICE_AVAILABLE, sizeof (bTmp), &bTmp, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_AVAILABLE=%s", f0, bTmp ? "YES" : "NO");
        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DEVICE_NAME, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_NAME=%s", f0, sTmp);
        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DEVICE_VENDOR, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_VENDOR=%s", f0, sTmp);
        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DRIVER_VERSION, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DRIVER_VERSION=%s", f0, sTmp);
        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DEVICE_PROFILE, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_PROFILE=%s", f0, sTmp);
        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DEVICE_VERSION, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_VERSION=%s", f0, sTmp);
        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof (iDim), iDim, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_MAX_WORK_ITEM_SIZES=%ix%ix%i", f0, iDim[0], iDim[1], iDim[2]);
        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (size_t), iDim, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_MAX_WORK_GROUP_SIZE=%i", f0, iDim[0]);
        ciErr = clGetDeviceInfo(cdDevices[f0], CL_DEVICE_EXTENSIONS, TMP_BUFFER_SIZE, sTmp, NULL);
        CheckOpenCLError(ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_EXTENSIONS=%s", f0, sTmp);
        printf("\n");
    }

    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) platform,
        0
    };



    //create context
    context = clCreateContext(cps, 1, &cdDevices[deviceIndex], NULL, NULL, &ciErr);
    CheckOpenCLError(ciErr, "clCreateContext");
    //may use clCreateContextFromType than choose a device based on the returned devices

    //create a command queue
    commandQueue = clCreateCommandQueue(context, cdDevices[deviceIndex],
                                        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ciErr);
    CheckOpenCLError(ciErr, "clCreateCommandQueue");

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
                                 width * height * sizeof (cl_uchar4),
                                 h_inputImageData,
                                 0,
                                 0,
                                 0);

    CheckOpenCLError(ciErr, "Copy input image data");


    //output image buffer - write only
    d_outputImageBuffer = clCreateBuffer(context,
                                         CL_MEM_WRITE_ONLY,
                                         width * height * pixelSize,
                                         0,
                                         &ciErr);
    CheckOpenCLError(ciErr, "Allocate output buffer");

    ciErr = clEnqueueWriteBuffer(commandQueue,
                                 d_outputImageBuffer,
                                 CL_TRUE, //blocking write
                                 0,
                                 width * height * sizeof (cl_uchar4),
                                 h_inputImageData,
                                 0,
                                 0,
                                 0);

    CheckOpenCLError(ciErr, "Copy output image data");


    //create mid buffers dependind on algorithm
    if (algorithm == B_KMEANS)
    {
        /* K-means section */

        /* Set all mid buffers needed by k-means here */
        d_pixels = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE,
                                  width * height * sizeof (cl_uint), // ke kazdemu pixelu staci uchovat cislo clusteru
                                  0, &ciErr);
        CheckOpenCLError(ciErr, "CreateBuffer pixels (k-means)");

        d_centroids = clCreateBuffer(context,
                                     CL_MEM_READ_WRITE,
                                     K * sizeof (cl_uchar4), // K centroidu, u kazdeho RGB
                                     0, &ciErr);
        CheckOpenCLError(ciErr, "CreateBuffer centroids (k-means)");

        // nahodne vybrani K stredu a zkopirovani do bufferu
        cl_uchar4 *centers = new cl_uchar4[K];
        generateCenters(K, centers);
        ciErr = clEnqueueWriteBuffer(commandQueue,
                                     d_centroids,
                                     CL_TRUE, //blocking write
                                     0,
                                     K * sizeof (cl_uchar4),
                                     centers,
                                     0,
                                     0,
                                     0);

        CheckOpenCLError(ciErr, "Copy centroids buffer data (k-means)");
        delete centers;
    }
    else
    {
        /* Mean-shift section */
        /* Set all mid buffers needed by mean-shift here */
        d_peaksBuffer = clCreateBuffer(context,
                                       CL_MEM_READ_WRITE,
                                       width * height * sizeof (cl_uint),
                                       0,
                                       &ciErr);
        CheckOpenCLError(ciErr, "CreateBuffer peaks (mean-shift)");

        d_countsBuffer = clCreateBuffer(context,
                                        CL_MEM_READ_WRITE,
                                        width * height * sizeof (cl_uint),
                                        0,
                                        &ciErr);
        CheckOpenCLError(ciErr, "CreateBuffer counts (mean-shift)");

        d_peakCount = clCreateBuffer(context,
                                     CL_MEM_READ_WRITE,
                                     sizeof (cl_uint),
                                     0,
                                     &ciErr);
        CheckOpenCLError(ciErr, "CreateBuffer peakCounts (mean-shift)");

        cl_uint peakCount = -1;
        ciErr = clEnqueueWriteBuffer(commandQueue,
                                     d_peakCount,
                                     CL_TRUE, //blocking write
                                     0,
                                     sizeof (cl_uint),
                                     &peakCount,
                                     0,
                                     0,
                                     0);

        CheckOpenCLError(ciErr, "Copy centroids buffer data (k-means)");

        d_uniquePeaks = clCreateBuffer(context,
                                       CL_MEM_READ_WRITE,
                                       width * height * sizeof (cl_uint),
                                       0,
                                       &ciErr);
        CheckOpenCLError(ciErr, "CreateBuffer uniquePeaks (mean-shift)");

        d_colors = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE,
                                  width * height * sizeof (cl_uchar4),
                                  0,
                                  &ciErr);
        CheckOpenCLError(ciErr, "CreateBuffer colors (mean-shift)");
    }


    //=================================================================================
    // Create and compile and openCL program

    char *cSourceCL = loadProgSource("kernels.cl");

    program = clCreateProgramWithSource(context, 1, (const char **) &cSourceCL, NULL, &ciErr);
    CheckOpenCLError(ciErr, "clCreateProgramWithSource");
    free(cSourceCL);

    ciErr = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    cl_int logStatus;

    //build log
    char *buildLog = NULL;
    size_t buildLogSize = 0;
    logStatus = clGetProgramBuildInfo(program,
                                      cdDevices[deviceIndex],
                                      CL_PROGRAM_BUILD_LOG,
                                      buildLogSize,
                                      buildLog,
                                      &buildLogSize);

    CheckOpenCLError(logStatus, "clGetProgramBuildInfo.");

    buildLog = (char*) malloc(buildLogSize);
    if (buildLog == NULL)
    {
        printf("Failed to allocate host memory. (buildLog)");
        return -1;
    }
    memset(buildLog, 0, buildLogSize);

    logStatus = clGetProgramBuildInfo(program,
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

    CheckOpenCLError(ciErr, "clBuildProgram");


    size_t tempKernelWorkGroupSize;

    if (algorithm == B_KMEANS)
    {
        /* ================================================================== */
        /* K-means section */
        /* ================================================================== */

        // kernels - create kernels
        kmeans = clCreateKernel(program, "kmeans", &ciErr);
        CheckOpenCLError(ciErr, "clCreateKernel kmeans");
        //edgeXKernel = clCreateKernel(program, "edge_x", &ciErr);
        //CheckOpenCLError(ciErr, "clCreateKernel edge_x");
        //edgeYKernel = clCreateKernel(program, "edge_y", &ciErr);
        //CheckOpenCLError(ciErr, "clCreateKernel edge_y");

        // Check group size against group size returned by kernel
        ciErr = clGetKernelWorkGroupInfo(kmeans,
                                         cdDevices[deviceIndex],
                                         CL_KERNEL_WORK_GROUP_SIZE,
                                         sizeof (size_t),
                                         &tempKernelWorkGroupSize,
                                         0);
        CheckOpenCLError(ciErr, "clGetKernelInfo");
        kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);

        /*ciErr = clGetKernelWorkGroupInfo(kmeansPixelAssignKernel,
                                         cdDevices[deviceIndex],
                                         CL_KERNEL_WORK_GROUP_SIZE,
                                         sizeof (size_t),
                                         &tempKernelWorkGroupSize,
                                         0);
        CheckOpenCLError(ciErr, "clGetKernelInfo");
        kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);

        ciErr = clGetKernelWorkGroupInfo(edgeYKernel,
                                         cdDevices[deviceIndex],
                                         CL_KERNEL_WORK_GROUP_SIZE,
                                         sizeof (size_t),
                                         &tempKernelWorkGroupSize,
                                         0);
        CheckOpenCLError(ciErr, "clGetKernelInfo");
        kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);*/
    }
    else
    {
        /* ================================================================== */
        /* Mean-shift section */
        /* ================================================================== */

        // kernels - create kernels
        //        meanshift = clCreateKernel(program, "meanshift", &ciErr);
        //        CheckOpenCLError(ciErr, "clCreateKernel meanshift");
        //
        //        // Check group size against group size returned by kernel
        //        ciErr = clGetKernelWorkGroupInfo(meanshift,
        //
        //                                         cdDevices[deviceIndex],
        //                                         CL_KERNEL_WORK_GROUP_SIZE,
        //                                         sizeof (size_t),
        //                                         &tempKernelWorkGroupSize,
        //                                         0);
        //        CheckOpenCLError(ciErr, "clGetKernelInfo");
        //        kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);

        msBegin = clCreateKernel(program, "ms_begin", &ciErr);
        CheckOpenCLError(ciErr, "clCreateKernel ms_begin");

        // Check group size against group size returned by kernel
        ciErr = clGetKernelWorkGroupInfo(msBegin,
                cdDevices[deviceIndex],
                CL_KERNEL_WORK_GROUP_SIZE,
                sizeof (size_t),
                &tempKernelWorkGroupSize,
                0);
        CheckOpenCLError(ciErr, "clGetKernelInfo");
        kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);

        msOptimize = clCreateKernel(program, "ms_optimize", &ciErr);
        CheckOpenCLError(ciErr, "clCreateKernel ms_optimize");

        // Check group size against group size returned by kernel
        ciErr = clGetKernelWorkGroupInfo(msOptimize,
                cdDevices[deviceIndex],
                CL_KERNEL_WORK_GROUP_SIZE,
                sizeof (size_t),
                &tempKernelWorkGroupSize,
                0);
        CheckOpenCLError(ciErr, "clGetKernelInfo");
        kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);

        msColor = clCreateKernel(program, "ms_color", &ciErr);
        CheckOpenCLError(ciErr, "clCreateKernel ms_color");

        // Check group size against group size returned by kernel
        ciErr = clGetKernelWorkGroupInfo(msColor,
                cdDevices[deviceIndex],
                CL_KERNEL_WORK_GROUP_SIZE,
                sizeof (size_t),
                &tempKernelWorkGroupSize,
                0);
        CheckOpenCLError(ciErr, "clGetKernelInfo");
        kernelWorkGroupSize = MIN(tempKernelWorkGroupSize, kernelWorkGroupSize);
    }


    if ((blockSizeX * blockSizeY) > kernelWorkGroupSize)
    {
        printf("Out of Resources!\n");
        printf("Group Size specified: %i\n", blockSizeX * blockSizeY);
        printf("Max Group Size supported on the kernel: %i\n", kernelWorkGroupSize);
        printf("Falling back to %i.\n", kernelWorkGroupSize);

        if (blockSizeX > kernelWorkGroupSize)
        {
            blockSizeX = kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }

    free(cdDevices);

    return 0;
}

/**
 * This function runs kernels for k-means algorithm
 *
 * @return Zero if pass
 */
int runKMeansKernels()
{
    int status;
    cl_event event_kmeans;

    /* Setup arguments to the kernel */

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Kernel kmeans

    /* input buffer */
    status = clSetKernelArg(kmeans,
                            0,
                            sizeof (cl_mem),
                            &d_inputImageBuffer);
    CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

    /* output buffer */
    status = clSetKernelArg(kmeans,
                            1,
                            sizeof (cl_mem),
                            &d_outputImageBuffer);
    CheckOpenCLError(status, "clSetKernelArg. (uotputImage)");

    /* buffer centroidu */
    status = clSetKernelArg(kmeans,
                            2,
                            sizeof (cl_mem),
                            &d_centroids);
    CheckOpenCLError(status, "clSetKernelArg. (centroids)");

    /* image pixelu a jejich naleyitosti k centroidum */
    status = clSetKernelArg(kmeans,
                            3,
                            sizeof (cl_mem),
                            &d_pixels);

    CheckOpenCLError(status, "clSetKernelArg. (pixels)");

    /* image width */
    status = clSetKernelArg(kmeans,
                            4,
                            sizeof (cl_uint),
                            &width);

    CheckOpenCLError(status, "clSetKernelArg. (width)");

    /* image height */
    status = clSetKernelArg(kmeans,
                            5,
                            sizeof (cl_uint),
                            &height);

    CheckOpenCLError(status, "clSetKernelArg. (height)");

    /* K */
    status = clSetKernelArg(kmeans,
                            6,
                            sizeof (cl_uint),
                            &K);

    CheckOpenCLError(status, "clSetKernelArg. (K)");

    /* local memory */
    /*int cache_size = (blockSizeX) * (blockSizeY);
    status = clSetKernelArg(kmeansPixelAssignKernel,
                            5,
                            sizeof (cl_float4) * cache_size,
                            0);

    CheckOpenCLError(status, "clSetKernelArg. (local cache) %u", sizeof (cl_float4) * cache_size);*/

    //the global number of threads in each dimension has to be divisible
    // by the local dimension numbers
    size_t globalThreadsPixels[] = {
        /*width, */blockSizeX,
        height
    };
    size_t localThreadsPixels[] = {/*width, */blockSizeX, 1};

    status = clEnqueueNDRangeKernel(commandQueue,
                                    kmeans,
                                    2,
                                    NULL, //offset
                                    globalThreadsPixels,
                                    /*NULL, */localThreadsPixels,
                                    0,
                                    NULL,
                                    &event_kmeans);

    CheckOpenCLError(status, "clEnqueueNDRangeKernel kmeans.");


    status = clWaitForEvents(1, &event_kmeans);
    CheckOpenCLError(status, "clWaitForEvents K-means.");
    //////////////////////////////////////////////////////////////////////////////////////////////////
    printTiming(event_kmeans, "K-means Result: ");

    //Read back the image - if textures were used for showing this wouldn't be necessary
    //blocking read
    status = clEnqueueReadBuffer(commandQueue,
                                 d_outputImageBuffer,
                                 CL_TRUE,
                                 0,
                                 width * height * sizeof (cl_uchar4),
                                 h_outputImageData,
                                 0,
                                 0,
                                 0);

    CheckOpenCLError(status, "read output.");

    return 0;
}

/**
 * This function runs kernels for mean-shift algorithm
 *
 * @return Zero if pass
 */
int runMeanShiftKernels()
{
    int status;
    //cl_event event_result, event_meanshift, event_peaks;
    //cl_event event_meanshift;
    cl_event event_begin, event_optimize, event_color, event_readPeaks, event_writeColors;

    /* Setup arguments to the kernel */

    /* input buffer */
    //    status = clSetKernelArg(meanshift,
    //                            0,
    //                            sizeof (cl_mem),
    //                            &d_inputImageBuffer);
    //    CheckOpenCLError(status, "clSetKernelArg. (inputImage)");
    //
    //    /* output buffer */
    //    status = clSetKernelArg(meanshift,
    //                            1,
    //                            sizeof (cl_mem),
    //                            &d_outputImageBuffer);
    //    CheckOpenCLError(status, "clSetKernelArg. (outputImage)");
    //
    //    /* peaks buffer */
    //    status = clSetKernelArg(meanshift,
    //                            2,
    //                            sizeof (cl_mem),
    //                            &d_peaksBuffer);
    //    CheckOpenCLError(status, "clSetKernelArg. (peaksBuffer)");
    //
    //    /* counts buffer */
    //    status = clSetKernelArg(meanshift,
    //                            3,
    //                            sizeof (cl_mem),
    //                            &d_countsBuffer);
    //    CheckOpenCLError(status, "clSetKernelArg. (countsBuffer)");
    //
    //    /* image width */
    //    status = clSetKernelArg(meanshift,
    //                            4,
    //                            sizeof (cl_uint),
    //                            &width);
    //    CheckOpenCLError(status, "clSetKernelArg. (width)");
    //
    //    /* image height */
    //    status = clSetKernelArg(meanshift,
    //                            5,
    //                            sizeof (cl_uint),
    //                            &height);
    //    CheckOpenCLError(status, "clSetKernelArg. (height)");
    //
    //    /* window size */
    //    status = clSetKernelArg(meanshift,
    //                            6,
    //                            sizeof (cl_uint),
    //                            &msWinSize);
    //    CheckOpenCLError(status, "clSetKernelArg. (msWinSize)");
    //
    //    /* peakCount buffer */
    //
    //    status = clSetKernelArg(meanshift,
    //                            7,
    //                            sizeof (cl_mem),
    //                            &d_peakCount);
    //    CheckOpenCLError(status, "clSetKernelArg. (peakCount)");
    //
    //    /* uniquePeaks buffer */
    //    status = clSetKernelArg(meanshift,
    //                            8,
    //                            sizeof (cl_mem),
    //                            &d_uniquePeaks);
    //    CheckOpenCLError(status, "clSetKernelArg. (uniquePeaks)");
    //
    //    /* colors buffer */
    //    status = clSetKernelArg(meanshift,
    //                            9,
    //                            sizeof (cl_mem),
    //                            &d_colors);
    //    CheckOpenCLError(status, "clSetKernelArg. (colors)");
    //
    //    /* Kernel enqueue */
    //    size_t globalThreadsMeanshift[] = {width, height};
    //    size_t localThreadsMeanshift[] = {width, 1};
    //
    //    status = clEnqueueNDRangeKernel(commandQueue,
    //                                    meanshift,
    //                                    2,
    //                                    NULL, //offset
    //                                    globalThreadsMeanshift,
    //            localThreadsMeanshift,
    //                                    0,
    //                                    NULL,
    //                                    &event_meanshift);
    //
    //    CheckOpenCLError(status, "clEnqueueNDRangeKernel meanshift.");
    //
    //    status = clWaitForEvents(1, &event_meanshift);
    //    CheckOpenCLError(status, "clWaitForEvents meanshift.");

    status = clSetKernelArg(msBegin,
            0,
            sizeof (cl_mem),
            &d_inputImageBuffer);
    CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

    /* peaks buffer */
    status = clSetKernelArg(msBegin,
            1,
            sizeof (cl_mem),
            &d_peaksBuffer);
    CheckOpenCLError(status, "clSetKernelArg. (peaksBuffer)");


    /* image width */
    status = clSetKernelArg(msBegin,
            2,
            sizeof (cl_uint),
            &width);
    CheckOpenCLError(status, "clSetKernelArg. (width)");

    /* image height */
    status = clSetKernelArg(msBegin,
            3,
            sizeof (cl_uint),
            &height);
    CheckOpenCLError(status, "clSetKernelArg. (height)");

    /* window size */
    status = clSetKernelArg(msBegin,
            4,
            sizeof (cl_uint),
            &msWinSize);
    CheckOpenCLError(status, "clSetKernelArg. (msWinSize)");

    /* Kernel enqueue */
    size_t globalThreadsMSBegin[] = {width, height};
    size_t localThreadsMSBegin[] = {width, 1};

    status = clEnqueueNDRangeKernel(commandQueue,
            msBegin,
            2,
            NULL, //offset
            globalThreadsMSBegin,
            localThreadsMSBegin,
            0,
            NULL,
            &event_begin);

    CheckOpenCLError(status, "clEnqueueNDRangeKernel meanshift.");

    status = clWaitForEvents(1, &event_begin);
    CheckOpenCLError(status, "clWaitForEvents meanshift.");

    ///////////////////////////////////////////////////////////////////////////
    // msOptimize

    /* peaks buffer */
    status = clSetKernelArg(msOptimize,
            0,
            sizeof (cl_mem),
            &d_peaksBuffer);
    CheckOpenCLError(status, "clSetKernelArg. (peaksBuffer)");


    /* image width */
    status = clSetKernelArg(msOptimize,
            1,
            sizeof (cl_uint),
            &width);
    CheckOpenCLError(status, "clSetKernelArg. (width)");

    /* Kernel enqueue */
    cl_event wait_event[] = {event_begin};
    size_t globalThreadsMSOptimize[] = {width, height};
    size_t localThreadsMSOptimize[] = {width, 1};

    status = clEnqueueNDRangeKernel(commandQueue,
            msOptimize,
            2,
            NULL, //offset
            globalThreadsMSOptimize,
            localThreadsMSOptimize,
            1,
            wait_event,
            &event_optimize);

    CheckOpenCLError(status, "clEnqueueNDRangeKernel msOptimize.");

    status = clWaitForEvents(1, &event_optimize);
    CheckOpenCLError(status, "clWaitForEvents msOptimize.");

    ///////////////////////////////////////////////////////////////////////////
    // CPU part - count peaks and choose colors
    cl_uint *peaks = new cl_uint[width * height];

    status = clEnqueueReadBuffer(commandQueue,
            d_peaksBuffer,
            CL_TRUE,
            0,
            width * height * sizeof (cl_uint),
            peaks,
            0,
            0,
            &event_readPeaks);

    CheckOpenCLError(status, "read peaksBuffer.");

    status = clWaitForEvents(1, &event_readPeaks);
    CheckOpenCLError(status, "clWaitForEvents readPeaks.");

    /* Count unique peaks */
    vector<uint> uniquePeaks;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int actpeak = peaks[x + width * y];

            /* Search the array for actpeak */
            int i;
            for (i = 0; i < uniquePeaks.size(); i++) {
                if (uniquePeaks[i] == actpeak) {
                    break;
                }
            }

            /* If went through all the elements = haven't found */
            if (i == uniquePeaks.size()) {
                uniquePeaks.push_back(actpeak);
            }
        }
    }
    cout << uniquePeaks.size() << endl;

    cl_uchar4 *colors = new cl_uchar4[width * height];

    /* Choose colors */
    int colorStep = (256 * 256 * 256) / uniquePeaks.size();
    for (int n = 0; n < uniquePeaks.size(); n++) {

        int actstep = n*colorStep;
        colors[uniquePeaks[n]].s0 = (actstep / 65536) % 256;
        colors[uniquePeaks[n]].s1 = (actstep / 256) % 256;
        colors[uniquePeaks[n]].s2 = actstep % 256;
        colors[uniquePeaks[n]].s3 = 255;
    }

    status = clEnqueueWriteBuffer(commandQueue,
            d_colors,
            CL_TRUE, //blocking write
            0,
            uniquePeaks.size() * sizeof (cl_uchar4),
            colors,
            0,
            0,
            &event_writeColors);

    CheckOpenCLError(status, "Copy unique colors buffer data");

    status = clWaitForEvents(1, &event_writeColors);
    CheckOpenCLError(status, "clWaitForEvents writeColors.");

    delete [] colors;
    delete [] peaks;
    ///////////////////////////////////////////////////////////////////////////
    // msColor

    /* output buffer */
    status = clSetKernelArg(msColor,
            0,
            sizeof (cl_mem),
            &d_outputImageBuffer);
    CheckOpenCLError(status, "clSetKernelArg. (outputImageBuffer)");


    /* peaks buffer */
    status = clSetKernelArg(msColor,
            1,
            sizeof (cl_mem),
            &d_colors);
    CheckOpenCLError(status, "clSetKernelArg. (colorsBuffer)");

    /* peaks buffer */
    status = clSetKernelArg(msColor,
            2,
            sizeof (cl_mem),
            &d_peaksBuffer);
    CheckOpenCLError(status, "clSetKernelArg. (peaksBuffer)");

    /* image width */
    status = clSetKernelArg(msColor,
            3,
            sizeof (cl_uint),
            &width);
    CheckOpenCLError(status, "clSetKernelArg. (width)");

    /* Kernel enqueue */
    cl_event wait_event2[] = {event_optimize};
    size_t globalThreadsMSColor[] = {width, height};
    size_t localThreadsMSColor[] = {width, 1};

    status = clEnqueueNDRangeKernel(commandQueue,
            msColor,
            2,
            NULL, //offset
            globalThreadsMSColor,
            localThreadsMSColor,
            1,
            wait_event2,
            &event_color);

    CheckOpenCLError(status, "clEnqueueNDRangeKernel msColor.");

    status = clWaitForEvents(1, &event_color);
    CheckOpenCLError(status, "clWaitForEvents msColor.");


    printTiming(event_begin, "msBegin: ");
    printTiming(event_optimize, "msOptimize: ");
    printTiming(event_color, "msColor: ");

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // mean-shift

    //Read back the image - if textures were used for showing this wouldn't be necessary
    //blocking read
    status = clEnqueueReadBuffer(commandQueue,
                                 d_outputImageBuffer,
                                 CL_TRUE,
                                 0,
                                 width * height * sizeof (cl_uchar4),
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

    if (algorithm == B_KMEANS)
    {
        /* K-means section */
        status = clReleaseKernel(kmeans);
        CheckOpenCLError(status, "clReleaseKernel kmeans.");

        status = clReleaseMemObject(d_centroids);
        CheckOpenCLError(status, "clReleaseMemObject centroids");
        status = clReleaseMemObject(d_pixels);
        CheckOpenCLError(status, "clReleaseMemObject pixels");
    }
    else
    {
        /* Mean-shift section */
        //        status = clReleaseKernel(meanshift);
        //        CheckOpenCLError(status, "clReleaseKernel meanshift.");
        status = clReleaseKernel(msBegin);
        CheckOpenCLError(status, "clReleaseKernel msBegin.");
        status = clReleaseKernel(msOptimize);
        CheckOpenCLError(status, "clReleaseKernel msBegin.");
        status = clReleaseKernel(msColor);
        CheckOpenCLError(status, "clReleaseKernel msBegin.");

        status = clReleaseMemObject(d_countsBuffer);
        CheckOpenCLError(status, "clReleaseMemObject countsBuffer");
        status = clReleaseMemObject(d_peaksBuffer);
        CheckOpenCLError(status, "clReleaseMemObject peaksBuffer");
        status = clReleaseMemObject(d_colors);
        CheckOpenCLError(status, "clReleaseMemObject colors");
        status = clReleaseMemObject(d_uniquePeaks);
        CheckOpenCLError(status, "clReleaseMemObject uniquePeaks");
        status = clReleaseMemObject(d_peakCount);
        CheckOpenCLError(status, "clReleaseMemObject peakCount");
    }

    status = clReleaseProgram(program);
    CheckOpenCLError(status, "clReleaseProgram.");

    status = clReleaseMemObject(d_inputImageBuffer);
    CheckOpenCLError(status, "clReleaseMemObject input");

    status = clReleaseMemObject(d_outputImageBuffer);
    CheckOpenCLError(status, "clReleaseMemObject output");

    status = clReleaseCommandQueue(commandQueue);
    CheckOpenCLError(status, "clReleaseCommandQueue.");

    status = clReleaseContext(context);
    CheckOpenCLError(status, "clReleaseContext.");

    /* release program resources (input memory etc.) */
    if (h_inputImageData)
        free(h_inputImageData);

    if (h_outputImageData)
        free(h_outputImageData);

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        cerr << "Nedostatecny pocet parametru!" << endl;
        return 1;
    }

    if (string(argv[1]) == "km")
        algorithm = B_KMEANS;
    else if (string(argv[1]) == "ms")
        algorithm = B_MEANSHIFT;
    else
    {
        cerr << "Nerozpoznany parametr: " << argv[1] << endl;
        cerr << "Pouzijte 'ms' pro mean-shift nebo 'km' pro k-means." << endl;
        return 1;
    }

    // Init SDL - only video subsystem will be used
    if (SDL_Init(SDL_INIT_VIDEO) < 0) throw SDL_Exception();
    // Shutdown SDL when program ends
    atexit(SDL_Quit);

#if SDL_IMAGE_PATCHLEVEL >= 10
    // load support for the JPG and PNG image formats
    int flags = IMG_INIT_JPG | IMG_INIT_PNG;
    int initted = IMG_Init(flags);
    if ((initted & flags) != flags)
    {
        logMessage(DEBUG_LEVEL_ERROR, "IMG_Init: Failed to init required jpg and png support!");
        logMessage(DEBUG_LEVEL_ERROR, IMG_GetError());
        throw SDL_Exception();
    }
    atexit(IMG_Quit);
#endif

    //load image
    if (setupHost(argv[2]) != 0)
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
    if (setupCL() != 0)
        return;
    if (algorithm == B_KMEANS)
    {
        if (runKMeansKernels() != 0)
            return;
    }
    else
    {
        if (runMeanShiftKernels() != 0)
            return;
    }
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
{
    switch (key)
    {
    case SDLK_q:
    case SDLK_x:
    case SDLK_ESCAPE:
        quit();
    }
}

/**
 * Called when the key was released
 * @param key The key that was released
 * @param mod Modifiers
 */
void onKeyUp(SDLKey key, SDLMod mod)
{
}

/**
 * Called when the mouse moves over the window
 * @param x The new x position
 * @param y The new y position
 * @param xrel Relative move from last x position
 * @param yrel Relative move from last y position
 * @param buttons Mask of the buttons that are pressed
 */
void onMouseMove(unsigned x, unsigned y, int xrel, int yrel, Uint8 buttons)
{
}

/**
 * Called when a mouse button was pressed
 * @param button The button that was pressed
 * @param x The x position where the mouse was clicked
 * @param y The y position where the mouse was clicked
 */
void onMouseDown(Uint8 button, unsigned x, unsigned y)
{
}

/**
 * Called when a mouse button was released
 * @param button The button that was released
 * @param x The x position where the mouse was clicked
 * @param y The y position where the mouse was clicked
 */
void onMouseUp(Uint8 button, unsigned x, unsigned y)
{
}
