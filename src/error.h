#ifndef _CL_ERROR_H__
#define _CL_ERROR_H__

#include <CL/opencl.h>

const char *CLErrorString(cl_int _err);

void CheckOpenCLError(cl_int _ciErr, const char *_sMsg, ...);

#endif