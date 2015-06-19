// Author: Qi Shan <shanqi@cs.washington.edu>
// This file is for addressing compatibility issues to previous CUDA versions.
// This file also includes a few useful macros.

#ifndef CUTIL_H_
#define CUTIL_H_

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>
#include <helper_image.h>
#include <helper_math.h>
#include <helper_string.h>
#include <helper_timer.h>
#include "help.h"

#define SAFE_WRITE(fd, input, length) { ssize_t ret=write(fd, input, length); if (ret<0) {printf("File writting unsuccessful.\n");} }
#define SAFE_FREAD(content, size, count, fd) { ssize_t ret=fread(content, size, count, fd); if (ret<0) {printf("File reading unsuccessful.\n");} }


#define CUDA_SAFE_CALL(x) {checkCudaErrors((x));}

#define CUT_CHECK_ERROR(x) {fprintf(stderr, "CUT_CHECK_ERROR: %s\n", x);}

inline void cutLoadPGMf(const char* file, float** data, unsigned int* w, unsigned int* h) {
  sdkLoadPGM<float>(file, data, w, h);
}

inline void cutLoadPPM4ub(const char* file, unsigned char** data, unsigned int *w,unsigned int *h) {
  sdkLoadPPM4<unsigned char>(file, data, w, h);
  printf("Image size [%d, %d]\n", *w, *h);
}

inline void cutSavePGMub(const char* file, unsigned char* data, unsigned int w, unsigned int h) {
  sdkSavePGM<unsigned char>(file, data, w, h);
}

inline void cutSavePGMf(const char* file, float* data, unsigned int w, unsigned int h) {
  sdkSavePGM<float>(file, data, w, h);
}

#endif	// CUTIL_H_

