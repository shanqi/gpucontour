// Edited: Qi Shan <shanqi@cs.washington.edu>

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>
#include <fcntl.h>
#include <float.h>
#include <unistd.h>
#include "cutil.h"
#include "texton.h"
#include "convert.h"
#include "intervening.h"
#include "lanczos.h"
#include "stencilMVM.h"

#include "localcues.h"
#include "combine.h"
#include "nonmax.h"
#include "spectralPb.h"
#include "globalPb.h"
#include "skeleton.h"

#define __TIMER_SPECFIC

#define TEXTON64 2
#define TEXTON32 1

float* loadArray(char* filename, uint& width, uint& height) {
  FILE* fp;
  fp = fopen(filename, "r");
  int dim;
  SAFE_FREAD(&dim, sizeof(int), 1, fp);
  assert(dim == 2);
  SAFE_FREAD(&width, sizeof(int), 1, fp);
  SAFE_FREAD(&height, sizeof(int), 1, fp);
  float* buffer = (float*)malloc(sizeof(float) * width * height);
  int counter = 0;
  for(int col = 0; col < width; col++) {
    for(int row = 0; row < height; row++) {
      float element;
      SAFE_FREAD(&element, sizeof(float), 1, fp);
      counter++;
      buffer[row * width + col] = element;
    }
  }
  return buffer;
}

void writeTextImage(const char* filename, uint width, uint height, float* image) {
  FILE* fp = fopen(filename, "w");
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      fprintf(fp, "%f ", image[row * width + col]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void writeFile(char* file, int width, int height, int* input) {
    int fd;
    float* pb = (float*)malloc(sizeof(float)*width*height);
    for(int i = 0; i < width * height; i++) {
      pb[i] = (float)input[i];
    }
    fd = open(file, O_CREAT|O_WRONLY, 0666);
    SAFE_WRITE(fd, &width, sizeof(int));
    SAFE_WRITE(fd, &height, sizeof(int));
    SAFE_WRITE(fd, pb, width*height*sizeof(float));
    close(fd);
}

void writeFile(char* file, int width, int height, float* pb) {
    int fd;

    fd = open(file, O_CREAT|O_WRONLY, 0666);
    SAFE_WRITE(fd, &width, sizeof(int));
    SAFE_WRITE(fd, &height, sizeof(int));
    SAFE_WRITE(fd, pb, width*height*sizeof(float));
    close(fd);
}

void writeGradients(char* file, int width, int height, int pitchInFloats, int norients, int scales, float* pb) {
    int fd;

    fd = open(file, O_CREAT|O_WRONLY, 0666);
    SAFE_WRITE(fd, &width, sizeof(int));
    SAFE_WRITE(fd, &height, sizeof(int));
    SAFE_WRITE(fd, &norients, sizeof(int));
    SAFE_WRITE(fd, &scales, sizeof(int));
    for(int scale = 0; scale < scales; scale++) {
      for(int orient = 0; orient < norients; orient++) {
        float* currentPointer = &pb[pitchInFloats * orient + pitchInFloats * scale * norients];
        SAFE_WRITE(fd, currentPointer, width*height*sizeof(float));
      }
    }
    close(fd);
}

void writeArray(char* file, int ndim, int* dim, float* input) {
  int fd;
  fd = open(file, O_CREAT|O_WRONLY|O_TRUNC, 0666);
  int size = 1;
  for(int i = 0; i < ndim; i++) {
    size *= dim[i];
  }
  SAFE_WRITE(fd, &ndim, sizeof(int));
  SAFE_WRITE(fd, dim, sizeof(int) * ndim);
  SAFE_WRITE(fd, input, sizeof(float) * size);
  close(fd);
}

void transpose(int width, int height, float* input, float* output) {
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      output[col * height + row] = input[row * width + col];
    }
  }                                         
}

void checkInputValue(int& nEigNum, float& fEigTolerance, int& nTextonChoice)
{
	if (nEigNum > 25)
	{
		printf("\nException: Do not support for more than 25 eigen vectors.\n");
		nEigNum = 25;
	}
	if (nEigNum < 2)
	{
		printf("\nException: Do not support for less than 2 eigen vectors.\n");
		nEigNum = 9;
	}
	if  (fEigTolerance < 1e-5)
	{
		printf("\nException: Do not support for accuracy below 1e-5.\n");
		fEigTolerance = 1e-4;
	}
	if  (fEigTolerance > 1e-1)
	{
		printf("\nException: Do not support for accuracy above 1e-1.\n");
		fEigTolerance = 1e-3;
	}
	if (nTextonChoice > 2 || nTextonChoice < 1)
	{
		printf("\nException: Only support choice 1 (32 bins) and choice 2 (64 bins)\n");
	}
}

void parsingCommand(int argc, char** argv, int& nEigNum, float& fEigTolerance, int& nTextonChoice)
{
	if (argc < 3)
	{
		nEigNum = 9;
		fEigTolerance = 1e-3;
		nTextonChoice = TEXTON64;
		return;
	}
	if (argc < 4)
	{
		nEigNum = atoi(argv[2]);
		fEigTolerance = 1e-3;
		nTextonChoice = TEXTON64;
		checkInputValue(nEigNum, fEigTolerance, nTextonChoice);
		return;
	}
	if (argc < 5)
	{
		nEigNum = atoi(argv[2]);
		fEigTolerance = atof(argv[3]);
		nTextonChoice = TEXTON64;
		checkInputValue(nEigNum, fEigTolerance, nTextonChoice);
		return;
	}

	if (argc < 6)
	{
		nEigNum = atoi(argv[2]);
		fEigTolerance = atof(argv[3]);
		nTextonChoice = atoi(argv[4]);
		checkInputValue(nEigNum, fEigTolerance, nTextonChoice);
		return;
	}

}

int main_single_patch(uint width, uint height, uint* devRgbU, float** pdevGPb, float** pdevGPball, float** pdevMPb, int& matrixPitchInFloats, int nTextonChoice, int nEigNum, float fEigTolerance) {

  printf("Image found: %i x %i pixels\n", width, height);
  assert(width > 0);
  assert(height > 0);
#ifdef __TIMER_SPECFIC
  StopWatchInterface *timer_specific = NULL;
#endif
  int nPixels = width * height;

  size_t totalMemory, availableMemory;
  cuMemGetInfo(&availableMemory,&totalMemory );
  printf("Available %lu bytes on GPU\n", availableMemory);

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
 
#ifdef __TIMER_SPECFIC
  sdkCreateTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

  float* devGreyscale;
  rgbUtoGreyF(width, height, devRgbU, &devGreyscale);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  float mulTime = sdkGetTimerValue(&timer_specific);
  printf(">+< rgbUtoGrayF | %f | ms\n", mulTime);
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  
  int* devTextons;
  findTextons(width, height, devGreyscale, &devTextons, nTextonChoice);
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< texton | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

  float* devL;
  float* devA;
  float* devB;
  rgbUtoLab3F(width, height, 2.5, devRgbU, &devL, &devA, &devB);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< rgbUtoLab3F | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  normalizeLab(width, height, devL, devA, devB);
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< normalizeLab | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  //int border = 30;
  //int borderWidth = width + 2 * border;
  //int borderHeight = height + 2 * border;
  //float* devLMirrored;
  //mirrorImage(width, height, border, devL, &devLMirrored);
 
  cudaThreadSynchronize();
  cudaFree(devRgbU);
  cudaFree(devGreyscale);
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< mirrorImage | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  float* devBg;
  float* devCga;
  float* devCgb;
  float* devTg;  
 
  StopWatchInterface *localcuestimer = NULL;
  sdkCreateTimer(&localcuestimer);
  sdkStartTimer(&localcuestimer);

  localCues(width, height, devL, devA, devB, devTextons, &devBg, &devCga, &devCgb, &devTg, &matrixPitchInFloats, nTextonChoice);

  sdkStopTimer(&localcuestimer);
  printf("localcues time: %f seconds\n", sdkGetTimerValue(&localcuestimer)/1000.0);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< localcues | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

  cudaFree(devTextons);
  cudaFree(devL);
  cudaFree(devA);
  cudaFree(devB);
  
  float* devMPbO;
  float *devCombinedGradient;
  combine(width, height, matrixPitchInFloats, devBg, devCga, devCgb, devTg, &devMPbO, &devCombinedGradient, nTextonChoice);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< combine | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

  CUDA_SAFE_CALL(cudaFree(devBg));
  CUDA_SAFE_CALL(cudaFree(devCga));
  CUDA_SAFE_CALL(cudaFree(devCgb));
  CUDA_SAFE_CALL(cudaFree(devTg));

  float* devMPb;
  cudaMalloc((void**)&devMPb, sizeof(float) * nPixels);
  nonMaxSuppression(width, height, devMPbO, matrixPitchInFloats, devMPb);
  CUDA_SAFE_CALL(cudaFree(devMPbO));		// probably not right
  *pdevMPb = devMPb;

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< nonmaxsupression | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  
  int radius = 5;
  //int radius = 10;

  Stencil theStencil(radius, width, height, matrixPitchInFloats);
  int nDimension = theStencil.getStencilArea();
  float* devMatrix;
  intervene(theStencil, devMPb, &devMatrix);
  printf("Intervening contour completed\n");
 
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< intervene | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

  float* eigenvalues;
  float* devEigenvectors;
  //int nEigNum = 17;
  generalizedEigensolve(theStencil, devMatrix, matrixPitchInFloats, nEigNum, &eigenvalues, &devEigenvectors, fEigTolerance);
  CUDA_SAFE_CALL(cudaFree(devMatrix));

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< generalizedEigensolve | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  float* devSPb = 0;
  size_t devSPb_pitch = 0;
  CUDA_SAFE_CALL(cudaMallocPitch((void**)&devSPb, &devSPb_pitch, nPixels *  sizeof(float), 8));
  cudaMemset(devSPb, 0, matrixPitchInFloats * sizeof(float) * 8);

  spectralPb(eigenvalues, devEigenvectors, width, height, nEigNum, devSPb, matrixPitchInFloats);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< spectralPb | %f | ms\n", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif  
  float* devGPb = 0;
  float* devGPball = 0;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devGPb, sizeof(float) * nPixels));
  CUDA_SAFE_CALL(cudaMalloc((void**)&devGPball, sizeof(float) * matrixPitchInFloats * 8));
  StartCalcGPb(nPixels, matrixPitchInFloats, 8, devCombinedGradient, devSPb, devMPb, devGPball, devGPb);
  *pdevGPb = devGPb;
  *pdevGPball = devGPball;
 
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< StartCalcGpb | %f | ms\n", sdkGetTimerValue(&timer_specific));
#endif

  CUDA_SAFE_CALL(cudaFree(devEigenvectors));
  CUDA_SAFE_CALL(cudaFree(devCombinedGradient));
  CUDA_SAFE_CALL(cudaFree(devSPb));
  
  return 0;
}

int main(int argc, char** argv) {
  cuInit(0);
  chooseLargestGPU(true);

  if (argc < 2) {
	printf("\nUsage: %s input_image.ppm eigenvector_num eigenvector_tolerance texton_choice", argv[0]);
	printf("\nInput image should be in ppm format");
	printf("\nThe number of eigenvectors is from 2 to 25");
	printf("\nThe eigenvector tolerance is from 1e-2 to 1e-5");
	printf("\nFor the texton choice parameter, 1 for 32 bins, 2 for 64 bins\n");
    exit(1);
  }

  char* filename = argv[1];
  char outputPGMfilename[1000];
  char outputthinPGMfilename[1000];
  char outputPBfilename[1000];
  char outputthinPBfilename[1000];
  char outputgpbAllfilename[1000];
  printf("Processing: %s, output in ", filename);
  char* period = strrchr(filename, '.');
  if (period == 0) {
    period = strrchr(filename, 0);
  }
  strncpy(outputPGMfilename, filename, period - filename);
  sprintf(&outputPGMfilename[0] + (period - filename) , "Pb.pgm");
  strncpy(outputthinPGMfilename, filename, period - filename);
  sprintf(&outputthinPGMfilename[0] + (period - filename) , "Pbthin.pgm");
  
  strncpy(outputPBfilename, filename, period - filename);
  sprintf(&outputPBfilename[0] + (period - filename), ".pb");
  strncpy(outputthinPBfilename, filename, period - filename);
  sprintf(&outputthinPBfilename[0] + (period - filename), ".thin.pb");
  
  printf("%s and %s\n", outputPGMfilename, outputPBfilename);
  strncpy(outputgpbAllfilename, filename, period - filename);
  sprintf(&outputgpbAllfilename[0] + (period - filename), "GpbAll.ary");

  int nEigNum = 9;
  float fEigTolerance = 1e-3;
  int nTextonChoice = TEXTON64;

  parsingCommand(argc, argv, nEigNum, fEigTolerance, nTextonChoice);
  printf("\n Eig %d Tol %f Texton %d\n", nEigNum, fEigTolerance, nTextonChoice);

  uint width;
  uint height;
  uint* devRgbU;
  // load the image
  loadPPM_rgbU(filename, &width, &height, &devRgbU);
  int nPixels = width * height;
  float* devGPb = 0;
  float* devGPball = 0;
  float* devMPb;
  int matrixPitchInFloats;
  
  //--------------------------------------------------------------------------------------------
  // decompose the image into multiple parts if it is too big, make sure there is enough padding  
  {
	int max_patch_length = 800;
	int pad_size = 200;
	
	if ( (width<=max_patch_length) && (height<=max_patch_length) ) {  
	  float* tpatchDevGPb = 0;
	  float* tpatchDevGPball = 0;
	  float* tpatchDevMPb = 0;
	  int tpatchMatrixPitchInFloats;
	  main_single_patch(width, height, devRgbU, &tpatchDevGPb, &tpatchDevGPball, &tpatchDevMPb, tpatchMatrixPitchInFloats, nTextonChoice, nEigNum, fEigTolerance);
	  matrixPitchInFloats = tpatchMatrixPitchInFloats;
	  devGPb = tpatchDevGPb;
	  devGPball = tpatchDevGPball;
	  devMPb = tpatchDevMPb;
	  //CUDA_SAFE_CALL(cudaFree(devGPb));
	  //CUDA_SAFE_CALL(cudaFree(devGPball));
	  //CUDA_SAFE_CALL(cudaFree(devMPb));
	  fprintf(stdout,"\nshanqi: TOCHECK [%d, %d] --> matrixPitchInFloats %d\n", width, height, matrixPitchInFloats);
	} else {
	
	  int w_seg = width/max_patch_length + ((width%max_patch_length)==0? 0:1);
	  int h_seg = height/max_patch_length + ((height%max_patch_length)==0? 0:1);
	  
	  int seg_width = width/w_seg + ((width%w_seg)==0? 0:1);
	  int seg_height = height/h_seg + ((height%h_seg)==0? 0:1);
	  
	  //matrixPitchInFloats = width*height;
	  int nPixels = width * height;
	  matrixPitchInFloats = findPitchInFloats(nPixels);
	  
	  CUDA_SAFE_CALL(cudaMalloc((void**)&devGPb, sizeof(float) * nPixels));
	  CUDA_SAFE_CALL(cudaMalloc((void**)&devGPball, sizeof(float) * matrixPitchInFloats * 8));
	  CUDA_SAFE_CALL(cudaMalloc((void**)&devMPb, sizeof(float) * nPixels));
	  
	  for (int ih=0; ih<h_seg; ++ih) {
	    for (int iw=0; iw<w_seg; ++iw) {
		  fprintf( stdout, "\nimage_size = [%d, %d], ih, iw = [%d, %d], seg_size = [%d, %d]\n", width, height, ih, iw, seg_width, seg_height );
		  int xstart = iw*seg_width;
		  int xend = min(xstart+seg_width-1, width-1);
		  int left_pad = min(pad_size, xstart);
		  int right_pad = min(pad_size, width-xend-1);
		  
		  int ystart = ih*seg_height;
		  int yend = min(ystart+seg_height-1, height-1);
		  int top_pad = min(pad_size, ystart);
		  int bottom_pad = min(pad_size, height-yend-1);
		  
		  int patch_width = xend-xstart+1 + right_pad + left_pad;
		  int patch_height = yend-ystart+1 + top_pad + bottom_pad;
		  
		  int patch_width_inner = xend-xstart+1;
		  //int patch_height_inner = yend-ystart+1;
		  
		  int patch_xstart = xstart-left_pad;
		  int patch_ystart = ystart-top_pad;
		  
		  float* tpatchDevGPb = 0;
		  float* tpatchDevGPball = 0;
		  float* tpatchDevMPb = 0;
		  int tpatchMatrixPitchInFloats;

		  // copy data in		  
		  uint patchLineSizeUint = sizeof(uint) * patch_width;
		  uint patchSizeUint = patchLineSizeUint * patch_height;		  
		  uint* tpatchDevRgbU = 0;
		  cudaMalloc((void**)&tpatchDevRgbU, patchSizeUint);
		  for ( int ik=0; ik<patch_height; ++ ik ) {
		    uint* ptarget = &(tpatchDevRgbU[ik*patch_width]);
		    uint* psource = &(devRgbU[(ik+patch_ystart)*width+patch_xstart]);
		    cudaMemcpy(ptarget, psource, patchLineSizeUint, cudaMemcpyDeviceToDevice);
		  }		  
		  main_single_patch(patch_width, patch_height, tpatchDevRgbU, &tpatchDevGPb, &tpatchDevGPball, &tpatchDevMPb, tpatchMatrixPitchInFloats, nTextonChoice, nEigNum, fEigTolerance);
		  // copy data out
		  uint patchLineInnerSizeFloat = sizeof(float) * patch_width_inner;
		  for ( int ik=0; ik<patch_height; ++ ik ) {
		    float* ptarget = &(devGPb[(ik+ystart)*width+xstart]);
		    float* psource = &(tpatchDevGPb[(ik+top_pad)*patch_width+left_pad]);
		    cudaMemcpy(ptarget, psource, patchLineInnerSizeFloat, cudaMemcpyDeviceToDevice);
		  }
		  for ( int jj=0; jj<8; ++ jj ) {
			for ( int ik=0; ik<patch_height; ++ ik ) {
		      float* ptarget = &(devGPball[jj*matrixPitchInFloats + (ik+ystart)*width+xstart]);
			  float* psource = &(tpatchDevGPball[jj*tpatchMatrixPitchInFloats + (ik+top_pad)*patch_width+left_pad]);
			  cudaMemcpy(ptarget, psource, patchLineInnerSizeFloat, cudaMemcpyDeviceToDevice);
			}
		  }
		  for ( int ik=0; ik<patch_height; ++ ik ) {
		    float* ptarget = &(devMPb[(ik+ystart)*width+xstart]);
		    float* psource = &(tpatchDevMPb[(ik+top_pad)*patch_width+left_pad]);
		    cudaMemcpy(ptarget, psource, patchLineInnerSizeFloat, cudaMemcpyDeviceToDevice);
		  }
		  //fprintf(stdout, "%p, %p, %p, %p\n", tpatchDevRgbU, tpatchDevGPb, tpatchDevGPball, tpatchDevMPb);
		  //CUDA_SAFE_CALL(cudaFree(tpatchDevRgbU));		// already been released
		  CUDA_SAFE_CALL(cudaFree(tpatchDevGPb));
		  CUDA_SAFE_CALL(cudaFree(tpatchDevGPball));
		  CUDA_SAFE_CALL(cudaFree(tpatchDevMPb));
		}
	  }
	}
  }
  //--------------------------------------------------------------------------------------------
  
#ifdef __TIMER_SPECFIC
  StopWatchInterface *timer_specific = NULL;
#endif
#ifdef __TIMER_SPECFIC 
  sdkCreateTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  float* devGPb_thin = 0;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devGPb_thin, nPixels * sizeof(float) ));
  PostProcess(width, height, width, devGPb, devMPb, devGPb_thin); //note: 3rd param width is the actual pitch of the image
  NormalizeGpbAll(nPixels, 8, matrixPitchInFloats, devGPball);
  
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  
  cudaThreadSynchronize();
  sdkStopTimer(&timer);
  printf("CUDA Status : %s\n", cudaGetErrorString(cudaGetLastError()));

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  printf(">+< PostProcess | %f | ms\n", sdkGetTimerValue(&timer_specific));
#endif
  printf(">+< Computation time: | %f | seconds\n", sdkGetTimerValue(&timer)/1000.0);  
  float* hostGPb = (float*)malloc(sizeof(float)*nPixels);
  memset(hostGPb, 0, sizeof(float) * nPixels);
  cudaMemcpy(hostGPb, devGPb, sizeof(float)*nPixels, cudaMemcpyDeviceToHost);

  cutSavePGMf(outputPGMfilename, hostGPb, width, height);
  writeFile(outputPBfilename, width, height, hostGPb);

  /* thin image */
  float* hostGPb_thin = (float*)malloc(sizeof(float)*nPixels);
  memset(hostGPb_thin, 0, sizeof(float) * nPixels);
  cudaMemcpy(hostGPb_thin, devGPb_thin, sizeof(float)*nPixels, cudaMemcpyDeviceToHost);
  cutSavePGMf(outputthinPGMfilename, hostGPb_thin, width, height);
  writeFile(outputthinPBfilename, width, height, hostGPb);
  free(hostGPb_thin);
  /* end thin image */

  float* hostGPbAll = (float*)malloc(sizeof(float) * matrixPitchInFloats * 8);
  cudaMemcpy(hostGPbAll, devGPball, sizeof(float) * matrixPitchInFloats * 8, cudaMemcpyDeviceToHost);

  int oriMap[] = {3, 2, 1, 0, 7, 6, 5, 4};
  float* hostGPbAllConcat = (float*)malloc(sizeof(float) * width * height * 8);
  for(int i = 0; i < 8; i++) {
    transpose(width, height, hostGPbAll + matrixPitchInFloats * oriMap[i], hostGPbAllConcat + width * height * i);
  }
  int dim[3];
  dim[0] = 8; 
  dim[1] = width;
  dim[2] = height;
  writeArray(outputgpbAllfilename, 3, dim, hostGPbAllConcat);

  free(hostGPb);
  free(hostGPbAll);
  free(hostGPbAllConcat);

  CUDA_SAFE_CALL(cudaFree(devGPb));
  CUDA_SAFE_CALL(cudaFree(devGPb_thin));
  CUDA_SAFE_CALL(cudaFree(devGPball));
  CUDA_SAFE_CALL(cudaFree(devMPb));
  //CUDA_SAFE_CALL(cudaFree(devRgbU));		// already been released

}
