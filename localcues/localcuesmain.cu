#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>


#include "spec.h"
#include "convert.h"
#include "localcues.h"
#include "stencilMVM.h"
#include "texton.h"
#include "cutil.h"

void writeGra(char* file, int width, int height, int norients, int nscale, int cuePitchInFloats, float* hostGradient)
{
    int fd;

    fd = open(file, O_CREAT|O_WRONLY|O_TRUNC, 0666);
    write(fd, &width, sizeof(int));
    write(fd, &height, sizeof(int));
    write(fd, &norients, sizeof(int));
    write(fd, &nscale, sizeof(int));
    for(int scale = 0; scale < nscale; scale++) {
      for(int orient = 0; orient < norients; orient++) {
        write(fd, &hostGradient[(scale*norients + orient) * cuePitchInFloats], width*height*sizeof(float));
      }
    }
    close(fd);
}


int main(int argc, char** argv)
{
  cuInit(0);
  chooseLargestGPU(true);
  size_t total, free;
  cuMemGetInfo(&free, &total);
  printf("This GPU has %d bytes of memory\n", total);
  printf("This GPU has %d bytes of free memory\n", free);
  if (argc != 2)
  {
    printf("give me a file!\n");
    return 0;
  }
  char* filename = argv[1];
  uint width;
  uint height;
  uint* devRgbU;
  loadPPM_rgbU(filename, &width, &height, &devRgbU);
  
  cuMemGetInfo(&free, &total);
  printf("After loading the image, there are %d bytes of free memory\n", free);
  float* devGreyscale;
  rgbUtoGreyF(width, height, devRgbU, &devGreyscale);
  cuMemGetInfo(&free, &total);
  printf("After converting to greyscale, there are %d bytes of free memory\n", free);
  int* devTextons;
  int textonChoice = 1;
  findTextons(width, height, devGreyscale, &devTextons, textonChoice);
  cudaFree(devGreyscale);
  cuMemGetInfo(&free, &total);
  printf("After finding textons, there are %d bytes of free memory\n", free);
 
  
  float* devL;
  float* devA;
  float* devB;
  rgbUtoLab3F(width, height, 2.5, devRgbU, &devL, &devA, &devB);
  cudaFree(devRgbU);
  normalizeLab(width, height, devL, devA, devB);
  cuMemGetInfo(&free, &total);
  printf("After converting to normalized LAB, there are %d bytes of free memory\n", free);
 
  
  float* devBg;
  float* devCga;
  float* devCgb;
  float* devTg;
  int cuePitchInFloats;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  localCues(width, height, devL, devA, devB, devTextons, &devBg, &devCga, &devCgb, &devTg, &cuePitchInFloats, textonChoice);
  sdkStopTimer(&timer);
  float mulTime = sdkGetTimerValue(&timer);
  printf("Local cues time: %f ms\n", mulTime);
  cuMemGetInfo(&free, &total);
  printf("After local cues, there are %d bytes of free memory\n", free);
  int size = sizeof(float) * cuePitchInFloats * 8 * 3;
  float* hostBg = (float*)malloc(size);
  float* hostCga = (float*)malloc(size);
  float* hostCgb = (float*)malloc(size);
  float* hostTg = (float*)malloc(size);

  
  cudaMemcpy(hostBg, devBg, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCga, devCga, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCgb, devCgb, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostTg, devTg, size, cudaMemcpyDeviceToHost);

  int norients = 8;
  int nscale = 3;
  writeGra((char*)("bg.gra"), width, height, norients, nscale, cuePitchInFloats, hostBg);
  writeGra((char*)("cga.gra"), width, height, norients, nscale, cuePitchInFloats, hostCga);
  writeGra((char*)("cgb.gra"), width, height, norients, nscale, cuePitchInFloats, hostCgb);
  writeGra((char*)("tg.gra"), width, height, norients, nscale, cuePitchInFloats, hostTg);
  
}

