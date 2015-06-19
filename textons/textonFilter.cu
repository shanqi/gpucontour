#include <cuda.h>
#include <cutil.h>
#include <stdio.h>
#include <cublas.h>
#include "texton.h"

void chooseLargestGPU(bool verbose) {
  int cudaDeviceCount;
  cudaGetDeviceCount(&cudaDeviceCount);
  int cudaDevice = 0;
  int maxSps = 0;
  struct cudaDeviceProp dp;
  for (int i = 0; i < cudaDeviceCount; i++) {
    cudaGetDeviceProperties(&dp, i);
    if (dp.multiProcessorCount >= maxSps) {
      maxSps = dp.multiProcessorCount;
      cudaDevice = i;
    }
  }
  cudaGetDeviceProperties(&dp, cudaDevice);
  if (verbose) {
    printf("Using cuda device %i: %s\n", cudaDevice, dp.name);
  }
  cudaSetDevice(cudaDevice);
}


int main(int argc, char** argv) {
  chooseLargestGPU(true);
  printf("Loading image...");
  char filename[] = "polynesia.pgm";
  //char* filename = "tiny.pgm";
  float* hostImage = 0;
  unsigned int width;
  unsigned int height;
  cutLoadPGMf(filename, &hostImage, &width, &height);
  int nPixels = width * height;
  printf("width = %i, height = %i\n", width, height);
  float* devImage;
  cudaMalloc((void**)&devImage, sizeof(float) * nPixels);
  cudaMemcpy(devImage, hostImage, sizeof(float) * nPixels, cudaMemcpyHostToDevice);
  int* devClusters;

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  findTextons(width, height, devImage, &devClusters, 1);
  
  int* hostClusters = (int*)malloc(sizeof(int)*nPixels);
  unsigned char* hostClustersUb = (unsigned char*)malloc(sizeof(unsigned char) * nPixels);

  cudaThreadSynchronize();
  sdkStopTimer(&timer);
  float mulTime = sdkGetTimerValue(&timer);
  printf("Texton time: %f\n", mulTime);
  cudaMemcpy(hostClusters, devClusters, sizeof(int) * nPixels, cudaMemcpyDeviceToHost);
  for(int i = 0; i < nPixels; i++) {
    hostClustersUb[i] = (unsigned char)hostClusters[i] * 4;
  }
  cutSavePGMub("newClusters.pgm", hostClustersUb, width, height);
  /* float* sgemmResult; */
/*   cudaMalloc((void**)&sgemmResult, sizeof(float) * nPixels * clusterCount); */
/*   cublasSgemm('n', 't', nPixels, filterCount, clusterCount, 1.0f, devResponses, nPixels, centroids, clusterCount, 0.0f, sgemmResult, nPixels); */
 
/*   FILE* fp; */
/*   fp = fopen("iterationTimes.txt", "w"); */
/*   for (int j = 0; j < i; j++) { */
/*     fprintf(fp, "%i ", j); */
/*     floatVector* currentIteration = times[j]; */
/*     for(std::vector<float>::iterator it = currentIteration->begin(); it != currentIteration->end(); it++) { */
/*       fprintf(fp, "%e ", *it); */
/*     } */
/*     fprintf(fp, "\n"); */
/*   } */
/*   fclose(fp); */
  
  FILE* fp;
  fp = fopen("newClusters.txt", "w");
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      fprintf(fp, "%i ", hostClusters[col + row * width]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  //testSgemm();
}
