#include <iostream>
#include <chrono>
#include <cstring>
using namespace std;
#define THREADS_PER_BLOCK 64


__global__ void polynomial_expansion (float poly[], int degree, unsigned long int n, float array[]) {
  
  int index = threadIdx.x + blockIdx.x * 1000;
  float x = array[index];
  float out = 0.;
  float xtothepowerof = 1.;
  for (int i=0; i<=degree; ++i) {
    out += xtothepowerof * (poly[i]);
    xtothepowerof *= x;
  }
  array[index] = out; 

}


int main (int argc, char* argv[]) {
  //TODO: add usage
  
  if (argc < 3) {
     std::cerr<<"usage: "<<argv[0]<<" <n> <degree>"<<std::endl;
     return -1;
  }

  unsigned long int n = atoi(argv[1]); //TODO: atoi is an unsafe function
  int degree = atoi(argv[2]);
  int nbiter = 1;

  float array[] = new float[n];
  float poly[] = new float[degree+1];

  int array_size = n * sizeof(float);
  int poly_size = (degree+1) * sizeof(float);
  
  //array = (float *)malloc(array_size);
  //poly = (float *)malloc(poly_size);

  for (int i=0; i<n; ++i)
    array[i] = 1.;

  for (int i=0; i<degree+1; ++i)
    poly[i] = 1.;

  float *d_array, *d_poly;
  //int *d_degree, *d_n;
  //cudaMalloc((void **) &d_degree, sizeof(int));
  //cudaMalloc((void **) &d_n, sizeof(int));
  
  cudaMalloc((void **)&d_array, array_size);

  cudaMalloc((void **)&d_poly, poly_size);

  cudaMemcpy(d_array, array, array_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_poly, poly, poly_size, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_degree, &degree, sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
  
  std::chrono::time_point<std::chrono::system_clock> begin, end;
  begin = std::chrono::system_clock::now();
  
  for (int iter = 0; iter<nbiter; ++iter)
    polynomial_expansion<<<n/1000,1000>>> (d_poly, degree, n, d_array);
  cudaDeviceSynchronize(); 
 //for (int i=0; i<n; ++i)
  //  array[i] = 3.;
  cudaMemcpy(array, d_array, array_size, cudaMemcpyDeviceToHost);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> totaltime = (end-begin)/nbiter;

  for (int i = 0; i < 3; ++i)
  {
    std::cerr<<"result array["<<i<<"]: "<<array[i]<<std::endl;
  }

  std::cerr<<array[0]<<std::endl;
  std::cout<<n<<" "<<degree<<" "<<totaltime.count()<<std::endl;

  delete[] array;
  delete[] poly;
  cudaFree(d_array);
  cudaFree(d_poly);
  //cudaFree(d_degree);
  //cudaError_t cudaGetLastError(void);
  //char *cudaGetErrorString(cudaError_t);
  //std::cout<<cudaGetErrorString(cudaError_t);
  return 0;
}
