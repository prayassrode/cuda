#include <iostream>
#include <chrono>

#define THREADS_PER_BLOCK 64

__device__ float polynomial (float x, float* poly, int degree) {
  float out = 0.;
  float xtothepowerof = 1.;
  for (int i=0; i<=degree; ++i) {
    out += xtothepowerof*poly[i];
    xtothepowerof *= x;
  }
  return out;
}

__global__ void polynomial_expansion (float* poly, int degree,
         int n, float* array) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    array[i] = polynomial (array[i], poly, degree);
}

__global__ void polynomial_expansion (float* poly, int degree, int n, float* array) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n){
    float out = 0.;
    float xtothepowerof = 1.;
    for (int i=0; i<=degree; ++i) {
      out += xtothepowerof*poly[i];
      xtothepowerof *= array[index];
  }
    array[index] = out; 
  }

}


int main (int argc, char* argv[]) {
  //TODO: add usage
  
  if (argc < 4) {
     std::cerr<<"usage: "<<argv[0]<<" <n> <degree> <block_size> "<<std::endl;
     return -1;
  }

  int n = atoi(argv[1]); //TODO: atoi is an unsafe function
  int degree = atoi(argv[2]);
  int nbiter = 1;

  float* array = new float[n];
  float* poly = new float[degree+1];
  // int* degree;
  // degree = d;
  for (int i=0; i<n; ++i)
    array[i] = 1.;

  for (int i=0; i<degree+1; ++i)
    poly[i] = 1.;

  float *d_array, *d_poly;
  // int *d_degree;
  // cudaMalloc((void **) &d_degree, sizeof(int));
  
  int array_size = n * sizeof(float);
  cudaMalloc((void **) &d_array, array_size);

  int poly_size = degree * sizeof(float);
  cudaMalloc((void **) &d_poly, poly_size);

  cudaMemcpy(d_array, array, array_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_poly, poly, poly_size, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_degree, degree, sizeof(int), cudaMemcpyHostToDevice);

  
  std::chrono::time_point<std::chrono::system_clock> begin, end;
  begin = std::chrono::system_clock::now();
  
  for (int iter = 0; iter<nbiter; ++iter)
    polynomial_expansion<<<(n+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>> (poly, degree, n, array);

  cudaMemcpy(array, d_array, array_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < n; ++i)
  {
    std::cout<<"result array[i]: "<<array[i]<<std::endl;
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> totaltime = (end-begin)/nbiter;

  std::cerr<<array[0]<<std::endl;
  std::cout<<n<<" "<<degree<<" "<<totaltime.count()<<std::endl;

  delete[] array;
  delete[] poly;
  cudaFree(d_array);
  cudaFree(d_poly);
  // cudaFree(d_degree);

  return 0;
}
