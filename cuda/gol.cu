#define NEPOCHS 1000
#define DIMENSIONX 1000
#define DIMENSIONY 1000

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

__global__
void Evolve(bool *u, int n, int dx, int dy)
{
    int entry_index = blockIdx.x*blockDim.x + threadIdx.x;
    if (entry_index>=dx*dy) return;
    
    int i = entry_index % dy;
    int j = entry_index / dy;

    //get number of neighbors
    size_t NActiveNeighbors = 0;
    int nmin_x = max(0, i - 1);
    int nmax_x = min(i + 1, dx - 1);
    int nmin_y = max(0, j - 1);
    int nmax_y = min(j + 1, dy - 1);
    for (int ii = nmin_x; ii <= nmax_x; ii++)
        for (int jj = nmin_y; jj <= nmax_y; jj++)
        {
            if ((i == ii) && (j == jj))
                continue;
            if (u[n*dx*dy+ii*dx+jj]) NActiveNeighbors += 1;
        }

    bool active_pre, active_post;
    int n_next = n + 1;
        
    active_pre = u[n*dx*dy+i*dx+j];
    active_post = false;
    if (active_pre && (NActiveNeighbors == 2))
        active_post = true;
    else if (active_pre && (NActiveNeighbors == 3))
        active_post = true;
    else if ((!active_pre) && (NActiveNeighbors == 3))
        active_post = true;
    u[n_next*dx*dy+i*dx+j] = active_post;

    return;
}

bool *allocate_universe(int n, int dx, int dy)
{
    int Nelements = n*dx*dy;
    bool *universe = new bool[Nelements];
    for (int i = 0; i < Nelements; i++)
    {
      universe[i] = false;
    }
    return universe;
}

void set_initial_conditions(bool *u, int dx, int dy, float p)
{
    for (int i = 0; i < dx; i++)
        for (int j = 0; j < dy; j++)
        {
            if (((rand() % 1000) < p * 1000))
            {
                u[i*dx+j] = true;
            }
        }
    rand();
}

int main(void)
{
  long int NENTRIES = NEPOCHS*DIMENSIONX*DIMENSIONY;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::cout << "Defining the universe and creating the initial conditions" << std::endl;
  
  bool *universe = allocate_universe(NEPOCHS, DIMENSIONX, DIMENSIONY);
  set_initial_conditions(universe, DIMENSIONX, DIMENSIONY, 0.25);
  
  bool *cuda_universe;
  cudaMalloc((void**)&cuda_universe, NENTRIES*sizeof(bool));
  cudaMemcpy(cuda_universe, universe, NENTRIES*sizeof(bool), cudaMemcpyHostToDevice);
  cudaEventRecord(start);
  for (int n=0; n<NEPOCHS-1; n++) {
    if (n % 10 == 0)
    std::cout << n << "/" << NEPOCHS << std::endl;
    Evolve<<<1+DIMENSIONX*DIMENSIONY/256, 256>>>(cuda_universe, n, DIMENSIONX, DIMENSIONY);
  }
  cudaEventRecord(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Total execution time (ms): %f\n ", milliseconds);
  cudaMemcpy(universe, cuda_universe, NENTRIES*sizeof(bool), cudaMemcpyDeviceToHost);
  
  //write everything to file  
  std::ofstream outfile;
  outfile.open ("evolution.txt");
  outfile << NEPOCHS << " , " << DIMENSIONX << " , " << DIMENSIONY << "\n";
  int nepoch, nx, ny;
  for (int i=0; i<NENTRIES; i++) {
    if (universe[i]) {
      nepoch = i/(DIMENSIONX*DIMENSIONY);
      nx = (i % (DIMENSIONX*DIMENSIONY)) / DIMENSIONX;
      ny = i % DIMENSIONX;
      outfile << nepoch << " , " << nx << " , " << ny << "\n";
    }
  }
  outfile.close();
  
  //clean up
  cudaFree(cuda_universe);
  free(universe);
}