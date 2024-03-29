//nvcc -o gol gol.cu

#define NEPOCHS 1000
#define DIMENSIONX 100
#define DIMENSIONY 100

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

__global__
void Evolve(bool *u, int n, int dx, int dy)
{
    int n_next = n + 1;
    bool active_pre, active_post;
    
    __shared__ bool shared_u[DIMENSIONX*DIMENSIONY];
    //__shared__ bool shared_u_next[DIMENSIONX*DIMENSIONY];
    //copy state in shared memory
    for (int i=threadIdx.x; i<DIMENSIONX*DIMENSIONY; i+=blockDim.x) {
        shared_u[i] = u[n*dx*dy+i];
    }

    __syncthreads();

    for (int entry_index=threadIdx.x+blockIdx.x*blockDim.x; entry_index<DIMENSIONX*DIMENSIONY; entry_index+=blockDim.x*gridDim.x) {
        int i = entry_index / dy;
        int j = entry_index % dy;
    
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
                if (shared_u[ii*dy+jj]) NActiveNeighbors += 1;
            }
            
        active_pre = shared_u[i*dy+j];
        active_post = false;
        if (active_pre && (NActiveNeighbors == 2))
            active_post = true;
        else if (active_pre && (NActiveNeighbors == 3))
            active_post = true;
        else if ((!active_pre) && (NActiveNeighbors == 3))
            active_post = true;
            u[n_next*dx*dy+i*dy+j] = active_post;
    }

    /*
    __syncthreads();
    for (int entry_index=threadIdx.x+blockIdx.x*blockDim.x; entry_index<DIMENSIONX*DIMENSIONY; entry_index+=blockDim.x*gridDim.x) {
        int i = entry_index / dy;
        int j = entry_index % dy;
        
        u[n_next*dx*dy+i*dy+j] = shared_u_next[i*dy+j];
    }    

    __syncthreads();
    */
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


bool glider_gun_field(int i, int j) {
    if (((i%40)==1)&&((j%30)==5)) return true;
    if (((i%40)==1)&&((j%30)==6)) return true;
    if (((i%40)==2)&&((j%30)==5)) return true;
    if (((i%40)==2)&&((j%30)==6)) return true;
    if (((i%40)==11)&&((j%30)==5)) return true;
    if (((i%40)==11)&&((j%30)==6)) return true;
    if (((i%40)==11)&&((j%30)==7)) return true;
    if (((i%40)==12)&&((j%30)==4)) return true;
    if (((i%40)==12)&&((j%30)==8)) return true;
    if (((i%40)==13)&&((j%30)==3)) return true;
    if (((i%40)==13)&&((j%30)==9)) return true;
    if (((i%40)==14)&&((j%30)==3)) return true;
    if (((i%40)==14)&&((j%30)==9)) return true;
    if (((i%40)==15)&&((j%30)==6)) return true;
    if (((i%40)==16)&&((j%30)==4)) return true;
    if (((i%40)==16)&&((j%30)==8)) return true;
    if (((i%40)==17)&&((j%30)==5)) return true;
    if (((i%40)==17)&&((j%30)==6)) return true;
    if (((i%40)==17)&&((j%30)==7)) return true;
    if (((i%40)==18)&&((j%30)==6)) return true;
    if (((i%40)==21)&&((j%30)==3)) return true;
    if (((i%40)==21)&&((j%30)==4)) return true;
    if (((i%40)==21)&&((j%30)==5)) return true;
    if (((i%40)==22)&&((j%30)==3)) return true;
    if (((i%40)==22)&&((j%30)==4)) return true;
    if (((i%40)==22)&&((j%30)==5)) return true;
    if (((i%40)==23)&&((j%30)==2)) return true;
    if (((i%40)==23)&&((j%30)==6)) return true;
    if (((i%40)==25)&&((j%30)==1)) return true;
    if (((i%40)==25)&&((j%30)==2)) return true;
    if (((i%40)==25)&&((j%30)==6)) return true;
    if (((i%40)==25)&&((j%30)==7)) return true;
    if (((i%40)==35)&&((j%30)==3)) return true;
    if (((i%40)==35)&&((j%30)==4)) return true;
    if (((i%40)==36)&&((j%30)==3)) return true;
    if (((i%40)==36)&&((j%30)==4)) return true;

    return false;
}

void set_initial_conditions(bool *u, int dx, int dy, float p)
{
    for (int i = 0; i < dx; i++)
        for (int j = 0; j < dy; j++)
        {
            //float _value = (rand() % 10000)/10000.;//((i*173+j*51) % 100) / 100.;
            //if (_value < p)
            if (glider_gun_field(i, j))
            {  
                u[i*dy+j] = true;
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
    Evolve<<<32, 64>>>(cuda_universe, n, DIMENSIONX, DIMENSIONY);
    cudaDeviceSynchronize();
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
      nx = (i % (DIMENSIONX*DIMENSIONY)) / DIMENSIONY;
      ny = i % DIMENSIONY;
      outfile << nepoch << " , " << nx << " , " << ny << "\n";
    }
  }
  outfile.close();
  
  //clean up
  cudaFree(cuda_universe);
  free(universe);
}
