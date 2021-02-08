#define NEPOCHS 1000
#define DIMENSIONX 1000
#define DIMENSIONY 1000

#include <iostream>
#include "include/universe.hh"
#include <fstream>
#include <chrono>

int main()
{

    std::cout << "Defining the universe and creating the initial conditions" << std::endl;
    auto start = std::chrono::steady_clock::now();
    bool ***universe = allocate_universe(NEPOCHS, DIMENSIONX, DIMENSIONY);
    set_initial_conditions(universe, DIMENSIONX, DIMENSIONY, 0.25);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Ellapsed time in seconds: " << elapsed_seconds.count() << std::endl;

    std::cout << "Evolutions" << std::endl;
    start = std::chrono::steady_clock::now();
    for (int n = 0; n < NEPOCHS; n++)
    {
        if (n + 1 == NEPOCHS)
            continue;
        Evolve(universe, n, DIMENSIONX, DIMENSIONY);
        if (n % 10 == 0)
            std::cout << n << "/" << NEPOCHS << std::endl;
    }
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Ellapsed time in seconds: " << elapsed_seconds.count() << std::endl;

    //write everything to file  
    std::ofstream outfile;
    outfile.open ("evolution.txt");
    outfile << NEPOCHS << " , " << DIMENSIONX << " , " << DIMENSIONY << "\n";
    int nepoch, nx, ny;
    for (int nepoch=0; nepoch<NEPOCHS; nepoch++) for (int nx=0; nx<DIMENSIONX; nx++) for (int ny=0; ny<DIMENSIONY; ny++) {
        if (universe[nepoch][nx][ny]) {
            outfile << nepoch << " , " << nx << " , " << ny << "\n";
        }
    }
    outfile.close();

    return 0;
}