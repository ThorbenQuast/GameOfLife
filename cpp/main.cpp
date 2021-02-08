#include <iostream>
#include "include/universe.hh"
#include "TCanvas.h"
#include "TH2F.h"
#include "TFile.h"
#include <sstream>
#include <chrono>

#define NEPOCHS 15
#define DIMENSIONX 10000
#define DIMENSIONY 10000

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

    TCanvas *canvas = new TCanvas("canvas", "canvas", 900, 800);
    for (int n = 0; n < NEPOCHS; n++)
    {
        std::ostringstream s1;
        s1 << "epoch_ " << n;
        TH2F *graphic = new TH2F(s1.str().c_str(), s1.str().c_str(), DIMENSIONX, -0.5, DIMENSIONX - 0.5, DIMENSIONY, -0.5, DIMENSIONY - 0.5);

        for (int i = 0; i < DIMENSIONX; i++)
            for (int j = 0; j < DIMENSIONY; j++)
            {
                float weight = universe[n][i][j] ? 1 : 0.01;
                graphic->SetBinContent(i + 1, j + 1, weight);
            }

        graphic->SetTitle(s1.str().c_str());
        graphic->GetXaxis()->SetTitle("x (a.u.)");
        graphic->GetYaxis()->SetTitle("y (a.u.)");
        graphic->GetXaxis()->SetLabelSize(0);
        graphic->GetYaxis()->SetLabelSize(0);
        graphic->SetStats(false);

        graphic->Draw("COLZ");
        s1 << ".png";
        canvas->Print(s1.str().c_str());
    }

    return 0;
}