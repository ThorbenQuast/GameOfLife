#include <algorithm>

size_t GetNActiveNeighbours(bool ***u, int n, int dx, int dy, int nx, int ny)
{
    size_t NActives = 0;
    int nmin_x = std::max(0, nx - 1);
    int nmax_x = std::min(nx + 1, dx - 1);
    int nmin_y = std::max(0, ny - 1);
    int nmax_y = std::min(ny + 1, dy - 1);

    for (int i = nmin_x; i <= nmax_x; i++)
        for (int j = nmin_y; j <= nmax_y; j++)
        {
            if ((i == nx) && (j == ny))
                continue;
            if (u[n][i][j])
                NActives += 1;
        }

    return NActives;
}

void Evolve(bool ***u, int n, int dx, int dy)
{
    size_t NActiveNeighbors;
    bool active_pre, active_post;
    int n_next = n + 1;
    for (int i = 0; i < dx; i++)
        for (int j = 0; j < dy; j++)
        {
            NActiveNeighbors = GetNActiveNeighbours(u, n, dx, dy, i, j);
            active_pre = u[n][i][j];
            active_post = false;
            if (active_pre && (NActiveNeighbors == 2))
                active_post = true;
            else if (active_pre && (NActiveNeighbors == 3))
                active_post = true;
            else if ((!active_pre) && (NActiveNeighbors == 3))
                active_post = true;
            u[n_next][i][j] = active_post;
        }
}

bool ***allocate_universe(int n, int dx, int dy)
{
    bool ***universe = new bool **[n];
    for (int i = 0; i < n; i++)
    {
        universe[i] = new bool *[dx];
        for (int j = 0; j < dx; j++)
        {
            universe[i][j] = new bool[dy];
            for (int k = 0; k < dy; k++)
            {
                universe[i][j][k] = false;
            }
        }
    }
    return universe;
}

void set_initial_conditions(bool ***u, int dx, int dy, float p)
{
    for (int i = 0; i < dx; i++)
        for (int j = 0; j < dy; j++)
        {
            float _value = ((i*173+j*51) % 100) / 100.;
            if (_value < p)
            {
                u[0][i][j] = true;
            }
        }
    rand();
}