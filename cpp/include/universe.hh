size_t GetNActiveNeighbours(bool*** u, int n, int dx, int dy, int nx, int ny);
void Evolve(bool ***u, int n, int dx, int dy);
bool ***allocate_universe(int n, int dx, int dy);
void set_initial_conditions(bool*** u, int dx, int dy, float p);