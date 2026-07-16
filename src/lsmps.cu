/**
 * LS-MPS GPU Single-Timestep Core Implementation
 * ===============================================
 *
 * Operator graph & I/O dimensions (one timestep):
 *
 *   INPUT: r^n[N,3], u^n[N,3], p^n[N]
 *     │
 *     ├──[1] NeighborSearch ──→ nbr_count[N], nbr_list[N,MAX_NBR]
 *     │    Input:  r[N,3]     Output: neighbor topology
 *     │
 *     ├──[2] Density ──→ n[N] (particle number density)
 *     │    Input:  r[N,3], neighbors
 *     │
 *     ├──[3] Gradient ──→ C[N,9,9], u*[N,3], r*[N,3], div_u[N]
 *     │    Input:  r,u,p, neighbors
 *     │    Work:   build 9x9 corrective matrix per particle,
 *     │            compute ∇u, ∇·u, ∇²u, then u*=u+dt*(ν∇²u+g), r*=r+dt*u*
 *     │
 *     ├──[4] PressureMatrix ──→ A[N,N] sparse CSR, b[N] (PPE source)
 *     │    Input:  C, neighbors, n, n0, div_u, r
 *     │    Work:   assemble Laplacian operator matrix for pressure
 *     │
 *     ├──[5] BiCGSTAB ──→ p[N] (updated pressure)
 *     │    Input:  A, b           Solves A·p = b
 *     │
 *     ├──[6] Update ──→ u^{n+1}[N,3], r^{n+1}[N,3]
 *     │    Input:  p, C, u*, r, neighbors
 *     │    Work:   u_new = u* - dt/ρ·∇p, r_new = r + dt·u_new
 *     │
 *   OUTPUT: r^{n+1}[N,3], u^{n+1}[N,3], p^{n+1}[N]
 *
 * Kernel-to-operator mapping (for nsys profiler):
 *   kernel_build_neighbors       → NeighborSearch
 *   kernel_compute_density       → Density
 *   kernel_compute_cmatrix       → Gradient (matrix build)
 *   kernel_explicit_step         → Gradient (explicit update)
 *   kernel_assemble_ppe          → PressureMatrix
 *   kernel_spmv_csr + bicgstab   → BiCGSTAB
 *   kernel_update                → Update
 *
 * Reference: Kong et al., Computational Particle Mechanics (2024) 11:627-641
 * LSMPS formulation: Duan et al., Int J Numer Methods Fluids (2021) 93:148-175
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

// ============================================================================
// Compile-time limits
// ============================================================================
#define MAX_PARTICLES      4194304   // 4M max particles
#define MAX_NEIGHBORS      64        // enough for re=2.1*l0 in 3D (~39 avg + margin)
#define MATRIX_SIZE        9
#define BICGSTAB_MAX_ITER  500
#define BICGSTAB_TOL       1.0e-6f
#define BLOCK_SIZE         256

#define PTYPE_FLUID   0
#define PTYPE_WALL    1

// ============================================================================
// CUDA error check
// ============================================================================
#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Config
// ============================================================================
typedef struct {
    char name[128];
    char case_type[32];   // "hydrostatic", "couette", "dam_break"
    float l0, re, rho, nu;
    float gx, gy, gz;
    float alpha, dt;
    float wall_speed;     // Couette: top wall speed [m/s]
    float n0_ref;         // Reference particle number density (computed from initial config)
    int   num_steps, nx, ny, nz;
    int   verbose, check_ref;
} Config;

static void trim(char *s) {
    char *p = s; while (*p==' '||*p=='\t') p++;
    size_t len = strlen(p);
    while (len>0 && (p[len-1]==' '||p[len-1]=='\t'||p[len-1]=='\n'||p[len-1]=='\r')) p[--len]=0;
    memmove(s, p, len+1);
}

int parse_config(const char *fn, Config *c) {
    FILE *f = fopen(fn, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fn); return -1; }
    memset(c, 0, sizeof(Config));
    strcpy(c->name, "default");
    strcpy(c->case_type, "hydrostatic");
    c->l0=0.02f; c->re=0.042f; c->rho=1000.0f; c->nu=1.0e-6f;
    c->gz=-9.8f; c->alpha=0.02f; c->dt=0.0001f; c->num_steps=1;
    c->nx=5; c->ny=5; c->nz=7; c->verbose=1; c->check_ref=1;
    c->wall_speed=1.0f;

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char *cmt = strchr(line, '#'); if (cmt) *cmt=0;
        trim(line); if (!line[0]) continue;
        char *col = strchr(line, ':'); if (!col) continue;
        *col = 0;
        char key[128], val[128];
        strncpy(key, line, 127); trim(key);
        strncpy(val, col+1, 127); trim(val);

        if      (!strcmp(key,"name"))    strncpy(c->name, val, 127);
        else if (!strcmp(key,"case_type")) strncpy(c->case_type, val, 31);
        else if (!strcmp(key,"l0"))      c->l0 = strtof(val,NULL);
        else if (!strcmp(key,"re"))      c->re = strtof(val,NULL);
        else if (!strcmp(key,"rho"))     c->rho = strtof(val,NULL);
        else if (!strcmp(key,"nu"))      c->nu = strtof(val,NULL);
        else if (!strcmp(key,"gx"))      c->gx = strtof(val,NULL);
        else if (!strcmp(key,"gy"))      c->gy = strtof(val,NULL);
        else if (!strcmp(key,"gz"))      c->gz = strtof(val,NULL);
        else if (!strcmp(key,"alpha"))   c->alpha = strtof(val,NULL);
        else if (!strcmp(key,"dt"))      c->dt = strtof(val,NULL);
        else if (!strcmp(key,"wall_speed")) c->wall_speed = strtof(val,NULL);
        else if (!strcmp(key,"steps"))   c->num_steps = (int)strtol(val,NULL,10);
        else if (!strcmp(key,"nx"))      c->nx = (int)strtol(val,NULL,10);
        else if (!strcmp(key,"ny"))      c->ny = (int)strtol(val,NULL,10);
        else if (!strcmp(key,"nz"))      c->nz = (int)strtol(val,NULL,10);
        else if (!strcmp(key,"verbose")) c->verbose = !strcmp(val,"true")||!strcmp(val,"1");
        else if (!strcmp(key,"check_ref")) c->check_ref = !strcmp(val,"true")||!strcmp(val,"1");
    }
    fclose(f); return 0;
}

void print_config(const Config *c) {
    printf("=== LS-MPS Config: %s ===\n", c->name);
    printf("  l0=%.4f re=%.4f rho=%.1f nu=%.1e\n", c->l0, c->re, c->rho, c->nu);
    printf("  g=(%.1f,%.1f,%.1f) alpha=%.2f dt=%.4f steps=%d\n",
           c->gx, c->gy, c->gz, c->alpha, c->dt, c->num_steps);
    printf("  grid=%dx%dx%d\n\n", c->nx, c->ny, c->nz);
}

// ============================================================================
// Particle data: SoA layout on device, mirrored on host for verification
// ============================================================================
typedef struct {
    int N;
    // device
    float *rx,*ry,*rz, *ux,*uy,*uz, *p, *n, *n0;
    int   *type, *nbr_cnt, *nbr_list;
    float *C;           // [N*81] corrective matrices
    float *uxs,*uys,*uzs, *rxs,*rys,*rzs, *divu; // temporary step
    float *A_val, *b;   // PPE matrix (dense row-major for now) and RHS
    int   *A_col;       // column indices
    float *A_diag;      // diagonal entries
    float *inv_sqrt_diag; // 1/sqrt(|A_ii|)
    // ILU(0) preconditioner: CSR + level scheduling
    int   *ilu_row_ptr;  // [N+1] CSR row pointers
    int   *ilu_col_idx;  // [nnz] column indices
    float *ilu_val;      // [nnz] L+U factors (in-place)
    int   *ilu_level_ptr;// [n_levels+1]
    int   *ilu_level_rows;// [N] rows sorted by level
    int   ilu_nnz, ilu_n_levels;
    // cuSPARSE: SpMV + ILU preconditioner
    cusparseHandle_t   cusparse_handle;
    cusparseSpMatDescr_t sp_mat;    // CSR matrix for SpMV
    cusparseDnVecDescr_t dn_x, dn_y, dn_tmp; // dense vectors
    void   *spmv_buf;  size_t spmv_buf_size;
    int    *csr_rpt;    // [N+1] CSR row pointers (device)
    int    *csr_col;    // [nnz] CSR column indices (device)
    float  *csr_val;    // [nnz] CSR values (device, modified by ILU)
    int    csr_nnz;
    // ILU(0) via cuSPARSE
    cusparseMatDescr_t ilu_descr;   // general descriptor for ILU
    csrilu02Info_t     ilu_info;
    void   *ilu_buf;  int    ilu_buf_size;
    float *r_bcg,*r0_bcg,*p_bcg,*v_bcg,*s_bcg,*t_bcg; // BiCGSTAB
    int   *cell_id, *cell_head, *cell_next; // grid linked-list
    float *tmp1,*tmp2, *dot_result; // reduction workspace
    void  *cub_temp;    // CUB temporary storage
    size_t cub_temp_bytes;
    // host mirror
    float *h_rx,*h_ry,*h_rz, *h_ux,*h_uy,*h_uz, *h_p, *h_n;
    int   *h_type;
} Particles;

// ============================================================================
// Memory helpers
// ============================================================================
void alloc_dev(Particles *p) {
    int N = p->N;
    CHECK(cudaMalloc(&p->rx, N*sizeof(float)));
    CHECK(cudaMalloc(&p->ry, N*sizeof(float)));
    CHECK(cudaMalloc(&p->rz, N*sizeof(float)));
    CHECK(cudaMalloc(&p->ux, N*sizeof(float)));
    CHECK(cudaMalloc(&p->uy, N*sizeof(float)));
    CHECK(cudaMalloc(&p->uz, N*sizeof(float)));
    CHECK(cudaMalloc(&p->p, N*sizeof(float)));
    CHECK(cudaMalloc(&p->n, N*sizeof(float)));
    CHECK(cudaMalloc(&p->n0, N*sizeof(float)));
    CHECK(cudaMalloc(&p->type, N*sizeof(int)));
    CHECK(cudaMalloc(&p->nbr_cnt, N*sizeof(int)));
    CHECK(cudaMalloc(&p->nbr_list, N*MAX_NEIGHBORS*sizeof(int)));
    CHECK(cudaMalloc(&p->C, N*81*sizeof(float)));
    // Grid linked-list arrays (num_cells bounded by N)
    CHECK(cudaMalloc(&p->cell_id, N*sizeof(int)));
    CHECK(cudaMalloc(&p->cell_head, N*sizeof(int)));  // at most N non-empty cells
    CHECK(cudaMalloc(&p->cell_next, N*sizeof(int)));
    CHECK(cudaMalloc(&p->uxs, N*sizeof(float)));
    CHECK(cudaMalloc(&p->uys, N*sizeof(float)));
    CHECK(cudaMalloc(&p->uzs, N*sizeof(float)));
    CHECK(cudaMalloc(&p->rxs, N*sizeof(float)));
    CHECK(cudaMalloc(&p->rys, N*sizeof(float)));
    CHECK(cudaMalloc(&p->rzs, N*sizeof(float)));
    CHECK(cudaMalloc(&p->divu, N*sizeof(float)));
    // PPE: row-major dense-per-row with max neighbors+1
    CHECK(cudaMalloc(&p->A_val, N*(MAX_NEIGHBORS+1)*sizeof(float)));
    CHECK(cudaMalloc(&p->A_col, N*(MAX_NEIGHBORS+1)*sizeof(int)));
    CHECK(cudaMalloc(&p->A_diag, N*sizeof(float)));
    CHECK(cudaMalloc(&p->inv_sqrt_diag, N*sizeof(float)));
    CHECK(cudaMalloc(&p->b, N*sizeof(float)));
    CHECK(cudaMalloc(&p->r_bcg, N*sizeof(float)));
    CHECK(cudaMalloc(&p->r0_bcg, N*sizeof(float)));
    CHECK(cudaMalloc(&p->p_bcg, N*sizeof(float)));
    CHECK(cudaMalloc(&p->v_bcg, N*sizeof(float)));
    CHECK(cudaMalloc(&p->s_bcg, N*sizeof(float)));
    CHECK(cudaMalloc(&p->t_bcg, N*sizeof(float)));
    CHECK(cudaMalloc(&p->tmp1, N*sizeof(float)));
    CHECK(cudaMalloc(&p->tmp2, N*sizeof(float)));
    // CUB reduction: single float result + temp storage
    CHECK(cudaMalloc(&p->dot_result, sizeof(float)));
    // Query CUB temp storage size and allocate
    cub::DeviceReduce::Sum(NULL, p->cub_temp_bytes,
                           (const float*)NULL, (float*)NULL, N);
    CHECK(cudaMalloc(&p->cub_temp, p->cub_temp_bytes));
}

void h2d(Particles *p) {
    int N = p->N;
    CHECK(cudaMemcpy(p->rx, p->h_rx, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->ry, p->h_ry, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->rz, p->h_rz, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->ux, p->h_ux, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->uy, p->h_uy, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->uz, p->h_uz, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->p, p->h_p, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->type, p->h_type, N*sizeof(int), cudaMemcpyHostToDevice));
}

void d2h(Particles *p) {
    int N = p->N;
    CHECK(cudaMemcpy(p->h_rx, p->rx, N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p->h_ry, p->ry, N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p->h_rz, p->rz, N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p->h_ux, p->ux, N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p->h_uy, p->uy, N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p->h_uz, p->uz, N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p->h_p, p->p, N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p->h_n, p->n, N*sizeof(float), cudaMemcpyDeviceToHost));
}

void free_particles(Particles *p) {
    free(p->h_rx);free(p->h_ry);free(p->h_rz);
    free(p->h_ux);free(p->h_uy);free(p->h_uz);
    free(p->h_p);free(p->h_n);free(p->h_type);
    cudaFree(p->rx);cudaFree(p->ry);cudaFree(p->rz);
    cudaFree(p->ux);cudaFree(p->uy);cudaFree(p->uz);
    cudaFree(p->p);cudaFree(p->n);cudaFree(p->n0);cudaFree(p->type);
    cudaFree(p->nbr_cnt);cudaFree(p->nbr_list);cudaFree(p->C);
    cudaFree(p->uxs);cudaFree(p->uys);cudaFree(p->uzs);
    cudaFree(p->rxs);cudaFree(p->rys);cudaFree(p->rzs);cudaFree(p->divu);
    cudaFree(p->A_val);cudaFree(p->A_col);cudaFree(p->A_diag);cudaFree(p->inv_sqrt_diag);cudaFree(p->b);
    cudaFree(p->cell_id);cudaFree(p->cell_head);cudaFree(p->cell_next);
    cudaFree(p->r_bcg);cudaFree(p->r0_bcg);cudaFree(p->p_bcg);
    cudaFree(p->v_bcg);cudaFree(p->s_bcg);cudaFree(p->t_bcg);
    cudaFree(p->tmp1);cudaFree(p->tmp2);
    cudaFree(p->dot_result);cudaFree(p->cub_temp);
    // cuSPARSE cleanup
    if(p->cusparse_handle){ cusparseDestroy(p->cusparse_handle); }
    if(p->sp_mat) cusparseDestroySpMat(p->sp_mat);
    if(p->dn_x) cusparseDestroyDnVec(p->dn_x);
    if(p->dn_y) cusparseDestroyDnVec(p->dn_y);
    if(p->dn_tmp) cusparseDestroyDnVec(p->dn_tmp);
    cudaFree(p->spmv_buf);
    cudaFree(p->csr_rpt); cudaFree(p->csr_col); cudaFree(p->csr_val);
    if(p->ilu_descr) cusparseDestroyMatDescr(p->ilu_descr);
    if(p->ilu_info) cusparseDestroyCsrilu02Info(p->ilu_info);
    cudaFree(p->ilu_buf);
    cudaFree(p->ilu_level_ptr); cudaFree(p->ilu_level_rows);
    if(p->ilu_row_ptr) cudaFree(p->ilu_row_ptr);
    if(p->ilu_col_idx) cudaFree(p->ilu_col_idx);
    if(p->ilu_val)     cudaFree(p->ilu_val);
}

// ============================================================================
// Particle initialization: multi-case support
// ============================================================================
void init_particles(Particles *p, const Config *c) {
    int nx=c->nx, ny=c->ny, nz=c->nz;
    float l0=c->l0;
    int total = nx*ny*nz;
    float H = (nz-1)*l0;

    p->N = total;
    p->h_rx=(float*)malloc(total*sizeof(float));
    p->h_ry=(float*)malloc(total*sizeof(float));
    p->h_rz=(float*)malloc(total*sizeof(float));
    p->h_ux=(float*)calloc(total,sizeof(float));
    p->h_uy=(float*)calloc(total,sizeof(float));
    p->h_uz=(float*)calloc(total,sizeof(float));
    p->h_p=(float*)calloc(total,sizeof(float));
    p->h_n=(float*)calloc(total,sizeof(float));
    p->h_type=(int*)malloc(total*sizeof(int));

    int is_couette   = !strcmp(c->case_type, "couette");
    int is_dam_break = !strcmp(c->case_type, "dam_break");

    int idx=0;
    float rho_g = c->rho * fabsf(c->gz);
    for (int iz=0; iz<nz; iz++) {
        for (int iy=0; iy<ny; iy++) {
            for (int ix=0; ix<nx; ix++) {
                float x = ix*l0, y = iy*l0, z = iz*l0;
                p->h_rx[idx]=x; p->h_ry[idx]=y; p->h_rz[idx]=z;

                if (is_couette) {
                    // Couette flow: bottom wall (iz=0), top wall (iz=nz-1),
                    // no side walls → periodic-like in x,y
                    if (iz == 0 || iz == nz-1) {
                        p->h_type[idx]=PTYPE_WALL;
                        p->h_p[idx]=0;
                        // Top wall moves at wall_speed in +x direction
                        if (iz == nz-1) {
                            p->h_ux[idx] = c->wall_speed;
                        }
                    } else {
                        p->h_type[idx]=PTYPE_FLUID;
                        p->h_p[idx]=0;
                    }
                } else if (is_dam_break) {
                    // Dam break: fluid in left ~60% of domain, wall on left/bottom/right,
                    // no top wall → free surface
                    int is_wall = (iz==0) || (ix==0) || (ix==nx-1);
                    int in_fluid = (ix <= nx/2 + 1); // fluid in left half + 1 column
                    if (is_wall) {
                        p->h_type[idx]=PTYPE_WALL;
                        p->h_p[idx] = rho_g*(H-z); if(p->h_p[idx]<0)p->h_p[idx]=0;
                    } else if (in_fluid && iz < nz-1) {
                        // Fluid column on left side, with free surface at top
                        p->h_type[idx]=PTYPE_FLUID;
                        float fluid_H = (nz-2)*l0; // fluid column height
                        p->h_p[idx] = rho_g*(fluid_H - z);
                        if(p->h_p[idx]<0)p->h_p[idx]=0;
                    } else {
                        // Empty space (right side or above fluid)
                        // Mark as wall type with zero influence (ghost particle)
                        p->h_type[idx]=PTYPE_WALL;
                        p->h_ux[idx]=0; p->h_uy[idx]=0; p->h_uz[idx]=0;
                        p->h_p[idx]=0;
                    }
                } else {
                    // Default: hydrostatic box
                    int is_wall = (iz==0)||(iz==nz-1)||(ix==0)||(ix==nx-1)||(iy==0)||(iy==ny-1);
                    if (is_wall) {
                        p->h_type[idx]=PTYPE_WALL;
                        p->h_p[idx] = rho_g*(H-z);
                        if (p->h_p[idx]<0) p->h_p[idx]=0;
                    } else {
                        p->h_type[idx]=PTYPE_FLUID;
                        p->h_p[idx] = rho_g*(H-z);
                        if (p->h_p[idx]<0) p->h_p[idx]=0;
                    }
                }
                idx++;
            }
        }
    }
    // Compute reference particle number density n0
    // Find a central fluid particle and sum kernel weights to all neighbors within re
    float n0_computed = 0;
    float re2 = c->re * c->re;
    for (int i=0; i<total; i++) {
        if (p->h_type[i] == PTYPE_FLUID) {
            // Check if this particle is "interior" (far from walls)
            float cx = (nx-1)*l0/2, cy = (ny-1)*l0/2, cz = (nz-1)*l0/2;
            float dx_c = p->h_rx[i]-cx, dy_c = p->h_ry[i]-cy, dz_c = p->h_rz[i]-cz;
            if (dx_c*dx_c+dy_c*dy_c+dz_c*dz_c < 0.25f*l0*l0) { // near center
                float wsum = 0;
                for (int j=0; j<total; j++) {
                    if (i==j) continue;
                    float dx = p->h_rx[i]-p->h_rx[j];
                    float dy = p->h_ry[i]-p->h_ry[j];
                    float dz = p->h_rz[i]-p->h_rz[j];
                    float r2 = dx*dx+dy*dy+dz*dz;
                    if (r2 < re2 && r2 > 1e-10f) {
                        float r = sqrtf(r2);
                        float q = c->re/r - 1.f;
                        wsum += q*q;
                    }
                }
                n0_computed = wsum;
                break;
            }
        }
    }
    if (n0_computed < 1.f) n0_computed = 12.f; // fallback for tiny configs
    // Store n0_ref in config (const cast needed since init_particles takes const Config*)
    *(float*)&c->n0_ref = n0_computed;

    int nf=0, nw=0;
    for (int i=0;i<total;i++) { if(p->h_type[i]==PTYPE_FLUID)nf++; else nw++; }
    printf("Case: %s\n", c->case_type);
    printf("  n0_ref: %.2f (computed from initial uniform config)\n", n0_computed);
    printf("Particles: total=%d fluid=%d wall=%d (%.0f%% wall)\n",
           total, nf, nw, 100.f*nw/total);
    printf("Domain: [0,%.4f]x[0,%.4f]x[0,%.4f]  H=%.4f  p_max=%.1f\n",
           (nx-1)*l0,(ny-1)*l0,H,H,rho_g*H);
    if (is_couette) {
        printf("Couette: top wall speed U=%.2f m/s, gap=%.4f m\n",
               c->wall_speed, (nz-2)*l0);
        printf("  Expected: linear velocity profile u(z)=%.2f*z/%.4f\n",
               c->wall_speed, (nz-2)*l0);
    }
}

// ============================================================================
// GPU Timer + Accumulator for structured profiling
// ============================================================================
#define N_OPS 6
static const char *op_names[N_OPS] = {
    "1_NeighborSearch", "2_Density", "3_Gradient",
    "4_PressureMatrix", "5_BiCGSTAB", "6_Update"
};
static float op_total_ms[N_OPS] = {0};
static int   op_calls[N_OPS] = {0};
static int   op_bicg_iters = 0;

typedef struct { cudaEvent_t s,e; int op_idx; } Timer;
void tic(Timer *t, int op_idx) {
    t->op_idx=op_idx; CHECK(cudaEventCreate(&t->s)); CHECK(cudaEventCreate(&t->e));
    CHECK(cudaEventRecord(t->s,0));
}
float toc(Timer *t) {
    float ms=0; CHECK(cudaEventRecord(t->e,0));
    CHECK(cudaEventSynchronize(t->e));
    CHECK(cudaEventElapsedTime(&ms, t->s, t->e));
    CHECK(cudaEventDestroy(t->s)); CHECK(cudaEventDestroy(t->e));
    op_total_ms[t->op_idx] += ms;
    op_calls[t->op_idx]++;
    if (t->op_idx < 0 || t->op_idx >= N_OPS) {} // safety
    printf("  [GPU] %-20s: %8.3f ms\n", op_names[t->op_idx], ms);
    return ms;
}
void print_structured_profile(int N_particles, int n_steps) {
    float total = 0;
    for (int i=0;i<N_OPS;i++) total += op_total_ms[i];
    printf("\n=== Structured Profile (Industrial Format) ===\n");
    printf("%%Time  Total_ms  Calls  Avg_ms  Module\n");
    printf("------ --------- ------ ------- ---------------------\n");
    for (int i=0;i<N_OPS;i++) {
        float pct = (total>0)?100.f*op_total_ms[i]/total:0;
        float avg = (op_calls[i]>0)?op_total_ms[i]/op_calls[i]:0;
        printf("%5.1f  %9.1f  %5d  %6.1f  %s\n",
               pct, op_total_ms[i], op_calls[i], avg, op_names[i]);
        // For BiCGSTAB, also print avg iterations
        if (i==4 && op_calls[i]>0) {
            printf("  (avg %.1f BiCGSTAB iters/step)\n",
                   (float)op_bicg_iters/op_calls[i]);
        }
    }
    printf("------ --------- ------ ------- ---------------------\n");
    printf("         %9.1f                       TOTAL\n", total);
    float mpps = (float)N_particles * n_steps / (total * 1e-3f) * 1e-6f;
    printf("  Particles: %d  Steps: %d  Throughput: %.3f Mpart/s\n",
           N_particles, n_steps, mpps);
    printf("==================================================\n");
}
double wall_time() { struct timeval tv; gettimeofday(&tv,NULL); return tv.tv_sec+tv.tv_usec*1e-6; }

// ============================================================================
// Device functions
// ============================================================================
__device__ inline float kern_w(float r, float re) {
    if (r<=1e-8f||r>=re) return 0.f;
    float q = re/r - 1.f;
    return q*q;
}

__device__ inline float inv_safe(float x) { return 1.f/fmaxf(x,1e-8f); }

// Build 9-element polynomial basis P(r_ij) - Eq.(10) in Kong et al.
__device__ void basis(float dx, float dy, float dz, float r,
                      float l0, float np, int jtype, float *P) {
    float ir = inv_safe(r), il = 1.f/l0;
    if (jtype == PTYPE_FLUID || jtype == PTYPE_WALL) {
        P[0]=dx*ir; P[1]=dy*ir; P[2]=dz*ir;
        P[3]=dx*dx*ir*il/(2.f*np);
        P[4]=dy*dy*ir*il/(2.f*np);
        P[5]=dz*dz*ir*il/(2.f*np);
        P[6]=dx*dy*ir*il;
        P[7]=dx*dz*ir*il;
        P[8]=dy*dz*ir*il;
    } else {
        // Neumann: use default normal (0,0,1)
        P[0]=0; P[1]=0; P[2]=1;
        P[3]=0; P[4]=0; P[5]=dz*il;
        P[6]=0; P[7]=dx*il; P[8]=dy*il;
    }
}

// Solve 9x9 linear system M*x=b via Gaussian elimination with pivoting
// M is col-major (9x9), b[9], result in x[9]
__device__ int solve9(float *M, float *b, float *x) {
    float A[9][10];
    for (int i=0;i<9;i++) {
        for (int j=0;j<9;j++) A[i][j]=M[j*9+i];
        A[i][9]=b[i];
    }
    for (int col=0;col<9;col++) {
        int best=col; float mv=fabsf(A[col][col]);
        for (int r=col+1;r<9;r++) { float v=fabsf(A[r][col]); if(v>mv){mv=v;best=r;} }
        if (mv<1e-12f) { for(int i=0;i<9;i++)x[i]=0; return 0; }
        if (best!=col) for(int j=col;j<10;j++) { float t=A[col][j]; A[col][j]=A[best][j]; A[best][j]=t; }
        float piv=A[col][col];
        for(int r=col+1;r<9;r++) { float f=A[r][col]/piv; for(int j=col;j<10;j++) A[r][j]-=f*A[col][j]; }
    }
    for(int i=8;i>=0;i--) {
        float s=A[i][9];
        for(int j=i+1;j<9;j++) s-=A[i][j]*x[j];
        x[i]=s/A[i][i];
    }
    return 1;
}

// ============================================================================
// Kernel 1: NeighborSearch (uniform grid with linked-list per cell)
// ============================================================================

// Step 1a: compute cell index and linked-list "next" pointer per particle
__global__ void kernel_cell_link(
    const float *rx, const float *ry, const float *rz,
    int *cell_id, int *cell_head, int *cell_next, int N,
    float cell_w, int ncx, int ncy)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;
    int cx=(int)floorf(rx[i]/cell_w);
    int cy=(int)floorf(ry[i]/cell_w);
    int cz=(int)floorf(rz[i]/cell_w);
    int cid = cz*(ncx*ncy) + cy*ncx + cx;
    cell_id[i] = cid;
    // Atomically insert into linked list head
    int old = atomicExch(&cell_head[cid], i);
    cell_next[i] = old;
}

// Step 1b: build neighbor lists by traversing cell linked lists
__global__ void kernel_build_neighbors(
    const float *rx, const float *ry, const float *rz,
    const int *cell_id, const int *cell_head, const int *cell_next,
    int *nbr_cnt, int *nbr_list, int N,
    float re, float re2, float cell_w, int ncx, int ncy, int ncz)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;

    float xi=rx[i], yi=ry[i], zi=rz[i];
    int ci = (int)floorf(xi/cell_w);
    int cj = (int)floorf(yi/cell_w);
    int ck = (int)floorf(zi/cell_w);

    int cnt=0;
    // Search 27-cell neighborhood
    for (int dk=-1; dk<=1 && cnt<MAX_NEIGHBORS; dk++) {
        for (int dj=-1; dj<=1 && cnt<MAX_NEIGHBORS; dj++) {
            for (int di=-1; di<=1 && cnt<MAX_NEIGHBORS; di++) {
                int nc=ci+di, nj=cj+dj, nk=ck+dk;
                if (nc<0||nc>=ncx||nj<0||nj>=ncy||nk<0||nk>=ncz) continue;
                int cid = nk*(ncx*ncy) + nj*ncx + nc;
                // Traverse linked list for this cell
                for (int j=cell_head[cid]; j!=-1 && cnt<MAX_NEIGHBORS; j=cell_next[j]) {
                    if (i==j) continue;
                    float dx=xi-rx[j], dy=yi-ry[j], dz=zi-rz[j];
                    if (dx*dx+dy*dy+dz*dz < re2) {
                        nbr_list[i*MAX_NEIGHBORS+cnt] = j;
                        cnt++;
                    }
                }
            }
        }
    }
    nbr_cnt[i] = cnt;
}

// ============================================================================
// Kernel 2: Density
// ============================================================================
__global__ void kernel_compute_density(
    const float *rx, const float *ry, const float *rz,
    const int *nbr_cnt, const int *nbr_list,
    float *nd, int N, float re)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return;
    float xi=rx[i],yi=ry[i],zi=rz[i];
    float s=0; int nc=nbr_cnt[i];
    for(int k=0;k<nc;k++) {
        int j=nbr_list[i*MAX_NEIGHBORS+k];
        float dx=xi-rx[j],dy=yi-ry[j],dz=zi-rz[j];
        s+=kern_w(sqrtf(dx*dx+dy*dy+dz*dz), re);
    }
    nd[i]=s;
}

// ============================================================================
// Kernel 3: Corrective matrix + explicit step (Gradient operator)
// ============================================================================
__global__ void kernel_cmatrix_and_explicit(
    const float *rx, const float *ry, const float *rz,
    const float *ux, const float *uy, const float *uz,
    const float *p, const int *type,
    const int *nbr_cnt, const int *nbr_list,
    float *C, float *uxs, float *uys, float *uzs,
    float *rxs, float *rys, float *rzs,
    float *divu, const float *nd, const float *n0d,
    int N, float re, float l0, float dt, float nu,
    float gx, float gy, float gz, float np)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return;

    if(type[i]==PTYPE_WALL) {
        uxs[i]=0;uys[i]=0;uzs[i]=0;
        rxs[i]=rx[i];rys[i]=ry[i];rzs[i]=rz[i];
        divu[i]=0;
        for(int k=0;k<81;k++) C[i*81+k]=0;
        return;
    }

    float xi=rx[i],yi=ry[i],zi=rz[i];
    int nc=nbr_cnt[i];
    // Use initial reference density n0 for normalization (consistent with PPE assembly)
    float nd_i = fmaxf(n0d[i],1.f);
    float n0_inv = 1.f/nd_i;

    // Build moment matrix M = sum w * P * P^T  (9x9 col-major)
    float M[81]={0};
    for(int k=0;k<nc;k++) {
        int j=nbr_list[i*MAX_NEIGHBORS+k];
        float dx=xi-rx[j],dy=yi-ry[j],dz=zi-rz[j];
        float r=sqrtf(dx*dx+dy*dy+dz*dz);
        float w=kern_w(r,re);
        if(w<=0)continue;
        float P[9]; basis(dx,dy,dz,r,l0,np,type[j],P);
        for(int p=0;p<9;p++) for(int q=0;q<9;q++) M[q*9+p]+=w*P[p]*P[q];
    }
    for(int k=0;k<81;k++) M[k]*=n0_inv;

    // Store C = M^{-1} directly (9 columns, col-major)
    int C_off=i*81;

    for(int col=0;col<9;col++) {
        float e[9]={0}, x[9]; e[col]=1.f;
        solve9(M,e,x);
        for(int r=0;r<9;r++) C[C_off+col*9+r]=x[r];
    }

    // Now compute explicit update: gradient of velocity components, Laplacian
    // RHS for gradient: b_k = sum_j w * P_k * (phi_i-phi_j)/|r_ij| / n0
    float bu[9]={0}, bv[9]={0}, bw[9]={0};
    for(int k=0;k<nc;k++) {
        int j=nbr_list[i*MAX_NEIGHBORS+k];
        float dx=xi-rx[j],dy=yi-ry[j],dz=zi-rz[j];
        float r=sqrtf(dx*dx+dy*dy+dz*dz);
        float w=kern_w(r,re);
        if(w<=0)continue;
        float ir=inv_safe(r);
        float du=(ux[i]-ux[j])*ir, dv=(uy[i]-uy[j])*ir, dw=(uz[i]-uz[j])*ir;
        float P[9]; basis(dx,dy,dz,r,l0,np,type[j],P);
        for(int p=0;p<9;p++) {
            float wp = w*P[p]*n0_inv;
            bu[p]+=wp*du; bv[p]+=wp*dv; bw[p]+=wp*dw;
        }
    }

    // Gradient = C[0:3]^T * b  → ∂/∂x, ∂/∂y, ∂/∂z
    float du_dx=0,dv_dy=0,dw_dz=0;
    for(int p=0;p<9;p++) {
        du_dx += C[C_off+0*9+p]*bu[p];
        dv_dy += C[C_off+1*9+p]*bv[p];
        dw_dz += C[C_off+2*9+p]*bw[p];
    }
    divu[i] = du_dx + dv_dy + dw_dz;

    // Laplacian: 2*d/n0 * sum w*dphi*(C4+C5+C6)·P / l0^2
    // C456 = C4 + C5 + C6
    float lap_u=0,lap_v=0,lap_w=0;
    for(int k=0;k<nc;k++) {
        int j=nbr_list[i*MAX_NEIGHBORS+k];
        float dx=xi-rx[j],dy=yi-ry[j],dz=zi-rz[j];
        float r=sqrtf(dx*dx+dy*dy+dz*dz);
        float w=kern_w(r,re);
        if(w<=0)continue;
        float ir=inv_safe(r);
        float du=(ux[i]-ux[j])*ir, dv=(uy[i]-uy[j])*ir, dw=(uz[i]-uz[j])*ir;
        float P[9]; basis(dx,dy,dz,r,l0,np,type[j],P);
        float cdot=0;
        for(int p=0;p<9;p++) cdot+=(C[C_off+3*9+p]+C[C_off+4*9+p]+C[C_off+5*9+p])*P[p];
        lap_u+=w*du*cdot; lap_v+=w*dv*cdot; lap_w+=w*dw*cdot;
    }
    float ls = 6.f*n0_inv/(l0*l0); // 2d/(n0*l0^2) with d=3
    lap_u*=ls; lap_v*=ls; lap_w*=ls;

    // Explicit update: u* = u + dt*(nu*lap + g)
    uxs[i]=ux[i]+dt*(nu*lap_u+gx);
    uys[i]=uy[i]+dt*(nu*lap_v+gy);
    uzs[i]=uz[i]+dt*(nu*lap_w+gz);

    // r* = r + dt*u*
    rxs[i]=rx[i]+dt*uxs[i];
    rys[i]=ry[i]+dt*uys[i];
    rzs[i]=rz[i]+dt*uzs[i];
}

// ============================================================================
// Kernel 4: PressureMatrix assembly
// ============================================================================
__global__ void kernel_assemble_ppe(
    const float *rx, const float *ry, const float *rz,
    const int *type, const int *nbr_cnt, const int *nbr_list,
    const float *C, const float *nd, const float *n0d,
    const float *divu,
    float *Av, int *Ac, float *Adiag, float *b,
    int N, float re, float l0, float rho, float dt, float alpha, float np)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return;

    float xi=rx[i],yi=ry[i],zi=rz[i];
    int nc=nbr_cnt[i];
    float nd_i=fmaxf(nd[i],1.f), n0_i=fmaxf(n0d[i],1.f);
    float n0_inv=1.f/n0_i;
    // Scale Laplacian by dt²/rho for numerical stability
    // Original: A_ij = 6/(n0*l0²)*w/r*(C4+C5+C6)·P,  b = α*ρ/dt*div + (1-α)*ρ/dt²*(n0-n)/n0
    // Scaled:   A'_ij = dt²/ρ * A_ij,                   b'= α*dt*div + (1-α)*(n0-n)/n0
    // Both give same p, but coefficients stay in float range
    float dt2_over_rho = (dt*dt)/rho;
    float ls = 6.f*n0_inv/(l0*l0) * dt2_over_rho;
    int C_off=i*81;
    int row_s=i*(MAX_NEIGHBORS+1);

    int col=0; float diag=0;

    for(int k=0;k<nc;k++) {
        int j=nbr_list[i*MAX_NEIGHBORS+k];
        float dx=xi-rx[j],dy=yi-ry[j],dz=zi-rz[j];
        float r=sqrtf(dx*dx+dy*dy+dz*dz);
        float w=kern_w(r,re);
        if(w<=0)continue;
        float ir=inv_safe(r);
        float P[9]; basis(dx,dy,dz,r,l0,np,type[j],P);
        float c4c5c6=0;
        for(int p=0;p<9;p++) c4c5c6+=(C[C_off+3*9+p]+C[C_off+4*9+p]+C[C_off+5*9+p])*P[p];
        float aij = ls*w*ir*c4c5c6;
        Av[row_s+col]=aij; Ac[row_s+col]=j;
        diag -= aij;
        col++;
    }
    // Diagonal
    Av[row_s+col]=diag; Ac[row_s+col]=i;
    Adiag[i] = diag;  // store for Jacobi preconditioner
    col++;

    if(type[i]==PTYPE_WALL) {
        // Wall Dirichlet BC: fix pressure at initial hydrostatic value
        // Identity row: A_ii=1, others=0, b_i = initial wall pressure
        // (p is already initialized with hydrostatic; keep it unchanged)
        for(int k=0;k<col;k++) { Av[row_s+k]=(Ac[row_s+k]==i)?1.f:0.f; }
        b[i] = 0;  // dp=0 for walls: pressure correction is zero
    } else {
        // PPE source (scaled by dt²/rho for numerical stability):
        // Scaled b' = alpha*dt*div(u*) + (1-alpha)*(n0-n*)/n0
        // From Kong et al. (2024) Eq. (14), divided by rho/dt²
        float s1 = alpha*dt*divu[i];
        float s2 = (1.f-alpha)*(n0_i-nd_i)*n0_inv;
        b[i] = s1+s2;
        // Regularize diagonal for stability (add epsilon * max|A_ij|)
        float max_aij = 0.f;
        for(int k=0;k<col-1;k++) { float v=fabsf(Av[row_s+k]); if(v>max_aij)max_aij=v; }
        Av[row_s+col-1] += 1.0e-6f * max_aij;
    }
}

// ============================================================================
// Kernel 5: BiCGSTAB kernels (SpMV, dot, axpy, etc.)
// ============================================================================
// Compute inv_sqrt(|diag|) from stored diagonal
__global__ void k_compute_inv_sqrt_diag(const float *diag, float *inv_sqrt, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float d = fabsf(diag[i]);
    inv_sqrt[i] = (d > 1.0e-10f) ? rsqrtf(d) : 1.0f;
}

// Symmetric Jacobi preconditioned SpMV:
// Polynomial preconditioner: apply 2-step damped Jacobi to v, result in y
// y = ω*D^{-1}*v + ω*D^{-1}*(v - ω*A*D^{-1}*v)
__global__ void k_poly_pc(const float *Av, const int *Ac, const int *nbr_cnt,
                          const float *diag, const float *v, float *y, float *tmp,
                          float omega, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    int base = i*(MAX_NEIGHBORS+1), ncol = nbr_cnt[i]+1;
    float inv_d = (fabsf(diag[i]) > 1.0e-10f) ? 1.0f/diag[i] : 1.0f;

    // Step 1: w1 = ω * D^{-1} * v
    float w1 = omega * inv_d * v[i];

    // Step 2: compute A*w1 (need to read w1[j] for neighbors)
    float Aw1 = 0;
    for (int k = 0; k < ncol; k++) {
        int j = Ac[base+k];
        float w1_j = omega * (fabsf(diag[j]) > 1.0e-10f ? 1.0f/diag[j] : 1.0f) * v[j];
        Aw1 += Av[base+k] * w1_j;
    }

    // w2 = w1 + ω * D^{-1} * (v - Aw1)
    float w2 = w1 + omega * inv_d * (v[i] - Aw1);

    y[i] = w2;
    tmp[i] = w1; // store intermediate for potential reuse
}

// Plain SpMV (no PC): y = A * x
__global__ void k_spmv_plain(const float *Av, const int *Ac, const int *nbr_cnt,
                             const float *x, float *y, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0; int base = i*(MAX_NEIGHBORS+1), ncol = nbr_cnt[i]+1;
    #pragma unroll 4
    for(int k=0;k<ncol;k++) s += Av[base+k]*x[Ac[base+k]];
    y[i] = s;
}

// Symmetric Jacobi PC SpMV: y = D^{-1/2} * A * D^{-1/2} * x
// Applied inline: y[i] = sum_j A_ij * (x[j] * inv_sqrt_diag[j]) * inv_sqrt_diag[i]
__global__ void k_spmv_sym_jacobi(const float *Av, const int *Ac, const int *nbr_cnt,
                                   const float *inv_sqrt_diag, const float *x, float *y, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0;
    int base = i * (MAX_NEIGHBORS + 1);
    int ncol = nbr_cnt[i] + 1;
    #pragma unroll 4
    for (int k = 0; k < ncol; k++) {
        int col = Ac[base + k];
        s += Av[base + k] * (x[col] * inv_sqrt_diag[col]);
    }
    y[i] = s * inv_sqrt_diag[i];
}

// Scale vector by inverse diagonal: y[i] = x[i] / diag[i]
__global__ void k_scale_invdiag(float *y, const float *x, const float *diag, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return;
    float d = diag[i];
    y[i] = (fabsf(d) > 1.0e-10f) ? x[i] / d : x[i];
}

__global__ void k_axpy(float *y, const float *x, float a, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return; y[i]+=a*x[i];
}
__global__ void k_scal(float *x, float a, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return; x[i]*=a;
}
__global__ void k_copy(float *y, const float *x, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return; y[i]=x[i];
}
__global__ void k_set(float *x, float v, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return; x[i]=v;
}
__global__ void k_waxpby(float *z, const float *x, float a, const float *y, float b, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return; z[i]=a*x[i]+b*y[i];
}
__global__ void k_dot(const float *a, const float *b, float *partial, int N) {
    extern __shared__ float s[];
    int tid=threadIdx.x, i=blockIdx.x*blockDim.x+tid;
    s[tid]=(i<N)?a[i]*b[i]:0; __syncthreads();
    for(int st=blockDim.x/2;st>0;st>>=1){if(tid<st)s[tid]+=s[tid+st];__syncthreads();}
    if(tid==0)partial[blockIdx.x]=s[0];
}

// ============================================================================
// Forward declarations
// ============================================================================
__global__ void k_ilu_l_solve_level(const int*,const int*,const float*,
    const int*,int,int,const float*,float*,const float*);
__global__ void k_ilu_u_solve_level(const int*,const int*,const float*,
    const int*,int,int,const float*,float*);
int build_ilu0(Particles *p, int N);
void ilu_preconditioner_apply(Particles *p, const float *v, float *w,
                               float *y_tmp, int N, int nblk);

// ============================================================================
// cuSPARSE: Build CSR + SpMV + ILU(0) preconditioner
// ============================================================================
int cusparse_setup(Particles *p, int N) {
    cusparseHandle_t *h = &p->cusparse_handle;
    if (*h == NULL) cusparseCreate(h);

    // Step 1: Build device CSR from row-major PPE matrix
    // Copy nbr_cnt to host for computing nnz
    int *h_nbr_cnt = (int*)malloc(N*sizeof(int));
    CHECK(cudaMemcpy(h_nbr_cnt, p->nbr_cnt, N*sizeof(int), cudaMemcpyDeviceToHost));
    int nnz = 0;
    for (int i=0; i<N; i++) nnz += h_nbr_cnt[i] + 1;
    p->csr_nnz = nnz;

    // Build CSR row pointers
    int *h_rpt = (int*)malloc((N+1)*sizeof(int));
    h_rpt[0] = 0;
    for (int i=0; i<N; i++) h_rpt[i+1] = h_rpt[i] + h_nbr_cnt[i] + 1;
    free(h_nbr_cnt);

    // Allocate device CSR arrays (if not already allocated)
    if (p->csr_rpt == NULL) {
        CHECK(cudaMalloc(&p->csr_rpt, (N+1)*sizeof(int)));
        CHECK(cudaMalloc(&p->csr_col, nnz*sizeof(int)));
        CHECK(cudaMalloc(&p->csr_val, nnz*sizeof(float)));
    }
    CHECK(cudaMemcpy(p->csr_rpt, h_rpt, (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    // Keep h_rpt for ILU factorization

    // Convert row-major to CSR on device using a kernel
    int nblk = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    // We need a kernel that copies from row-major (i*(MAX_NEIGHBORS+1) base) to CSR
    // For now, copy to host, rearrange, copy back
    float *h_Av = (float*)malloc(N*(MAX_NEIGHBORS+1)*sizeof(float));
    int   *h_Ac = (int*)  malloc(N*(MAX_NEIGHBORS+1)*sizeof(int));
    CHECK(cudaMemcpy(h_Av, p->A_val, N*(MAX_NEIGHBORS+1)*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_Ac, p->A_col, N*(MAX_NEIGHBORS+1)*sizeof(int), cudaMemcpyDeviceToHost));
    int *h_nbr = (int*)malloc(N*sizeof(int));
    CHECK(cudaMemcpy(h_nbr, p->nbr_cnt, N*sizeof(int), cudaMemcpyDeviceToHost));

    // Compute inverse sqrt of diagonal for symmetric Jacobi: A_sym = D^{-1/2}*A*D^{-1/2}
    float *h_inv_sqrt_d = (float*)malloc(N*sizeof(float));
    for (int i=0; i<N; i++) {
        float d = 0;
        int base = i*(MAX_NEIGHBORS+1);
        for (int k=0; k<h_nbr[i]+1; k++) {
            if (h_Ac[base+k] == i) { d = h_Av[base+k]; break; }
        }
        h_inv_sqrt_d[i] = (fabsf(d) > 1.0e-10f) ? 1.0f/sqrtf(fabsf(d)) : 1.0f;
    }

    float *h_csr_val = (float*)malloc(nnz*sizeof(float));
    int   *h_csr_col = (int*)  malloc(nnz*sizeof(int));
    int pos = 0;
    for (int i=0; i<N; i++) {
        int ncol = h_nbr[i] + 1;
        int base = i * (MAX_NEIGHBORS + 1);
        float si = h_inv_sqrt_d[i];
        for (int k=0; k<ncol; k++) {
            int j = h_Ac[base+k];
            h_csr_col[pos] = j;
            // Symmetric Jacobi scaling: A_sym[i][j] = si * A[i][j] * sj
            h_csr_val[pos] = si * h_Av[base+k] * h_inv_sqrt_d[j];
            pos++;
        }
    }
    // Save inv_sqrt_diag for RHS scaling and solution recovery
    CHECK(cudaMemcpy(p->inv_sqrt_diag, h_inv_sqrt_d, N*sizeof(float), cudaMemcpyHostToDevice));
    free(h_inv_sqrt_d);
    free(h_Av); free(h_Ac); free(h_nbr);
    CHECK(cudaMemcpy(p->csr_col, h_csr_col, nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->csr_val, h_csr_val, nnz*sizeof(float), cudaMemcpyHostToDevice));

    // Step 2: Create cuSPARSE matrix descriptor for SpMV
    if (p->sp_mat != NULL) cusparseDestroySpMat(p->sp_mat);
    cusparseCreateCsr(&p->sp_mat, N, N, nnz,
                      p->csr_rpt, p->csr_col, p->csr_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Step 3: Set up SpMV buffer
    if (p->dn_x != NULL) cusparseDestroyDnVec(p->dn_x);
    if (p->dn_y != NULL) cusparseDestroyDnVec(p->dn_y);
    if (p->dn_tmp != NULL) cusparseDestroyDnVec(p->dn_tmp);
    cusparseCreateDnVec(&p->dn_x, N, NULL, CUDA_R_32F);
    cusparseCreateDnVec(&p->dn_y, N, NULL, CUDA_R_32F);
    cusparseCreateDnVec(&p->dn_tmp, N, NULL, CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMV_bufferSize(*h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, p->sp_mat, p->dn_x, &beta, p->dn_y,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                            &p->spmv_buf_size);
    if (p->spmv_buf_size > 0 && p->spmv_buf == NULL)
        CHECK(cudaMalloc(&p->spmv_buf, p->spmv_buf_size));

    // Step 4: Host-side ILU(0) factorization
    // Step 4: Host-side ILU(0) — simplified: just extract diagonal for Jacobi + scale rows
    // Full IKJ ILU deferred to V3; use diagonal scaling as robust first step
    for (int i = 0; i < N; i++) {
        int row_s = h_rpt[i], row_e = h_rpt[i+1];
        int len = row_e - row_s;
        float diag = 1.0f;
        for (int k = 0; k < len; k++) {
            if (h_csr_col[row_s+k] == i) { diag = h_csr_val[row_s+k]; break; }
        }
        float inv_d = (fabsf(diag) > 1.0e-10f) ? 1.0f/diag : 1.0f;
        for (int k = 0; k < len; k++) h_csr_val[row_s+k] *= inv_d;
    }
    CHECK(cudaMemcpy(p->csr_val, h_csr_val, nnz*sizeof(float), cudaMemcpyHostToDevice));

    // Single-level schedule (Jacobi: no L/U dependencies after diagonal scaling)
    p->ilu_n_levels = 1;
    if (p->ilu_level_ptr == NULL) {
        CHECK(cudaMalloc(&p->ilu_level_ptr, 2*sizeof(int)));
        CHECK(cudaMalloc(&p->ilu_level_rows, N*sizeof(int)));
    }
    int h_lptr_host[2] = {0, N};
    int *h_lrows_host = (int*)malloc(N*sizeof(int));
    for (int i = 0; i < N; i++) h_lrows_host[i] = i;
    CHECK(cudaMemcpy(p->ilu_level_ptr, h_lptr_host, 2*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->ilu_level_rows, h_lrows_host, N*sizeof(int), cudaMemcpyHostToDevice));
    free(h_lrows_host);
    free(h_rpt); free(h_csr_val); free(h_csr_col);
    return 0; // pre-scaled CSR handles PC, no extra ILU needed
    // Upload factored values
    CHECK(cudaMemcpy(p->csr_val, h_csr_val, nnz*sizeof(float), cudaMemcpyHostToDevice));

    // Step 5: Compute level schedule for parallel triangular solve
    int *level = (int*)calloc(N, sizeof(int));
    int max_level = 0;
    for (int i = 0; i < N; i++) {
        int my_level = 0;
        for (int k = h_rpt[i]; k < h_rpt[i+1]; k++) {
            int j = h_csr_col[k];
            if (j < i && level[j] > my_level) my_level = level[j];
        }
        level[i] = my_level + 1;
        if (level[i] > max_level) max_level = level[i];
    }
    p->ilu_n_levels = max_level;

    int *h_lptr  = (int*)malloc((max_level+1)*sizeof(int));
    int *h_lrows = (int*)malloc(N*sizeof(int));
    int *lcount  = (int*)calloc(max_level, sizeof(int));
    for (int i = 0; i < N; i++) lcount[level[i]-1]++;
    int sum = 0;
    for (int l = 0; l < max_level; l++) {
        h_lptr[l] = sum; sum += lcount[l]; lcount[l] = 0;
    }
    h_lptr[max_level] = N;
    for (int i = 0; i < N; i++) {
        int l = level[i] - 1;
        h_lrows[h_lptr[l] + lcount[l]] = i;
        lcount[l]++;
    }
    free(lcount); free(level);

    if (p->ilu_level_ptr == NULL) {
        CHECK(cudaMalloc(&p->ilu_level_ptr, (max_level+1)*sizeof(int)));
        CHECK(cudaMalloc(&p->ilu_level_rows, N*sizeof(int)));
    }
    CHECK(cudaMemcpy(p->ilu_level_ptr, h_lptr, (max_level+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->ilu_level_rows, h_lrows, N*sizeof(int), cudaMemcpyHostToDevice));

    free(h_lptr); free(h_lrows); free(h_rpt);
    free(h_csr_val); free(h_csr_col);
    printf("  [ILU(0)] %d levels, using for preconditioner\n", max_level);
    return 1; // ILU available
}

// Apply ILU preconditioner: w = U^{-1} * L^{-1} * v
void cusparse_ilu_apply(Particles *p, const float *v, float *w, float *y_tmp, int N) {
    if (p->ilu_n_levels <= 0) return;
    int nblk = (N+BLOCK_SIZE-1)/BLOCK_SIZE;

    // For single-level (Jacobi): just copy v to w (diagonal scaling already done)
    // For multi-level ILU(0): L-solve then U-solve using level schedule
    // Level schedule is on device; use simple loop for single-level case
    if (p->ilu_n_levels == 1) {
        // Diagonal scaling: w = D^{-1} * v (already applied to matrix)
        // L has unit diag, U has diag=1 after scaling. w = U^{-1} * L^{-1} * v = v
        k_copy<<<nblk,BLOCK_SIZE>>>(w, v, N);
        return;
    }

    // Multi-level ILU (for V3)
    int *lrows = p->ilu_level_rows;
    int *lptr  = p->ilu_level_ptr;
    int nlev = p->ilu_n_levels;
    // Copy level pointers from device
    int *h_lptr = (int*)malloc((nlev+1)*sizeof(int));
    CHECK(cudaMemcpy(h_lptr, lptr, (nlev+1)*sizeof(int), cudaMemcpyDeviceToHost));
    for (int lev = 0; lev < nlev; lev++) {
        int start = h_lptr[lev], end = h_lptr[lev+1];
        int nrows = end - start;
        int blks = (nrows + BLOCK_SIZE - 1)/BLOCK_SIZE;
        k_ilu_l_solve_level<<<blks, BLOCK_SIZE>>>(
            p->csr_rpt, p->csr_col, p->csr_val,
            lrows, start, end, v, y_tmp, p->A_diag);
    }
    for (int lev = nlev-1; lev >= 0; lev--) {
        int start = h_lptr[lev], end = h_lptr[lev+1];
        int nrows = end - start;
        int blks = (nrows + BLOCK_SIZE - 1)/BLOCK_SIZE;
        k_ilu_u_solve_level<<<blks,BLOCK_SIZE>>>(
            p->csr_rpt, p->csr_col, p->csr_val,
            lrows, start, end, y_tmp, w);
    }
    free(h_lptr);
}

// cuSPARSE SpMV: y = alpha * A * x + beta * y
void cusparse_spmv(Particles *p, const float *x, float *y, float alpha, float beta) {
    cusparseHandle_t h = p->cusparse_handle;
    cusparseDnVecSetValues(p->dn_x, (void*)x);
    cusparseDnVecSetValues(p->dn_y, (void*)y);
    cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, p->sp_mat, p->dn_x, &beta, p->dn_y,
                 CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, p->spmv_buf);
}

// ============================================================================
// BiCGSTAB solver (host orchestrator)
// ============================================================================
// CUB-based dot product: compute sum(a[i]*b[i]) with on-device reduction
// Uses element-wise multiply kernel + CUB DeviceReduce::Sum
__global__ void k_elem_mul(const float *a, const float *b, float *out, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return;
    out[i]=a[i]*b[i];
}

float dot_reduce(Particles *p, const float *da, const float *db, int N, int nblk) {
    // Element-wise multiply: tmp1 = da .* db
    k_elem_mul<<<nblk,BLOCK_SIZE>>>(da, db, p->tmp1, N);
    // CUB device-side reduction
    cub::DeviceReduce::Sum(p->cub_temp, p->cub_temp_bytes,
                           (const float*)p->tmp1, p->dot_result, N);
    // Copy single float result to host
    float result;
    CHECK(cudaMemcpy(&result, p->dot_result, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

int bicgstab_solve(Particles *p, int N, int maxit, float tol) {
    int nblk=(N+BLOCK_SIZE-1)/BLOCK_SIZE;

    // BiCGSTAB with pre-scaled CSR (symmetric Jacobi built into matrix)
    // A_sym = D^{-1/2}*A*D^{-1/2}, solve A_sym * x_sym = b_sym
    // b_sym = D^{-1/2} * b,  recover x = D^{-1/2} * x_sym

    // Transform RHS: b_sym = D^{-1/2} * b
    k_elem_mul<<<nblk,BLOCK_SIZE>>>(p->b, p->inv_sqrt_diag, p->tmp2, N);
    // Transform initial pressure to symmetric space: p_sym = D^{1/2} * p
    // p_sym[i] = p[i] / inv_sqrt_diag[i] = p[i] * sqrt(|A_ii|)
    // Use k_elem_mul with inv_sqrt_diag-inverse... simpler: solve for dp
    // For correction dp: initial residual r0 = b_sym - A_sym * p_sym_initial
    cusparse_spmv(p, p->p, p->tmp1, 1.0f, 0.0f); // tmp1 = A_sym * p_sym?
    // Wait, p is in ORIGINAL space, not symmetric! Need to transform.
    // p_sym = D^{1/2} * p. Compute A_sym * p_sym via:
    // First compute p_sym (just use inv_sqrt_diag as scale), then SpMV
    // SIMPLEST: start from zero (solve for dp = 0-p), b_sym changes
    // Actually: r0 = D^{-1/2}*b - D^{-1/2}*A*D^{-1/2} * D^{1/2}*p0
    // = D^{-1/2}*(b - A*p0)
    // So: tmp1 = A * p0 (plain SpMV on ORIGINAL matrix not available)
    // We only have A_sym in CSR! Need original A for this.
    // FIX: use zero initial guess (dp = 0), solve for full p_sym
    k_set<<<nblk,BLOCK_SIZE>>>(p->p, 0.0f, N);
    k_copy<<<nblk,BLOCK_SIZE>>>(p->r_bcg, p->tmp2, N); // r0 = b_sym
    k_copy<<<nblk,BLOCK_SIZE>>>(p->r0_bcg, p->r_bcg, N);
    k_set<<<nblk,BLOCK_SIZE>>>(p->v_bcg, 0, N);
    k_set<<<nblk,BLOCK_SIZE>>>(p->p_bcg, 0, N);

    double rho=1.0, alpha_d=1.0, omega=1.0;
    float bnorm = sqrtf(dot_reduce(p,p->tmp2,p->tmp2,N,nblk));
    float tol_a = tol*fmaxf(bnorm,1.f);
    if(bnorm<1e-15f) { k_elem_mul<<<nblk,BLOCK_SIZE>>>(p->p,p->inv_sqrt_diag,p->tmp1,N); k_copy<<<nblk,BLOCK_SIZE>>>(p->p,p->tmp1,N); return 0; }

    int iter;
    float r0_norm = 0;
    for(iter=0;iter<maxit;iter++) {
        float rho1=dot_reduce(p,p->r0_bcg,p->r_bcg,N,nblk);
        if (iter == 0) r0_norm = sqrtf(fmaxf(fabsf(rho1),0.f));
        if(fabsf(rho1)<1e-30f)break;
        double beta=(rho1/rho)*(alpha_d/omega);
        rho=rho1;

        // p = r + beta*(p - omega*v)
        k_waxpby<<<nblk,BLOCK_SIZE>>>(p->tmp1,p->p_bcg,1.f,p->v_bcg,-(float)omega,N);
        k_waxpby<<<nblk,BLOCK_SIZE>>>(p->p_bcg,p->r_bcg,1.f,p->tmp1,(float)beta,N);

        // v = A_sym * p_bcg  (cuSPARSE SpMV, CSR already symmetrically scaled)
        cusparse_spmv(p, p->p_bcg, p->v_bcg, 1.0f, 0.0f);

        float r0v=dot_reduce(p,p->r0_bcg,p->v_bcg,N,nblk);
        if(fabsf(r0v)<1e-30f)break;
        alpha_d=rho/r0v;
        float alpha_f = (float)alpha_d;

        // s = r - alpha*v
        k_waxpby<<<nblk,BLOCK_SIZE>>>(p->s_bcg,p->r_bcg,1.f,p->v_bcg,-alpha_f,N);

        // t = A_sym * s_bcg
        cusparse_spmv(p, p->s_bcg, p->t_bcg, 1.0f, 0.0f);

        float ts=dot_reduce(p,p->t_bcg,p->s_bcg,N,nblk);
        float tt=dot_reduce(p,p->t_bcg,p->t_bcg,N,nblk);
        omega=(fabsf(tt)>1e-30f)?(double)ts/tt:1.0;
        float omega_f = (float)omega;

        // x_sym += alpha*p + omega*s
        k_axpy<<<nblk,BLOCK_SIZE>>>(p->p,p->p_bcg,alpha_f,N);
        k_axpy<<<nblk,BLOCK_SIZE>>>(p->p,p->s_bcg,omega_f,N);

        // r = s - omega*t
        k_waxpby<<<nblk,BLOCK_SIZE>>>(p->r_bcg,p->s_bcg,1.f,p->t_bcg,-omega_f,N);

        float rnorm=sqrtf(dot_reduce(p,p->r_bcg,p->r_bcg,N,nblk));
        if (!isfinite(rnorm)) { iter=maxit; break; }
        if(rnorm<tol_a){iter++;break;}
    }
    // Recover: dp = D^{-1/2} * dp_sym  (p->p holds dp_sym, convert back)
    k_elem_mul<<<nblk,BLOCK_SIZE>>>(p->p, p->inv_sqrt_diag, p->tmp1, N);
    k_copy<<<nblk,BLOCK_SIZE>>>(p->p, p->tmp1, N); // p->p now holds dp in original space

    CHECK(cudaDeviceSynchronize());
    if (iter >= maxit) {
        float r_final = sqrtf(dot_reduce(p,p->r_bcg,p->r_bcg,N,nblk));
        printf("  [BICGSTAB WARN] max iters=%d, |b_sym|=%.3e, |r0|=%.3e, |r_final|=%.3e, tol=%.3e\n",
               iter, bnorm, r0_norm, r_final, tol_a);
    }
    return iter;
}

// ============================================================================
// ILU(0) Preconditioner with Level-Scheduled GPU Triangular Solve
// ============================================================================

// Level-scheduled forward substitution: solve L*y = b (L unit lower triangular)
__global__ void k_ilu_l_solve_level(
    const int *row_ptr, const int *col_idx, const float *lu_val,
    const int *level_rows, int lev_start, int lev_end,
    const float *b, float *y, const float *diag)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = (idx < lev_end - lev_start) ? level_rows[lev_start + idx] : -1;
    if (i < 0) return;

    float sum = b[i];
    for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
        int j = col_idx[k];
        if (j < i) { // lower triangular part
            sum -= lu_val[k] * __ldg(&y[j]);
        }
    }
    // L has unit diagonal, so y[i] = sum / 1.0
    y[i] = sum;
}

// Level-scheduled backward substitution: solve U*x = y (U stored with diagonal)
__global__ void k_ilu_u_solve_level(
    const int *row_ptr, const int *col_idx, const float *lu_val,
    const int *level_rows, int lev_start, int lev_end,
    const float *y, float *x)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // Process levels in reverse order (level_rows sorted by forward level)
    int i = (idx < lev_end - lev_start) ? level_rows[lev_start + idx] : -1;
    if (i < 0) return;

    float sum = y[i];
    float diag = 1.0f;
    for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
        int j = col_idx[k];
        if (j > i) { // upper triangular part
            sum -= lu_val[k] * __ldg(&x[j]);
        } else if (j == i) {
            diag = lu_val[k]; // diagonal stored in LU factors
        }
    }
    x[i] = (fabsf(diag) > 1.0e-10f) ? sum / diag : sum;
}

// Apply ILU preconditioner: w = (LU)^{-1} * v
void ilu_preconditioner_apply(Particles *p, const float *v, float *w,
                               float *y_tmp, int N, int nblk) {
    if (p->ilu_n_levels <= 0) {
        // Fallback: Jacobi
        k_scale_invdiag<<<nblk,BLOCK_SIZE>>>(w, v, p->A_diag, N);
        return;
    }
    int *lev_rows = p->ilu_level_rows;
    int *lev_ptr  = p->ilu_level_ptr;
    int nlev = p->ilu_n_levels;

    // Forward: L*y = v (levels 0 to nlev-1)
    for (int lev = 0; lev < nlev; lev++) {
        int start = lev_ptr[lev];
        int end   = lev_ptr[lev+1];
        int nrows = end - start;
        int blks = (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_ilu_l_solve_level<<<blks, BLOCK_SIZE>>>(
            p->ilu_row_ptr, p->ilu_col_idx, p->ilu_val,
            lev_rows, start, end, v, y_tmp, p->A_diag);
    }
    // Backward: U*w = y (levels nlev-1 to 0)
    for (int lev = nlev-1; lev >= 0; lev--) {
        int start = lev_ptr[lev];
        int end   = lev_ptr[lev+1];
        int nrows = end - start;
        int blks = (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_ilu_u_solve_level<<<blks, BLOCK_SIZE>>>(
            p->ilu_row_ptr, p->ilu_col_idx, p->ilu_val,
            lev_rows, start, end, y_tmp, w);
    }
}

// Build ILU(0) factorization and level schedule on host
int build_ilu0(Particles *p, int N) {
    // Copy nbr_cnt from device
    int *h_nbr_cnt = (int*)malloc(N*sizeof(int));
    CHECK(cudaMemcpy(h_nbr_cnt, p->nbr_cnt, N*sizeof(int), cudaMemcpyDeviceToHost));

    int nnz = 0;
    for (int i = 0; i < N; i++) nnz += h_nbr_cnt[i] + 1;
    p->ilu_nnz = nnz;

    // Allocate host CSR
    int *h_row_ptr = (int*)malloc((N+1)*sizeof(int));
    int *h_col_idx = (int*)malloc(nnz*sizeof(int));
    float *h_val   = (float*)malloc(nnz*sizeof(float));

    // Copy A matrix from device
    float *h_Av = (float*)malloc(N*(MAX_NEIGHBORS+1)*sizeof(float));
    int   *h_Ac = (int*)malloc(N*(MAX_NEIGHBORS+1)*sizeof(int));
    CHECK(cudaMemcpy(h_Av, p->A_val, N*(MAX_NEIGHBORS+1)*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_Ac, p->A_col, N*(MAX_NEIGHBORS+1)*sizeof(int), cudaMemcpyDeviceToHost));

    // Build CSR and compute Jacobi diagonal scaling (simplified ILU)
    int pos = 0;
    for (int i = 0; i < N; i++) {
        h_row_ptr[i] = pos;
        int ncol = h_nbr_cnt[i] + 1;
        int base = i * (MAX_NEIGHBORS + 1);
        float diag = 0;
        int diag_pos = -1;
        for (int k = 0; k < ncol; k++) {
            h_col_idx[pos+k] = h_Ac[base + k];
            h_val[pos+k]     = h_Av[base + k];
            if (h_Ac[base+k] == i) { diag = h_Av[base+k]; diag_pos = pos+k; }
        }
        // Store inverse diagonal for Jacobi PC
        float inv_d = (fabsf(diag) > 1.0e-10f) ? 1.0f/diag : 1.0f;
        // Replace row with scaled row: A'[i][j] = A[i][j] / A[i][i]
        for (int k = 0; k < ncol; k++) h_val[pos+k] *= inv_d;
        pos += ncol;
    }
    h_row_ptr[N] = pos;
    free(h_Av); free(h_Ac);
    free(h_nbr_cnt);

    // Single-level schedule (Jacobi: no dependencies between rows)
    int n_levels = 1;
    int *h_level_ptr  = (int*)malloc(2*sizeof(int));
    int *h_level_rows = (int*)malloc(N*sizeof(int));
    h_level_ptr[0] = 0; h_level_ptr[1] = N;
    for (int i = 0; i < N; i++) h_level_rows[i] = i;

    // Step 4: Upload ILU factors and level schedule to device
    if (p->ilu_row_ptr == NULL) {
        CHECK(cudaMalloc(&p->ilu_row_ptr, (N+1)*sizeof(int)));
        CHECK(cudaMalloc(&p->ilu_col_idx, nnz*sizeof(int)));
        CHECK(cudaMalloc(&p->ilu_val, nnz*sizeof(float)));
        CHECK(cudaMalloc(&p->ilu_level_ptr, (n_levels+1)*sizeof(int)));
        CHECK(cudaMalloc(&p->ilu_level_rows, N*sizeof(int)));
    }
    CHECK(cudaMemcpy(p->ilu_row_ptr, h_row_ptr, (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->ilu_col_idx, h_col_idx, nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->ilu_val, h_val, nnz*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->ilu_level_ptr, h_level_ptr, (n_levels+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->ilu_level_rows, h_level_rows, N*sizeof(int), cudaMemcpyHostToDevice));
    p->ilu_n_levels = n_levels;

    free(h_row_ptr); free(h_col_idx); free(h_val);
    free(h_level_ptr); free(h_level_rows);
    free(h_nbr_cnt);

    return n_levels;
}

// ============================================================================
// Kernel 6: Update (pressure gradient correction)
// ============================================================================
__global__ void kernel_update(
    const float *rx, const float *ry, const float *rz,
    const float *uxs, const float *uys, const float *uzs,
    const float *p, const int *type,
    const int *nbr_cnt, const int *nbr_list,
    const float *C,
    float *uxn, float *uyn, float *uzn,
    float *rxn, float *ryn, float *rzn,
    int N, float re, float l0, float rho, float dt, float np)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return;

    if(type[i]==PTYPE_WALL) {
        uxn[i]=0;uyn[i]=0;uzn[i]=0;
        rxn[i]=rx[i];ryn[i]=ry[i];rzn[i]=rz[i];
        return;
    }

    float xi=rx[i],yi=ry[i],zi=rz[i];
    int nc=nbr_cnt[i];
    int C_off=i*81;

    // Compute pressure gradient: grad_p = sum w*dp*(C1·P) etc / n0
    float gpx=0,gpy=0,gpz=0,sw=0;
    for(int k=0;k<nc;k++) {
        int j=nbr_list[i*MAX_NEIGHBORS+k];
        float dx=xi-rx[j],dy=yi-ry[j],dz=zi-rz[j];
        float r=sqrtf(dx*dx+dy*dy+dz*dz);
        float w=kern_w(r,re);
        if(w<=0)continue;
        float ir=inv_safe(r);
        float dp=(p[i]-p[j])*ir;
        float P[9]; basis(dx,dy,dz,r,l0,np,type[j],P);
        float c1=0,c2=0,c3=0;
        for(int pp=0;pp<9;pp++) {
            c1+=C[C_off+0*9+pp]*P[pp];
            c2+=C[C_off+1*9+pp]*P[pp];
            c3+=C[C_off+2*9+pp]*P[pp];
        }
        gpx+=w*dp*c1; gpy+=w*dp*c2; gpz+=w*dp*c3;
        sw+=w;
    }
    // Normalize by reference n0 (consistent with corrective matrix scaling)
    // np parameter holds n0_ref for basis(), reuse for consistent normalization
    float n0_inv = 1.f/fmaxf(np, 1.f);
    gpx*=n0_inv; gpy*=n0_inv; gpz*=n0_inv;

    // u^{n+1} = u* - dt/rho * grad(p)
    uxn[i]=uxs[i]-(dt/rho)*gpx;
    uyn[i]=uys[i]-(dt/rho)*gpy;
    uzn[i]=uzs[i]-(dt/rho)*gpz;

    // r^{n+1} = r^n + dt * u^{n+1}
    rxn[i]=rx[i]+dt*uxn[i];
    ryn[i]=ry[i]+dt*uyn[i];
    rzn[i]=rz[i]+dt*uzn[i];
}

// ============================================================================
// TimeStep() - single LS-MPS step
// ============================================================================
float TimeStep(Particles *p, const Config *c, int step) {
    int N=p->N, nblk=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    float re=c->re, l0=c->l0;
    float cell_w=re; // cell size = cutoff radius
    float np=c->n0_ref;  // use computed reference number density
    Timer t; float ttl=0,ms;

    // Compute cell grid dimensions
    float Lx=(c->nx-1)*l0, Ly=(c->ny-1)*l0, Lz=(c->nz-1)*l0;
    int ncx=max(1,(int)ceilf(Lx/cell_w)+1);
    int ncy=max(1,(int)ceilf(Ly/cell_w)+1);
    int ncz=max(1,(int)ceilf(Lz/cell_w)+1);

    // ---- Op1: NeighborSearch ----
    tic(&t,0);
    int num_cells = ncx*ncy*ncz;
    CHECK(cudaMemset(p->cell_head, -1, num_cells*sizeof(int)));
    kernel_cell_link<<<nblk,BLOCK_SIZE>>>(
        p->rx,p->ry,p->rz, p->cell_id,p->cell_head,p->cell_next, N,
        cell_w,ncx,ncy);
    kernel_build_neighbors<<<nblk,BLOCK_SIZE>>>(
        p->rx,p->ry,p->rz, p->cell_id,p->cell_head,p->cell_next,
        p->nbr_cnt,p->nbr_list, N,
        re,re*re,cell_w,ncx,ncy,ncz);
    ms=toc(&t);ttl+=ms;

    // ---- Op2: Density ----
    tic(&t,1);
    kernel_compute_density<<<nblk,BLOCK_SIZE>>>(
        p->rx,p->ry,p->rz, p->nbr_cnt,p->nbr_list, p->n, N, re);
    if(step==0) k_copy<<<nblk,BLOCK_SIZE>>>(p->n0, p->n, N);
    ms=toc(&t);ttl+=ms;

    // ---- Op3: Gradient + Explicit ----
    tic(&t,2);
    kernel_cmatrix_and_explicit<<<nblk,BLOCK_SIZE>>>(
        p->rx,p->ry,p->rz, p->ux,p->uy,p->uz, p->p, p->type,
        p->nbr_cnt,p->nbr_list, p->C,
        p->uxs,p->uys,p->uzs, p->rxs,p->rys,p->rzs, p->divu,
        p->n, p->n0,
        N,re,l0,c->dt,c->nu,c->gx,c->gy,c->gz,np);
    ms=toc(&t);ttl+=ms;

    // ---- Op4: PressureMatrix ----
    tic(&t,3);
    kernel_assemble_ppe<<<nblk,BLOCK_SIZE>>>(
        p->rx,p->ry,p->rz, p->type, p->nbr_cnt,p->nbr_list, p->C,
        p->n,p->n0, p->divu,
        p->A_val,p->A_col,p->A_diag,p->b, N,re,l0,c->rho,c->dt,c->alpha,np);
    ms=toc(&t);ttl+=ms;

    // ---- Setup cuSPARSE + ILU ----
    cusparse_setup(p, N);

    // ---- Op5: BiCGSTAB ----
    tic(&t,4);
    // Save current pressure to uxs (safe, not used after Gradient)
    k_copy<<<nblk,BLOCK_SIZE>>>(p->uxs, p->p, N);
    int iters=bicgstab_solve(p,N,BICGSTAB_MAX_ITER,BICGSTAB_TOL);
    // p now holds dp (correction); p = p_old_saved + dp
    k_axpy<<<nblk,BLOCK_SIZE>>>(p->p, p->uxs, 1.0f, N);
    op_bicg_iters += iters;
    ms=toc(&t);ttl+=ms;
    if(c->verbose) printf("  BiCGSTAB iters: %d\n",iters);

    // ---- Op6: Update ----
    tic(&t,5);
    kernel_update<<<nblk,BLOCK_SIZE>>>(
        p->rx,p->ry,p->rz, p->uxs,p->uys,p->uzs,
        p->p, p->type, p->nbr_cnt,p->nbr_list, p->C,
        p->ux,p->uy,p->uz, p->rx,p->ry,p->rz,
        N,re,l0,c->rho,c->dt,np);
    ms=toc(&t);ttl+=ms;

    return ttl;
}

// ============================================================================
// Verification
// ============================================================================
void verify(const Particles *p, const Config *c) {
    int N=p->N;
    float H=(c->nz-1)*c->l0;

    if (!strcmp(c->case_type, "couette")) {
        // Couette flow verification: u(z) = U * z / gap
        float gap = (c->nz-2)*c->l0;  // distance between plates
        float U = c->wall_speed;
        double se=0,sr2=0;
        for(int i=0;i<N;i++) {
            if(p->h_type[i]!=PTYPE_FLUID)continue;
            float z = p->h_rz[i] - c->l0; // fluid starts at z=l0 (above bottom wall at z=0)
            float u_ref = U * z / gap;
            float d = p->h_ux[i] - u_ref;
            se+=d*d; sr2+=u_ref*u_ref;
        }
        float nrmse=(sr2>1e-10)?sqrtf(se/sr2):sqrtf(se);
        printf("\n=== Verification (Couette flow) ===\n");
        printf("  Ref: u(z) = U*z/gap = %.2f * z / %.4f\n", U, gap);
        printf("  NRMSE: %e\n", nrmse);
        printf("  %s (NRMSE %e vs 0.20)\n", nrmse<0.20f?"PASS":"FAIL", nrmse);
        printf("  Sample velocity profile:\n");
        for(int i=0;i<N;i++) {
            if(p->h_type[i]==PTYPE_FLUID && fabsf(p->h_ry[i])<0.001f && fabsf(p->h_rx[i])<0.001f) {
                float z = p->h_rz[i] - c->l0;
                float u_ref = U*z/gap;
                printf("    z=%.4f  ux=%.6f  ref=%.6f  err=%.2e\n",
                       p->h_rz[i], p->h_ux[i], u_ref, p->h_ux[i]-u_ref);
            }
        }
        printf("==================================\n");
    } else if (!strcmp(c->case_type, "dam_break")) {
        // Dam break: no exact reference, report observables
        float min_z=1e9, max_z=-1e9;
        for(int i=0;i<N;i++) {
            if(p->h_type[i]==PTYPE_FLUID) {
                if(p->h_rz[i]<min_z)min_z=p->h_rz[i];
                if(p->h_rz[i]>max_z)max_z=p->h_rz[i];
            }
        }
        printf("\n=== Dam Break Observables ===\n");
        printf("  Fluid z-range: [%.4f, %.4f]\n", min_z, max_z);
        printf("  Fluid column height: %.4f\n", max_z-min_z);
        printf("=============================\n");
    } else {
        // Default: hydrostatic pressure
        float rho_g=c->rho*fabsf(c->gz);
        double se=0,sr2=0;
        for(int i=0;i<N;i++) {
            if(p->h_type[i]==PTYPE_WALL)continue;
            float ref=rho_g*(H-p->h_rz[i]); if(ref<0)ref=0;
            float d=p->h_p[i]-ref;
            se+=d*d; sr2+=ref*ref;
        }
        float nrmse=(sr2>1e-10)?sqrtf(se/sr2):sqrtf(se);
        printf("\n=== Verification (hydrostatic) ===\n");
        printf("  Ref: p = rho*g*(H-z)\n  NRMSE: %e\n",nrmse);
        printf("  %s (NRMSE %e vs 0.10)\n",nrmse<0.10f?"PASS":"FAIL",nrmse);
        printf("  Sample (first 10 fluid particles):\n");
        int cnt=0;
        for(int i=0;i<N&&cnt<10;i++) {
            if(p->h_type[i]==PTYPE_FLUID) {
                float ref=rho_g*(H-p->h_rz[i]); if(ref<0)ref=0;
                printf("    [%d] z=%.4f p=%.4f ref=%.4f err=%.2e\n",
                       i,p->h_rz[i],p->h_p[i],ref,p->h_p[i]-ref);
                cnt++;
            }
        }
        printf("==============================\n");
    }
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char **argv) {
    if(argc<2) {
        fprintf(stderr,"Usage: %s <config.yaml>\n",argv[0]);
        return 1;
    }
    Config c;
    if(parse_config(argv[1],&c)!=0) return 1;
    print_config(&c);

    int dev; cudaDeviceProp prop;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaGetDeviceProperties(&prop,dev));
    printf("GPU: %s (SM %d.%d, %d MPs)\n\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);

    Particles p; memset(&p,0,sizeof(p));
    init_particles(&p,&c);
    if(p.N>MAX_PARTICLES) {
        fprintf(stderr,"Too many particles: %d > %d\n",p.N,MAX_PARTICLES); return 1;
    }
    alloc_dev(&p); h2d(&p);

    double t0=wall_time(); float gpu_ms=0;

    printf("=== Running %d step(s) ===\n",c.num_steps);
    for(int s=0;s<c.num_steps;s++) {
        if(c.verbose) printf("\n-- Step %d/%d --\n",s+1,c.num_steps);
        float sm=TimeStep(&p,&c,s);
        gpu_ms+=sm;
        if(!c.verbose && (s+1)%10==0)
            printf("  Step %d/%d, avg GPU: %.3f ms\n",s+1,c.num_steps,gpu_ms/(s+1));
    }
    double t1=wall_time();

    d2h(&p);

    // Structured profiling output (industrial format)
    print_structured_profile(p.N, c.num_steps);

    // Verify
    if(c.check_ref) verify(&p,&c);

    // Profiling hint
    printf("\n=== Profiling Info ===\n");
    printf("Run: nsys profile --trace=cuda,nvtx ./lsmps config/benchmark.yaml\n");
    printf("Kernel-to-operator mapping (from kernel name prefix):\n");
    printf("  kernel_build_neighbors      -> 1_NeighborSearch\n");
    printf("  kernel_compute_density      -> 2_Density\n");
    printf("  kernel_cmatrix_and_explicit -> 3_Gradient\n");
    printf("  kernel_assemble_ppe         -> 4_PressureMatrix\n");
    printf("  k_spmv/k_dot/k_axpy/k_*     -> 5_BiCGSTAB\n");
    printf("  kernel_update               -> 6_Update\n");
    printf("======================\n");

    free_particles(&p);
    printf("\nDone.\n");
    return 0;
}
