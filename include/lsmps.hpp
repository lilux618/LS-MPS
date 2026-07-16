#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace lsmps {

enum class ParticleType : uint8_t { Interior=0, FreeSurface=1, Splash=2, NearWall=3, Wall=4 };
struct Vec3 { double x{},y{},z{}; Vec3 operator+(const Vec3&o)const{return{x+o.x,y+o.y,z+o.z};} Vec3 operator-(const Vec3&o)const{return{x-o.x,y-o.y,z-o.z};} Vec3 operator*(double s)const{return{x*s,y*s,z*s};} };
struct Particle { Vec3 r,u; ParticleType type{ParticleType::Interior}; double pressure{}; };

struct Config {
 std::string case_name{"wall_film"}; size_t particles{20000}; uint32_t seed{7};
 double l0{0.01}, re_ratio{2.1}; int film_layers{4}, max_neighbors{96}, wls_samples{4096};
 int steps{5}, injected_per_step{0}, cnl_rebuild_interval{1};
 int ppe_max_unknowns{12000}, ppe_max_iter{400}; double ppe_tol{1e-9};
 double wls_regularization{1e-8}, boundary_constraint_weight{8.0}, virtual_weight{0.25};
 bool write_csv{true};
};
struct WlsStats { size_t sampled{},ok{},regularized{},failed{},virtual_points{}; double max_gradient_error{},rms_gradient_error{},max_laplacian_error{},rms_laplacian_error{}; };
struct PpeStats { size_t unknowns{},nnz{},dirichlet{},neumann{},failed_rows{}; int iterations{}; double initial_residual{},final_residual{},relative_l2_error{},max_error{},assembly_ms{},solve_ms{}; bool converged{}; };
struct Metrics {
 size_t particles{},interior{},free_surface{},splash{},near_wall{},wall{},neighbor_pairs{},truncated_particles{};
 double neighbor_mean{}; int neighbor_p50{},neighbor_p90{},neighbor_p99{}; double classify_ms{},neighbor_ms{},wls_ms{};
 WlsStats wls; PpeStats ppe;
};
Config load_config(const std::string& path);
std::vector<Particle> generate_case(const Config& cfg);
Metrics run_benchmark(const Config& cfg,std::vector<Particle>& particles);
void write_json(const std::string& path,const Config& cfg,const Metrics& m);
void write_csv(const std::string& path,const Config& cfg,const Metrics& m);
const char* type_name(ParticleType t);
}
