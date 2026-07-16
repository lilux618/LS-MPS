#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lsmps {

enum class ParticleType : uint8_t {
    Interior = 0,
    FreeSurface = 1,
    Splash = 2,
    NearWall = 3,
    Wall = 4
};

struct Vec3 {
    double x{}, y{}, z{};
    Vec3 operator+(const Vec3& o) const { return {x+o.x,y+o.y,z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x,y-o.y,z-o.z}; }
    Vec3 operator*(double s) const { return {x*s,y*s,z*s}; }
};

struct Particle {
    Vec3 r;
    Vec3 u;
    ParticleType type{ParticleType::Interior};
};

struct Config {
    std::string case_name{"wall_film"};
    size_t particles{20000};
    uint32_t seed{7};
    double l0{0.01};
    double re_ratio{2.1};
    double wall_fraction{0.20};
    double surface_fraction{0.20};
    double splash_fraction{0.05};
    int film_layers{4};
    int max_neighbors{96};
    int wls_samples{4096};
    int steps{5};
    int injected_per_step{0};
    int cnl_rebuild_interval{1};
    bool write_csv{true};
};

struct WlsStats {
    size_t sampled{};
    size_t ok{};
    size_t regularized{};
    size_t failed{};
    double max_gradient_error{};
    double rms_gradient_error{};
    double max_laplacian_error{};
    double rms_laplacian_error{};
};

struct Metrics {
    size_t particles{};
    size_t interior{};
    size_t free_surface{};
    size_t splash{};
    size_t near_wall{};
    size_t wall{};
    size_t neighbor_pairs{};
    size_t truncated_particles{};
    double neighbor_mean{};
    int neighbor_p50{};
    int neighbor_p90{};
    int neighbor_p99{};
    double build_ms{};
    double classify_ms{};
    double neighbor_ms{};
    double wls_ms{};
    WlsStats wls;
};

Config load_config(const std::string& path);
std::vector<Particle> generate_case(const Config& cfg);
Metrics run_benchmark(const Config& cfg, std::vector<Particle>& particles);
void write_json(const std::string& path, const Config& cfg, const Metrics& m);
void write_csv(const std::string& path, const Config& cfg, const Metrics& m);
const char* type_name(ParticleType t);

} // namespace lsmps
