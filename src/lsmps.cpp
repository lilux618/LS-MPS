#include "lsmps.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace lsmps {
namespace {
using Clock = std::chrono::steady_clock;
constexpr int K = 9;

struct Cell { int x{},y{},z{}; bool operator==(const Cell&o) const {return x==o.x&&y==o.y&&z==o.z;} };
struct CellHash { size_t operator()(const Cell& c) const { size_t h=1469598103934665603ull; h=(h^(uint32_t)c.x)*1099511628211ull; h=(h^(uint32_t)c.y)*1099511628211ull; h=(h^(uint32_t)c.z)*1099511628211ull; return h; } };

std::string trim(std::string s) {
    auto notsp=[](unsigned char c){return !std::isspace(c);};
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notsp));
    s.erase(std::find_if(s.rbegin(), s.rend(), notsp).base(), s.end());
    return s;
}

double norm2(const Vec3& v){return v.x*v.x+v.y*v.y+v.z*v.z;}
double norm(const Vec3& v){return std::sqrt(norm2(v));}

double weight(double r,double re){ if(r<=1e-12||r>=re) return 0.0; double q=re/r-1.0; return q*q; }

std::array<double,K> basis(const Vec3& d,double r,double l0){
    const double invr=1.0/r;
    return {d.x*invr,d.y*invr,d.z*invr,
            d.x*d.x/(2.0*l0*r),d.y*d.y/(2.0*l0*r),d.z*d.z/(2.0*l0*r),
            d.x*d.y/(l0*r),d.x*d.z/(l0*r),d.y*d.z/(l0*r)};
}

bool cholesky(std::array<double,K*K>& a,double reg,bool& used_reg){
    used_reg=false;
    for(int attempt=0;attempt<3;++attempt){
        auto m=a;
        double add = attempt==0?0.0:reg*std::pow(10.0,attempt-1);
        if(add>0){used_reg=true; for(int i=0;i<K;++i)m[i*K+i]+=add;}
        bool ok=true;
        for(int i=0;i<K&&ok;++i){
            for(int j=0;j<=i;++j){
                double s=m[i*K+j];
                for(int k=0;k<j;++k)s-=m[i*K+k]*m[j*K+k];
                if(i==j){ if(!(s>1e-14)||!std::isfinite(s)){ok=false;break;} m[i*K+j]=std::sqrt(s); }
                else m[i*K+j]=s/m[j*K+j];
            }
            for(int j=i+1;j<K;++j)m[i*K+j]=0;
        }
        if(ok){a=m;return true;}
    }
    return false;
}

std::array<double,K> chol_solve(const std::array<double,K*K>& l,const std::array<double,K>& b){
    std::array<double,K> y{},x{};
    for(int i=0;i<K;++i){double s=b[i];for(int k=0;k<i;++k)s-=l[i*K+k]*y[k];y[i]=s/l[i*K+i];}
    for(int i=K-1;i>=0;--i){double s=y[i];for(int k=i+1;k<K;++k)s-=l[k*K+i]*x[k];x[i]=s/l[i*K+i];}
    return x;
}

ParticleType classify_particle(const Particle& p,const std::vector<int>& nbr,const std::vector<Particle>& ps,double l0){
    if(p.type==ParticleType::Wall) return ParticleType::Wall;
    if(nbr.size()<10) return ParticleType::Splash;
    int sectors=0; bool seen[14]{};
    for(int j:nbr){ Vec3 d=ps[j].r-p.r; double ax=std::abs(d.x),ay=std::abs(d.y),az=std::abs(d.z); int s;
        if(ax>=ay&&ax>=az)s=d.x>=0?0:1; else if(ay>=ax&&ay>=az)s=d.y>=0?2:3; else s=d.z>=0?4:5;
        seen[s]=true;
        int oct=6+(d.x>=0?1:0)+2*(d.y>=0?1:0)+4*(d.z>=0?1:0); seen[oct]=true;
    }
    for(bool v:seen) sectors+=v;
    if(p.r.z<1.5*l0) return ParticleType::NearWall;
    if(sectors<11) return ParticleType::FreeSurface;
    return ParticleType::Interior;
}

struct NeighborData { std::vector<std::vector<int>> list; size_t pairs{}; size_t truncated{}; };
NeighborData build_neighbors(const std::vector<Particle>& ps,double re,int maxn){
    std::unordered_map<Cell,std::vector<int>,CellHash> grid; grid.reserve(ps.size()*2);
    auto cell=[&](const Vec3&r){return Cell{(int)std::floor(r.x/re),(int)std::floor(r.y/re),(int)std::floor(r.z/re)};};
    for(size_t i=0;i<ps.size();++i)grid[cell(ps[i].r)].push_back((int)i);
    NeighborData out; out.list.resize(ps.size()); const double re2=re*re;
    for(size_t i=0;i<ps.size();++i){ Cell c=cell(ps[i].r); auto& v=out.list[i];
        for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
            auto it=grid.find(Cell{c.x+dx,c.y+dy,c.z+dz}); if(it==grid.end())continue;
            for(int j:it->second){if((size_t)j==i)continue; if(norm2(ps[j].r-ps[i].r)<re2){if((int)v.size()<maxn)v.push_back(j);else out.truncated++;}}
        }
        out.pairs+=v.size();
    }
    return out;
}

WlsStats run_wls(const Config& cfg,const std::vector<Particle>& ps,const NeighborData& nd){
    WlsStats st; const size_t n=ps.size(); if(n==0)return st;
    size_t stride=std::max<size_t>(1,n/std::max(1,cfg.wls_samples));
    double se_g=0,se_l=0; const double re=cfg.re_ratio*cfg.l0;
    for(size_t i=0;i<n&&st.sampled<(size_t)cfg.wls_samples;i+=stride){
        if(ps[i].type==ParticleType::Wall||ps[i].type==ParticleType::Splash)continue;
        std::array<double,K*K> M{}; std::array<double,K> b{};
        const Vec3 ri=ps[i].r;
        auto f=[](const Vec3&r){return 1.0+2*r.x-3*r.y+0.5*r.z+0.7*r.x*r.x-0.2*r.y*r.y+0.4*r.z*r.z+0.3*r.x*r.y-0.1*r.x*r.z+0.25*r.y*r.z;};
        double fi=f(ri),wsum=0;
        for(int j:nd.list[i]){Vec3 d=ps[j].r-ri;double r=norm(d),w=weight(r,re);if(w==0)continue;auto q=basis(d,r,cfg.l0);double df=f(ps[j].r)-fi;wsum+=w;for(int a=0;a<K;++a){b[a]+=w*q[a]*(df/r);for(int c=0;c<K;++c)M[a*K+c]+=w*q[a]*q[c];}}
        if(wsum<=0){st.failed++;continue;} for(double&x:M)x/=wsum; for(double&x:b)x/=wsum;
        bool reg=false; if(!cholesky(M,1e-8,reg)){st.failed++;continue;} auto x=chol_solve(M,b);
        // Basis stores first derivatives directly. Quadratic slots are scaled; use empirical conversion matching basis definition.
        Vec3 grad{x[0],x[1],x[2]};
        Vec3 exact{2+1.4*ri.x+0.3*ri.y-0.1*ri.z,-3-0.4*ri.y+0.3*ri.x+0.25*ri.z,0.5+0.8*ri.z-0.1*ri.x+0.25*ri.y};
        double ge=norm(grad-exact); double lap=(x[3]+x[4]+x[5])/cfg.l0; double exact_lap=1.8; double le=std::abs(lap-exact_lap);
        st.sampled++; if(reg)st.regularized++;else st.ok++; st.max_gradient_error=std::max(st.max_gradient_error,ge);st.max_laplacian_error=std::max(st.max_laplacian_error,le);se_g+=ge*ge;se_l+=le*le;
    }
    if(st.sampled){st.rms_gradient_error=std::sqrt(se_g/st.sampled);st.rms_laplacian_error=std::sqrt(se_l/st.sampled);} return st;
}

void add_lattice(std::vector<Particle>& p,size_t target,int nx,int ny,int nz,double l0,double jitter,std::mt19937& rng,bool wall_bottom){
    std::uniform_real_distribution<double> u(-jitter,jitter);
    for(int k=0;k<nz&&p.size()<target;++k)for(int j=0;j<ny&&p.size()<target;++j)for(int i=0;i<nx&&p.size()<target;++i){
        Particle q; q.r={(i+0.5+u(rng))*l0,(j+0.5+u(rng))*l0,(k+0.5+u(rng))*l0}; if(wall_bottom&&k==0)q.type=ParticleType::Wall; p.push_back(q);
    }
}
}

const char* type_name(ParticleType t){switch(t){case ParticleType::Interior:return"interior";case ParticleType::FreeSurface:return"free_surface";case ParticleType::Splash:return"splash";case ParticleType::NearWall:return"near_wall";case ParticleType::Wall:return"wall";}return"unknown";}

Config load_config(const std::string& path){
    Config c; std::ifstream in(path); if(!in)throw std::runtime_error("cannot open config: "+path); std::string line;
    while(std::getline(in,line)){auto hash=line.find('#');if(hash!=std::string::npos)line.resize(hash);auto eq=line.find('=');if(eq==std::string::npos)continue;auto k=trim(line.substr(0,eq)),v=trim(line.substr(eq+1));
        if(k=="case")c.case_name=v;else if(k=="particles")c.particles=std::stoull(v);else if(k=="seed")c.seed=std::stoul(v);else if(k=="l0")c.l0=std::stod(v);else if(k=="re_ratio")c.re_ratio=std::stod(v);else if(k=="film_layers")c.film_layers=std::stoi(v);else if(k=="max_neighbors")c.max_neighbors=std::stoi(v);else if(k=="wls_samples")c.wls_samples=std::stoi(v);else if(k=="steps")c.steps=std::stoi(v);else if(k=="injected_per_step")c.injected_per_step=std::stoi(v);else if(k=="cnl_rebuild_interval")c.cnl_rebuild_interval=std::stoi(v);
    } return c;
}

std::vector<Particle> generate_case(const Config& cfg){
    auto t0=Clock::now(); (void)t0; std::mt19937 rng(cfg.seed); std::vector<Particle> p; p.reserve(cfg.particles); double n3=std::cbrt((double)cfg.particles); int nx=std::max(4,(int)std::ceil(n3)),ny=nx,nz=nx;
    if(cfg.case_name=="wall_film"){
        nz=std::max(3,cfg.film_layers+1); nx=std::max(8,(int)std::ceil(std::sqrt((double)cfg.particles/nz))); ny=nx; add_lattice(p,cfg.particles,nx,ny,nz,cfg.l0,0.04,rng,true);
    } else if(cfg.case_name=="narrow_gap"){
        nz=std::max(4,cfg.film_layers+2); nx=std::max(8,(int)std::ceil(std::sqrt((double)cfg.particles/nz)));ny=nx;add_lattice(p,cfg.particles,nx,ny,nz,cfg.l0,0.02,rng,true);for(auto&q:p)if(q.r.z>(nz-1.5)*cfg.l0)q.type=ParticleType::Wall;
    } else if(cfg.case_name=="rain_injection"){
        size_t film=(size_t)(0.75*cfg.particles); nz=std::max(3,cfg.film_layers+1);nx=std::max(8,(int)std::ceil(std::sqrt((double)film/nz)));ny=nx;add_lattice(p,film,nx,ny,nz,cfg.l0,0.05,rng,true);
        std::uniform_real_distribution<double> ux(0,nx*cfg.l0),uy(0,ny*cfg.l0),uz(4*cfg.l0,20*cfg.l0);while(p.size()<cfg.particles){Particle q;q.r={ux(rng),uy(rng),uz(rng)};q.u={0,0,-5};q.type=ParticleType::Splash;p.push_back(q);}    
    } else { add_lattice(p,cfg.particles,nx,ny,nz,cfg.l0,0.03,rng,false); }
    return p;
}

Metrics run_benchmark(const Config& cfg,std::vector<Particle>& ps){
    Metrics m; m.particles=ps.size(); auto t1=Clock::now(); auto nd=build_neighbors(ps,cfg.re_ratio*cfg.l0,cfg.max_neighbors); auto t2=Clock::now();
    for(size_t i=0;i<ps.size();++i) {
        ps[i].type=classify_particle(ps[i],nd.list[i],ps,cfg.l0);
    }
    auto t3=Clock::now();
    auto ws=run_wls(cfg,ps,nd); auto t4=Clock::now(); m.neighbor_pairs=nd.pairs;m.truncated_particles=nd.truncated;m.neighbor_ms=std::chrono::duration<double,std::milli>(t2-t1).count();m.classify_ms=std::chrono::duration<double,std::milli>(t3-t2).count();m.wls_ms=std::chrono::duration<double,std::milli>(t4-t3).count();m.wls=ws;
    std::vector<int> counts;counts.reserve(nd.list.size());for(auto&v:nd.list)counts.push_back((int)v.size());if(!counts.empty()){m.neighbor_mean=std::accumulate(counts.begin(),counts.end(),0.0)/counts.size();std::sort(counts.begin(),counts.end());auto q=[&](double p){return counts[std::min(counts.size()-1,(size_t)std::floor(p*(counts.size()-1)))];};m.neighbor_p50=q(.50);m.neighbor_p90=q(.90);m.neighbor_p99=q(.99);}for(auto&p:ps){switch(p.type){case ParticleType::Interior:m.interior++;break;case ParticleType::FreeSurface:m.free_surface++;break;case ParticleType::Splash:m.splash++;break;case ParticleType::NearWall:m.near_wall++;break;case ParticleType::Wall:m.wall++;break;}}
    return m;
}

void write_json(const std::string& path,const Config& c,const Metrics& m){std::ofstream o(path);o<<std::fixed<<std::setprecision(6);o<<"{\n  \"case\": \""<<c.case_name<<"\",\n  \"particles\": "<<m.particles<<",\n  \"particle_types\": {\"interior\": "<<m.interior<<", \"free_surface\": "<<m.free_surface<<", \"splash\": "<<m.splash<<", \"near_wall\": "<<m.near_wall<<", \"wall\": "<<m.wall<<"},\n  \"neighbors\": {\"pairs\": "<<m.neighbor_pairs<<", \"mean\": "<<m.neighbor_mean<<", \"p50\": "<<m.neighbor_p50<<", \"p90\": "<<m.neighbor_p90<<", \"p99\": "<<m.neighbor_p99<<", \"truncated\": "<<m.truncated_particles<<"},\n  \"timing_ms\": {\"neighbor\": "<<m.neighbor_ms<<", \"classify\": "<<m.classify_ms<<", \"wls\": "<<m.wls_ms<<"},\n  \"wls\": {\"sampled\": "<<m.wls.sampled<<", \"ok\": "<<m.wls.ok<<", \"regularized\": "<<m.wls.regularized<<", \"failed\": "<<m.wls.failed<<", \"gradient_rms_error\": "<<m.wls.rms_gradient_error<<", \"laplacian_rms_error\": "<<m.wls.rms_laplacian_error<<"}\n}\n";}
void write_csv(const std::string& path,const Config& c,const Metrics& m){std::ofstream o(path);o<<"case,particles,interior,free_surface,splash,near_wall,wall,neighbor_mean,p50,p90,p99,truncated,neighbor_ms,classify_ms,wls_ms,wls_sampled,wls_regularized,wls_failed,gradient_rms_error,laplacian_rms_error\n";o<<c.case_name<<','<<m.particles<<','<<m.interior<<','<<m.free_surface<<','<<m.splash<<','<<m.near_wall<<','<<m.wall<<','<<m.neighbor_mean<<','<<m.neighbor_p50<<','<<m.neighbor_p90<<','<<m.neighbor_p99<<','<<m.truncated_particles<<','<<m.neighbor_ms<<','<<m.classify_ms<<','<<m.wls_ms<<','<<m.wls.sampled<<','<<m.wls.regularized<<','<<m.wls.failed<<','<<m.wls.rms_gradient_error<<','<<m.wls.rms_laplacian_error<<'\n';}

} // namespace lsmps
