#include "lsmps.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>

int main(int argc,char**argv){
    try{
        std::string cfg_path=argc>1?argv[1]:"config/wall_film.cfg";
        std::string outdir=argc>2?argv[2]:"results";
        auto cfg=lsmps::load_config(cfg_path);
        auto particles=lsmps::generate_case(cfg);
        auto m=lsmps::run_benchmark(cfg,particles);
        std::filesystem::create_directories(outdir);
        auto base=outdir+"/"+cfg.case_name;
        lsmps::write_json(base+".json",cfg,m);lsmps::write_csv(base+".csv",cfg,m);
        std::cout<<"LS-MPS industrial workload benchmark\n"
                 <<"case="<<cfg.case_name<<" particles="<<m.particles<<"\n"
                 <<"types interior="<<m.interior<<" surface="<<m.free_surface<<" splash="<<m.splash<<" near_wall="<<m.near_wall<<" wall="<<m.wall<<"\n"
                 <<std::fixed<<std::setprecision(2)
                 <<"neighbors mean="<<m.neighbor_mean<<" p50="<<m.neighbor_p50<<" p90="<<m.neighbor_p90<<" p99="<<m.neighbor_p99<<" truncated="<<m.truncated_particles<<"\n"
                 <<"timing(ms): neighbor="<<m.neighbor_ms<<" classify="<<m.classify_ms<<" wls="<<m.wls_ms<<"\n"
                 <<std::scientific<<"WLS sampled="<<m.wls.sampled<<" regularized="<<m.wls.regularized<<" failed="<<m.wls.failed<<" grad_rms="<<m.wls.rms_gradient_error<<" lap_rms="<<m.wls.rms_laplacian_error<<"\n"
                 <<"output="<<base<<".{json,csv}\n";
        return m.wls.failed>m.wls.sampled/5?2:0;
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
