// Optional CUDA workload kernels. This file intentionally focuses on the two
// industrial hotspots identified in the PPT: particle classification and WLS
// moment-matrix assembly. The CPU executable is the correctness/reference path.
#include <cuda_runtime.h>
#include <stdint.h>

struct Float3 { float x,y,z; };

__global__ void classify_surface_splash(const int* row_ptr,const int* col_idx,const Float3* r,uint8_t* type,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int begin=row_ptr[i],end=row_ptr[i+1],nn=end-begin;if(nn<10){type[i]=2;return;}unsigned mask=0;
    for(int e=begin;e<end;++e){int j=col_idx[e];Float3 d{r[j].x-r[i].x,r[j].y-r[i].y,r[j].z-r[i].z};float ax=fabsf(d.x),ay=fabsf(d.y),az=fabsf(d.z);int s=ax>=ay&&ax>=az?(d.x>=0?0:1):(ay>=ax&&ay>=az?(d.y>=0?2:3):(d.z>=0?4:5));mask|=1u<<s;int oct=6+(d.x>=0)+2*(d.y>=0)+4*(d.z>=0);mask|=1u<<oct;}
    int coverage=__popc(mask);type[i]=coverage<11?1:0;
}

__device__ inline float wfun(float dist,float re){if(dist<=1e-12f||dist>=re)return 0;float q=re/dist-1;return q*q;}

__global__ void assemble_wls_upper45(const int* row_ptr,const int* col_idx,const Float3* r,const uint8_t* type,float l0,float re,float* upper45,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||type[i]==2||type[i]==4)return;float M[45]={0};float wsum=0;
    for(int e=row_ptr[i];e<row_ptr[i+1];++e){int j=col_idx[e];float dx=r[j].x-r[i].x,dy=r[j].y-r[i].y,dz=r[j].z-r[i].z;float rr=sqrtf(dx*dx+dy*dy+dz*dz),w=wfun(rr,re);if(w==0)continue;float q[9]={dx/rr,dy/rr,dz/rr,dx*dx/(2*l0*rr),dy*dy/(2*l0*rr),dz*dz/(2*l0*rr),dx*dy/(l0*rr),dx*dz/(l0*rr),dy*dz/(l0*rr)};wsum+=w;int k=0;for(int a=0;a<9;++a)for(int b=a;b<9;++b)M[k++]+=w*q[a]*q[b];}
    float inv=wsum>0?1.0f/wsum:0;for(int k=0;k<45;++k)upper45[i*45+k]=M[k]*inv;
}
