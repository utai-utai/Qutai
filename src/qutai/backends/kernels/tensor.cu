#include <cuda_bf16.h>
typedef long long I;
typedef REAL R;
#ifdef BFLOAT16
typedef __nv_bfloat16 T;
#else
typedef R T;
#endif
#define IDX I i = (I)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return

extern "C" __global__ void fill(T* out,I n,R re,R im,I parts) { IDX; out[i*parts]=re; if(parts==2) out[i*parts+1]=im; }
extern "C" __global__ void fill_int(I* out,I n,I v) { IDX; out[i]=v; }
__device__ double read_value(const void* p,I i,I code,I part=0) {
    if(code==0) return ((const float*)p)[i];
    if(code==1) return ((const double*)p)[i];
    if(code==2) return ((const float*)p)[2*i+part];
    if(code==3) return ((const double*)p)[2*i+part];
    if(code==5) return __bfloat162float(((const __nv_bfloat16*)p)[i]);
    return ((const I*)p)[i];
}
extern "C" __global__ void cast(const void* in,void* out,I n,I src,I dst) {
    IDX; double a=read_value(in,i,src), b=(src==2||src==3)?read_value(in,i,src,1):0;
    if(dst==0) ((float*)out)[i]=a;
    if(dst==1) ((double*)out)[i]=a;
    if(dst==2) { ((float*)out)[2*i]=a; ((float*)out)[2*i+1]=b; }
    if(dst==3) { ((double*)out)[2*i]=a; ((double*)out)[2*i+1]=b; }
    if(dst==4) ((I*)out)[i]=(I)a;
    if(dst==5) ((__nv_bfloat16*)out)[i]=__float2bfloat16_rn((float)a);
}
extern "C" __global__ void binary(const T* a,const T* b,T* out,I n,I op,unsigned long long* status) { IDX; R x=a[i], y=b[i]; if(op==3&&y==0) atomicExch(status,1ULL); out[i]=op==0?x+y:op==1?x-y:op==2?x*y:x/y; }
extern "C" __global__ void unary(const T* a,const T* g,T* out,I n,I op,R factor,unsigned long long* status) {
    IDX; R x=a[i];
    if(op==0) out[i]=-x;
    if(op==1) out[i]=x*x;
    if(op==2) out[i]=x*factor;
    if(op==3) out[i]=x>0?x:0;
    if(op==4) out[i]=x>0?(R)g[i]:(R)0;
    if(op==5) out[i]=x*((R)1+erf(x/sqrt((R)2)))/(R)2;
    if(op==6) out[i]=(R)g[i]*(((R)1+erf(x/sqrt((R)2)))/(R)2+x*exp(-x*x/(R)2)/sqrt((R)6.2831853071795864769));
    if(op==7) out[i]=exp(x);
    if(op==8) { if(x<=0) atomicExch(status,1ULL); out[i]=log(x); }
}
extern "C" __global__ void gather(const char* a,const I* pos,char* out,I n,I bytes) { IDX; for(I j=0;j<bytes;j++) out[i*bytes+j]=a[pos[i]*bytes+j]; }
extern "C" __global__ void scatter(const R* a,const I* pos,R* out,I n,I parts) { IDX; atomicAdd(out+pos[i/parts]*parts+i%parts,a[i]); }
extern "C" __global__ void broadcast(const char* a,char* out,const I* meta,I n,I rank,I bytes) {
    IDX; I x=i, source=0; for(I d=rank;d-->0;) { I coord=x%meta[d]; x/=meta[d]; source+=coord*meta[rank+d]; } for(I j=0;j<bytes;j++) out[i*bytes+j]=a[source*bytes+j];
}
extern "C" __global__ void reduce_sum(const T* a,T* out,I outer,I width,I inner) {
    I n=outer*inner; IDX; I start=(i/inner)*width*inner+i%inner; R total=0; for(I k=0;k<width;k++) total+=(R)a[start+k*inner]; out[i]=total;
}
extern "C" __global__ void matmul(const T* a,const T* b,T* out,const I* meta,I n,I rows,I inner,I columns,I rank) {
    IDX; I batch=i/(rows*columns), row=(i/columns)%rows, col=i%columns, astart=0,bstart=0;
    for(I d=rank;d-->0;) { I coord=batch%meta[d]; batch/=meta[d]; astart+=coord*meta[rank+d]; bstart+=coord*meta[2*rank+d]; }
    R total=0; for(I k=0;k<inner;k++) total+=(R)a[astart+row*inner+k]*(R)b[bstart+k*columns+col]; out[i]=total;
}
extern "C" __global__ void concatenate(const char* a,char* out,I n,I size,I total,I start,I bytes) {
    IDX; I dest=(i/size)*total+start+i%size; for(I j=0;j<bytes;j++) out[dest*bytes+j]=a[i*bytes+j];
}
extern "C" __global__ void argmax(const T* a,I* out,I n,I width,I inner) {
    IDX; I start=(i/inner)*width*inner+i%inner, best=0; for(I k=1;k<width;k++) if(a[start+k*inner]>a[start+best*inner]) best=k; out[i]=best;
}
extern "C" __global__ void cross_entropy(const T* a,const I* labels,T* out,I n,I columns) {
    IDX; R peak=a[i*columns], total=0; for(I j=1;j<columns;j++) peak=fmax(peak,(R)a[i*columns+j]);
    for(I j=0;j<columns;j++) total+=exp((R)a[i*columns+j]-peak); out[i]=log(total)-((R)a[i*columns+labels[i]]-peak);
}
extern "C" __global__ void cross_entropy_backward(T* p,const T* g,const I* labels,I rows,I columns) {
    I n=rows*columns; IDX; p[i]=((R)p[i]-(i%columns==labels[i/columns]?1:0))*(R)g[0]/rows;
}
__device__ void norm_merge(double& scale,double& sum,double other,double other_sum) {
    if(other==0) return;
    if(scale<other) { double r=scale/other; sum=other_sum+sum*r*r; scale=other; }
    else { double r=other/scale; sum+=other_sum*r*r; }
}
extern "C" __global__ void norm_blocks(const T* a,double* out,I n) {
    __shared__ double scales[256], sums[256];
    int t=threadIdx.x; double scale=0, sum=0;
    for(I j=(I)blockIdx.x*blockDim.x+t;j<n;j+=(I)gridDim.x*blockDim.x) {double x=fabs((double)a[j]); norm_merge(scale,sum,x,1.0);}
    scales[t]=scale; sums[t]=sum; __syncthreads();
    for(int d=128;d;d/=2) {if(t<d) norm_merge(scales[t],sums[t],scales[t+d],sums[t+d]); __syncthreads();}
    if(t==0) out[blockIdx.x]=scales[0]*sqrt(sums[0]);
}
extern "C" __global__ void tensor_norm(const T* a,double* out,I n) {
    __shared__ double scales[256], sums[256];
    int t=threadIdx.x; double scale=0,sum=0;
    for(I j=t;j<n;j+=256) norm_merge(scale,sum,fabs((double)a[j]),1.0);
    scales[t]=scale; sums[t]=sum; __syncthreads();
    for(int d=128;d;d/=2) {if(t<d) norm_merge(scales[t],sums[t],scales[t+d],sums[t+d]); __syncthreads();}
    if(t==0) out[0]=scales[0]*sqrt(sums[0]);
}
extern "C" __global__ void adamw(const T* a,const T* g,const T* m,const T* v,T* out,T* nm,T* nv,I n,R lr,R b1,R b2,R c1,R c2,R eps,R decay) {
    IDX; R x=(R)g[i], mi=b1*(R)m[i]+(1-b1)*x, vi=b2*(R)v[i]+(1-b2)*x*x; nm[i]=mi; nv[i]=vi;
    out[i]=(R)a[i]*(1-lr*decay)-lr*(mi/c1)/(sqrt(vi/c2)+eps);
}
