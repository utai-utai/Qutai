#include <cuda_bf16.h>
#include <mma.h>
typedef long long I;
typedef REAL R;
#ifdef A_BF16
typedef __nv_bfloat16 AT;
#else
typedef R AT;
#endif
#ifdef B_BF16
typedef __nv_bfloat16 BT;
#else
typedef R BT;
#endif
#ifdef OUT_BF16
typedef __nv_bfloat16 OT;
#else
typedef R OT;
#endif

template<class T> __device__ R read_value(T x) { return (R)x; }
template<> __device__ R read_value(__nv_bfloat16 x) { return (R)__bfloat162float(x); }
template<class T> __device__ __nv_bfloat16 bf16_value(T x) { return __float2bfloat16_rn((float)x); }
template<> __device__ __nv_bfloat16 bf16_value(__nv_bfloat16 x) { return x; }
__device__ void store_value(OT* p,R value,I accumulate) {
    if(accumulate) value+=read_value(*p);
#ifdef OUT_BF16
    *p=__float2bfloat16_rn((float)value);
#else
    *p=value;
#endif
}

// 32x32 输出分块，每线程计算 2x2；只在共享内存中重复使用输入。
extern "C" __global__ void gemm_tiled(const AT* a,const BT* b,OT* out,I M,I K,I N,I sa,I sb,I ta,I tb,I accumulate) {
    __shared__ R A[32][33], B[32][33];
    int x=threadIdx.x, y=threadIdx.y;
    I row=(I)blockIdx.y*32+y, col=(I)blockIdx.x*32+x, batch=blockIdx.z;
    a+=batch*sa; b+=batch*sb; out+=batch*M*N;
    R c00=0,c01=0,c10=0,c11=0;
    for(I start=0;start<K;start+=32) {
        for(int dy=0;dy<2;dy++) for(int dx=0;dx<2;dx++) {
            I r=(I)blockIdx.y*32+y+dy*16, k=start+x+dx*16;
            A[y+dy*16][x+dx*16]=(r<M&&k<K)?read_value(a[ta?k*M+r:r*K+k]):0;
            k=start+y+dy*16; I c=(I)blockIdx.x*32+x+dx*16;
            B[y+dy*16][x+dx*16]=(k<K&&c<N)?read_value(b[tb?c*K+k:k*N+c]):0;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<32;k++) {
            R a0=A[y][k],a1=A[y+16][k],b0=B[k][x],b1=B[k][x+16];
            c00+=a0*b0; c01+=a0*b1; c10+=a1*b0; c11+=a1*b1;
        }
        __syncthreads();
    }
    if(row<M&&col<N) store_value(out+row*N+col,c00,accumulate);
    if(row<M&&col+16<N) store_value(out+row*N+col+16,c01,accumulate);
    if(row+16<M&&col<N) store_value(out+(row+16)*N+col,c10,accumulate);
    if(row+16<M&&col+16<N) store_value(out+(row+16)*N+col+16,c11,accumulate);
}

// 四个 warp 覆盖 64x64，每个 warp 保留四个 16x16 FP32 累加片段。

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void gemm_bf16(const AT* a,const BT* b,OT* out,I M,I K,I N,I sa,I sb,I ta,I tb,I accumulate) {
    using namespace nvcuda;
    // 输入分块在最后一次屏障后已不再使用，与输出暂存复用同一共享内存。
    __shared__ __align__(32) unsigned char shared[64*64*sizeof(float)];
    auto A=reinterpret_cast<__nv_bfloat16 (*)[40]>(shared);
    auto B=reinterpret_cast<__nv_bfloat16 (*)[72]>(shared+64*40*sizeof(__nv_bfloat16));
    auto C=reinterpret_cast<float (*)[64]>(shared);
    int tid=threadIdx.x, warp=tid/32, wy=warp/2, wx=warp%2;
    I br=(I)blockIdx.y*64, bc=(I)blockIdx.x*64;
    a+=blockIdx.z*sa; b+=blockIdx.z*sb; out+=blockIdx.z*M*N;
    wmma::fragment<wmma::accumulator,16,16,16,float> c[2][2];
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::fill_fragment(c[i][j],0.0f);
    for(I start=0;start<K;start+=32) {
        for(int i=tid;i<64*32;i+=128) {
            int r=ta?i%64:i/32, k=ta?i/64:i%32;
            I gr=br+r,gk=start+k;
            A[r][k]=(gr<M&&gk<K)?bf16_value(a[ta?gk*M+gr:gr*K+gk]):__float2bfloat16_rn(0.0f);
        }
        for(int i=tid;i<32*64;i+=128) {
            int k=tb?i%32:i/64, col=tb?i/32:i%64;
            I gk=start+k,gc=bc+col;
            B[k][col]=(gk<K&&gc<N)?bf16_value(b[tb?gc*K+gk:gk*N+gc]):__float2bfloat16_rn(0.0f);
        }
        __syncthreads();
        for(int k=0;k<32;k+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> aa[2];
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bb[2];
            for(int i=0;i<2;i++) wmma::load_matrix_sync(aa[i],&A[wy*32+i*16][k],40);
            for(int j=0;j<2;j++) wmma::load_matrix_sync(bb[j],&B[k][wx*32+j*16],72);
            for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::mma_sync(c[i][j],aa[i],bb[j],c[i][j]);
        }
        __syncthreads();
    }
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::store_matrix_sync(&C[wy*32+i*16][wx*32+j*16],c[i][j],64,wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<64*64;i+=128) if(br+i/64<M&&bc+i%64<N) store_value(out+(br+i/64)*N+bc+i%64,C[i/64][i%64],accumulate);
}

#endif

// SplitMix64 对完整 seed 和两个不同 counter 分别混合，避免层间共享高斯角度。
__device__ unsigned long long random_mix(unsigned long long x) {
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
extern "C" __global__ void normal_fill(R* out,I n,I seed,R mean,R std) {
    I i=(I)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    unsigned long long key=random_mix((unsigned long long)seed);
    unsigned int x=(unsigned int)(random_mix(key^((unsigned long long)i*2))>>32);
    unsigned int y=(unsigned int)(random_mix(key^((unsigned long long)i*2+1))>>32);
    double u=((double)x+0.5)/4294967296.0;
    double v=((double)y+0.5)/4294967296.0;
    out[i]=mean+std*(R)(sqrt(-2.0*log(u))*cos(6.2831853071795864769*v));
}
