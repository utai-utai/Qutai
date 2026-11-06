#include <cuda_bf16.h>
#include <mma.h>
typedef long long I;
typedef REAL R;

// 32x32 输出分块，每线程计算 2x2；只在共享内存中重复使用输入。
extern "C" __global__ void gemm_tiled(const R* a,const R* b,R* out,I M,I K,I N,I sa,I sb,I ta,I tb) {
    __shared__ R A[32][33], B[32][33];
    int x=threadIdx.x, y=threadIdx.y;
    I row=(I)blockIdx.y*32+y, col=(I)blockIdx.x*32+x, batch=blockIdx.z;
    a+=batch*sa; b+=batch*sb; out+=batch*M*N;
    R c00=0,c01=0,c10=0,c11=0;
    for(I start=0;start<K;start+=32) {
        for(int dy=0;dy<2;dy++) for(int dx=0;dx<2;dx++) {
            I r=(I)blockIdx.y*32+y+dy*16, k=start+x+dx*16;
            A[y+dy*16][x+dx*16]=(r<M&&k<K)?a[ta?k*M+r:r*K+k]:0;
            k=start+y+dy*16; I c=(I)blockIdx.x*32+x+dx*16;
            B[y+dy*16][x+dx*16]=(k<K&&c<N)?b[tb?c*K+k:k*N+c]:0;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<32;k++) {
            R a0=A[y][k],a1=A[y+16][k],b0=B[k][x],b1=B[k][x+16];
            c00+=a0*b0; c01+=a0*b1; c10+=a1*b0; c11+=a1*b1;
        }
        __syncthreads();
    }
    if(row<M&&col<N) out[row*N+col]=c00;
    if(row<M&&col+16<N) out[row*N+col+16]=c01;
    if(row+16<M&&col<N) out[(row+16)*N+col]=c10;
    if(row+16<M&&col+16<N) out[(row+16)*N+col+16]=c11;
}

// 四个 warp 覆盖 64x64，每个 warp 保留四个 16x16 FP32 累加片段。

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void gemm_bf16(const float* a,const float* b,float* out,I M,I K,I N,I sa,I sb,I ta,I tb) {
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 A[64][32], B[32][64];
    __shared__ __align__(32) float C[64][64];
    int tid=threadIdx.x, warp=tid/32, wy=warp/2, wx=warp%2;
    I br=(I)blockIdx.y*64, bc=(I)blockIdx.x*64;
    a+=blockIdx.z*sa; b+=blockIdx.z*sb; out+=blockIdx.z*M*N;
    wmma::fragment<wmma::accumulator,16,16,16,float> c[2][2];
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::fill_fragment(c[i][j],0.0f);
    for(I start=0;start<K;start+=32) {
        for(int i=tid;i<64*32;i+=128) {
            I r=br+i/32,k=start+i%32;
            A[i/32][i%32]=__float2bfloat16_rn((r<M&&k<K)?a[ta?k*M+r:r*K+k]:0.0f);
        }
        for(int i=tid;i<32*64;i+=128) {
            I k=start+i/64,col=bc+i%64;
            B[i/64][i%64]=__float2bfloat16_rn((k<K&&col<N)?b[tb?col*K+k:k*N+col]:0.0f);
        }
        __syncthreads();
        for(int k=0;k<32;k+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> aa[2];
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bb[2];
            for(int i=0;i<2;i++) wmma::load_matrix_sync(aa[i],&A[wy*32+i*16][k],32);
            for(int j=0;j<2;j++) wmma::load_matrix_sync(bb[j],&B[k][wx*32+j*16],64);
            for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::mma_sync(c[i][j],aa[i],bb[j],c[i][j]);
        }
        __syncthreads();
    }
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::store_matrix_sync(&C[wy*32+i*16][wx*32+j*16],c[i][j],64,wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<64*64;i+=128) if(br+i/64<M&&bc+i%64<N) out[(br+i/64)*N+bc+i%64]=C[i/64][i%64];
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
