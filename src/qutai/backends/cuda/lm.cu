// Dense/MoE decoder kernels. Attention stores O(B*T*H*D), never a T*T matrix.
typedef REAL R;
typedef long long I;
#if ACT_BF16
#include <cuda_bf16.h>
#include <mma.h>
typedef __nv_bfloat16 A;
#else
typedef R A;
#endif
template<class T> __device__ R read(const T* values,I index) { return (R)values[index]; }
__device__ R value_probability(R p,I dim,I time) {
#if ACT_BF16
    // TC 前向的 P×V 以 BF16 概率计算；仅同一热点路径的 dV 使用同样精度。
    if(dim==64&&time>=16)return (R)__float2bfloat16_rn(p);
#endif
    return p;
}
#define ITEM I i=(I)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return
__device__ R block_sum(R x,R* tmp) {
    int t=threadIdx.x; tmp[t]=x; __syncthreads();
    for(int s=blockDim.x/2;s;s>>=1) { if(t<s) tmp[t]+=tmp[t+s]; __syncthreads(); }
    R result=tmp[0]; __syncthreads(); return result;
}
__device__ R block_max(R x,R* tmp) {
    int t=threadIdx.x; tmp[t]=x; __syncthreads();
    for(int s=blockDim.x/2;s;s>>=1) { if(t<s) tmp[t]=fmax(tmp[t],tmp[t+s]); __syncthreads(); }
    R result=tmp[0]; __syncthreads(); return result;
}
extern "C" __global__ void lm_add_into(A* out,const A* x,I n) { ITEM; out[i]=read(out,i)+read(x,i); }
extern "C" __global__ void lm_embedding(const R* w,const I* ids,A* out,I n,I width) { ITEM; out[i]=w[ids[i/width]*width+i%width]; }
// IDs are integer metadata grouped in input order; gradients are summed only here.
extern "C" __global__ void lm_embedding_backward(const A* g,const I* tokens,const I* offsets,const I* rows,R* out,I n,I width) {
    ITEM;
    I group=i/width, feature=i%width;
    R total=0;
    for(I p=offsets[group];p<offsets[group+1];p++) total+=read(g,rows[p]*width+feature);
    out[tokens[group]*width+feature]=total;
}
extern "C" __global__ void lm_rms(const A* x,const R* w,R* out,R* inv,I rows,I width,R eps) {
    I row=blockIdx.x, j=threadIdx.x; if(row>=rows) return; __shared__ R tmp[256];
    R total=0; for(I d=j;d<width;d+=blockDim.x) { R v=read(x,row*width+d); total+=v*v; }
    R scale=rsqrt(block_sum(total,tmp)/width+eps);
    if(j==0) inv[row]=scale;
    for(I d=j;d<width;d+=blockDim.x) out[row*width+d]=read(x,row*width+d)*scale*w[d];
}
extern "C" __global__ void lm_rms_dx(const R* g,const A* x,const R* w,const R* inv,A* out,I rows,I width) {
    I row=blockIdx.x,j=threadIdx.x; if(row>=rows) return; __shared__ R tmp[256];
    R total=0; for(I d=j;d<width;d+=blockDim.x) total+=g[row*width+d]*w[d]*read(x,row*width+d);
    R projected=block_sum(total,tmp)/width, scale=inv[row];
    for(I d=j;d<width;d+=blockDim.x) {I at=row*width+d;out[at]=scale*(g[at]*w[d]-read(x,at)*scale*scale*projected);}
}
extern "C" __global__ void lm_rms_dw(const R* g,const A* x,const R* inv,R* out,I n,I rows) { ITEM; R total=0; for(I r=0;r<rows;r++) total+=g[r*n+i]*read(x,r*n+i)*inv[r]; out[i]=total; }
extern "C" __global__ void lm_silu(const A* x,const A* g,A* out,I n,I backward) { ITEM;R v=read(x,i),s=v>=0?(R)1/((R)1+exp(-v)):exp(v)/((R)1+exp(v));out[i]=backward?read(g,i)*s*((R)1+v*((R)1-s)):v*s; }
extern "C" __global__ void lm_rope(const A* x,A* out,I n,I time,I heads,I dim,R theta,I offset,I inverse) {
    ITEM;I at=2*i,d=(i%(dim/2))*2,t=(i/(heads*(dim/2)))%time;
    R a=(offset+t)*pow(theta,-(R)d/dim)*(inverse?-1:1),c=cos(a),s=sin(a);
    out[at]=read(x,at)*c-read(x,at+1)*s;out[at+1]=read(x,at)*s+read(x,at+1)*c;
}
__device__ R warp_sum(R value) {
    for(int offset=16;offset;offset>>=1) value+=__shfl_down_sync(0xffffffff,value,offset);
    return __shfl_sync(0xffffffff,value,0);
}

extern "C" __global__ void lm_attention(const A* q,const A* k,const A* v,A* out,R* lse,I batch,I time,I heads,I kv_heads,I dim) {
    I row=blockIdx.x,lane=threadIdx.x;
    if(row>=batch*time*heads)return;
    I h=row%heads,t=(row/heads)%time,b=row/(heads*time),kh=h/(heads/kv_heads),qi=row*dim;
    R values[8],acc[8];
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;values[j]=d<dim?read(q,qi+d):0;acc[j]=0;}
    R maximum=-(R)__int_as_float(0x7f800000),normalizer=0,scale=rsqrt((R)dim);
    for(I s=0;s<=t;s++){
        I ki=((b*time+s)*kv_heads+kh)*dim;
        R dot=0;
        #pragma unroll
        for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)dot+=values[j]*read(k,ki+d);}
        dot=warp_sum(dot)*scale;
        R next=fmax(maximum,dot),alpha=exp(maximum-next),beta=exp(dot-next);
        normalizer=normalizer*alpha+beta;maximum=next;
        #pragma unroll
        for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)acc[j]=acc[j]*alpha+beta*read(v,ki+d);}
    }
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)out[qi+d]=acc[j]/normalizer;}
    if(lane==0)lse[row]=maximum+log(normalizer);
}

#if ACT_BF16
// D64 热点：16 个 query 共用 64 个 K/V；统计量 FP32，QK/PV 用 BF16 Tensor Core。
extern "C" __global__ void lm_attention_bf16_d64(const A* q,const A* k,const A* v,
        A* out,float* lse,I time,I heads,I kv_heads) {
    using namespace nvcuda;
    __shared__ __align__(32) A queries[16][64],keys[64][64],values[64][64],probabilities[16][64];
    __shared__ __align__(32) float scratch[16][64];
    int tid=threadIdx.x,warp=tid/32,row=tid/8,lane=tid%8;
    I query_base=(I)blockIdx.x*16,head=blockIdx.y,batch=blockIdx.z;
    I kv_head=head/(heads/kv_heads),query=query_base+row;
    for(int i=tid;i<16*64;i+=128) {
        I t=query_base+i/64;
        queries[i/64][i%64]=t<time?q[((batch*time+t)*heads+head)*64+i%64]:A(0.0f);
    }
    float accumulator[8]={0,0,0,0,0,0,0,0};
    float maximum=-__int_as_float(0x7f800000),normalizer=0;
    __syncthreads();
    for(I start=0;start<time&&start<query_base+16;start+=64) {
        for(int i=tid;i<64*64;i+=128) {
            I t=start+i/64,at=((batch*time+t)*kv_heads+kv_head)*64+i%64;
            keys[i/64][i%64]=t<time?k[at]:A(0.0f);
            values[i/64][i%64]=t<time?v[at]:A(0.0f);
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> score;
        wmma::fill_fragment(score,0.0f);
        for(int d=0;d<64;d+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,A,wmma::row_major> qa;
            wmma::fragment<wmma::matrix_b,16,16,16,A,wmma::col_major> kb;
            wmma::load_matrix_sync(qa,&queries[0][d],64);
            wmma::load_matrix_sync(kb,&keys[warp*16][d],64);
            wmma::mma_sync(score,qa,kb,score);
        }
        wmma::store_matrix_sync(&scratch[0][warp*16],score,64,wmma::mem_row_major);
        __syncthreads();
        float peak=-__int_as_float(0x7f800000);
        for(int col=lane;col<64;col+=8)
            if(start+col<=query&&start+col<time&&query<time)peak=fmaxf(peak,scratch[row][col]*0.125f);
        for(int offset=4;offset;offset/=2)peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,offset,8));
        if(query>=time)peak=0;
        float next=fmaxf(maximum,peak),alpha=expf(maximum-next),total=0;
        for(int col=lane;col<64;col+=8) {
            float probability=start+col<=query&&start+col<time&&query<time?expf(scratch[row][col]*0.125f-next):0.0f;
            total+=probability;
            probabilities[row][col]=__float2bfloat16_rn(probability);
        }
        for(int offset=4;offset;offset/=2)total+=__shfl_xor_sync(0xffffffff,total,offset,8);
        normalizer=normalizer*alpha+total;maximum=next;
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> partial;
        wmma::fill_fragment(partial,0.0f);
        for(int d=0;d<64;d+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,A,wmma::row_major> pa;
            wmma::fragment<wmma::matrix_b,16,16,16,A,wmma::row_major> vb;
            wmma::load_matrix_sync(pa,&probabilities[0][d],64);
            wmma::load_matrix_sync(vb,&values[d][warp*16],64);
            wmma::mma_sync(partial,pa,vb,partial);
        }
        wmma::store_matrix_sync(&scratch[0][warp*16],partial,64,wmma::mem_row_major);
        __syncthreads();
        for(int j=0;j<8;j++)accumulator[j]=accumulator[j]*alpha+scratch[row][lane+8*j];
        __syncthreads();
    }
    if(query<time) {
        I base=((batch*time+query)*heads+head)*64;
        for(int j=0;j<8;j++)out[base+lane+8*j]=__float2bfloat16_rn(accumulator[j]/normalizer);
        if(lane==0)lse[(batch*time+query)*heads+head]=maximum+logf(normalizer);
    }
}
// D64 backward: one block owns its output tile, so reductions need no atomics.
extern "C" __global__ void lm_attention_bf16_dq(const A* g,const A* q,const A* k,const A* v,
        const float* lse,const float* delta,A* dq,I time,I heads,I kv_heads) {
    using namespace nvcuda;
    __shared__ __align__(32) A queries[16][64],incoming[16][64],keys[64][64],values[64][64],derivative[16][64];
    __shared__ __align__(32) float score[16][64],product[16][64];
    int tid=threadIdx.x,warp=tid/32,row=tid/8,lane=tid%8;
    I start_query=(I)blockIdx.x*16,head=blockIdx.y,batch=blockIdx.z,kv=head/(heads/kv_heads);
    for(int i=tid;i<16*64;i+=128) {
        I t=start_query+i/64,at=((batch*time+t)*heads+head)*64+i%64;
        queries[i/64][i%64]=t<time?q[at]:A(0.0f);
        incoming[i/64][i%64]=t<time?g[at]:A(0.0f);
    }
    wmma::fragment<wmma::accumulator,16,16,16,float> gradient;
    wmma::fill_fragment(gradient,0.0f);
    __syncthreads();
    for(I start=0;start<time&&start<start_query+16;start+=64) {
        for(int i=tid;i<64*64;i+=128) {
            I t=start+i/64,at=((batch*time+t)*kv_heads+kv)*64+i%64;
            keys[i/64][i%64]=t<time?k[at]:A(0.0f);
            values[i/64][i%64]=t<time?v[at]:A(0.0f);
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> scores,products;
        wmma::fill_fragment(scores,0.0f);wmma::fill_fragment(products,0.0f);
        for(int d=0;d<64;d+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,A,wmma::row_major> qa,ga;
            wmma::fragment<wmma::matrix_b,16,16,16,A,wmma::col_major> kb,vb;
            wmma::load_matrix_sync(qa,&queries[0][d],64);
            wmma::load_matrix_sync(ga,&incoming[0][d],64);
            wmma::load_matrix_sync(kb,&keys[warp*16][d],64);
            wmma::load_matrix_sync(vb,&values[warp*16][d],64);
            wmma::mma_sync(scores,qa,kb,scores);wmma::mma_sync(products,ga,vb,products);
        }
        wmma::store_matrix_sync(&score[0][warp*16],scores,64,wmma::mem_row_major);
        wmma::store_matrix_sync(&product[0][warp*16],products,64,wmma::mem_row_major);
        __syncthreads();
        I t=start_query+row,at=(batch*time+t)*heads+head;
        for(int col=lane;col<64;col+=8) {
            float p=t<time&&start+col<=t?expf(score[row][col]*0.125f-lse[at]):0.0f;
            float ds=t<time?p*(product[row][col]-delta[at])*0.125f:0.0f;
            derivative[row][col]=__float2bfloat16_rn(ds);
        }
        __syncthreads();
        for(int d=0;d<64;d+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,A,wmma::row_major> da;
            wmma::fragment<wmma::matrix_b,16,16,16,A,wmma::row_major> kb;
            wmma::load_matrix_sync(da,&derivative[0][d],64);
            wmma::load_matrix_sync(kb,&keys[d][warp*16],64);
            wmma::mma_sync(gradient,da,kb,gradient);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&score[0][warp*16],gradient,64,wmma::mem_row_major);
    __syncthreads();
    I t=start_query+row,at=((batch*time+t)*heads+head)*64;
    if(t<time)for(int d=lane;d<64;d+=8)dq[at+d]=__float2bfloat16_rn(score[row][d]);
}

extern "C" __global__ void lm_attention_bf16_dkv(const A* g,const A* q,const A* k,const A* v,
        const float* lse,const float* delta,A* dk,A* dv,I time,I heads,I kv_heads) {
    using namespace nvcuda;
    __shared__ __align__(32) A queries[64][64],incoming[64][64],keys[16][64],values[16][64];
    __shared__ __align__(32) A probabilities[64][16],derivative[64][16];
    __shared__ __align__(32) float score[64][16],product[64][16];
    int tid=threadIdx.x,warp=tid/32,row=tid/2,lane=tid%2;
    I start_key=(I)blockIdx.x*16,kv=blockIdx.y,batch=blockIdx.z,repeats=heads/kv_heads;
    for(int i=tid;i<16*64;i+=128) {
        I t=start_key+i/64,at=((batch*time+t)*kv_heads+kv)*64+i%64;
        keys[i/64][i%64]=t<time?k[at]:A(0.0f);
        values[i/64][i%64]=t<time?v[at]:A(0.0f);
    }
    wmma::fragment<wmma::accumulator,16,16,16,float> gradient_k,gradient_v;
    wmma::fill_fragment(gradient_k,0.0f);wmma::fill_fragment(gradient_v,0.0f);
    __syncthreads();
    for(I head=kv*repeats;head<(kv+1)*repeats;head++) {
        for(I start=(start_key/64)*64;start<time;start+=64) {
            for(int i=tid;i<64*64;i+=128) {
                I t=start+i/64,at=((batch*time+t)*heads+head)*64+i%64;
                queries[i/64][i%64]=t<time?q[at]:A(0.0f);
                incoming[i/64][i%64]=t<time?g[at]:A(0.0f);
            }
            __syncthreads();
            wmma::fragment<wmma::accumulator,16,16,16,float> scores,products;
            wmma::fill_fragment(scores,0.0f);wmma::fill_fragment(products,0.0f);
            for(int d=0;d<64;d+=16) {
                wmma::fragment<wmma::matrix_a,16,16,16,A,wmma::row_major> qa,ga;
                wmma::fragment<wmma::matrix_b,16,16,16,A,wmma::col_major> kb,vb;
                wmma::load_matrix_sync(qa,&queries[warp*16][d],64);
                wmma::load_matrix_sync(ga,&incoming[warp*16][d],64);
                wmma::load_matrix_sync(kb,&keys[0][d],64);
                wmma::load_matrix_sync(vb,&values[0][d],64);
                wmma::mma_sync(scores,qa,kb,scores);wmma::mma_sync(products,ga,vb,products);
            }
            wmma::store_matrix_sync(&score[warp*16][0],scores,16,wmma::mem_row_major);
            wmma::store_matrix_sync(&product[warp*16][0],products,16,wmma::mem_row_major);
            __syncthreads();
            I t=start+row,at=(batch*time+t)*heads+head;
            for(int col=lane;col<16;col+=2) {
                float p=t<time&&start_key+col<time&&start_key+col<=t?expf(score[row][col]*0.125f-lse[at]):0.0f;
                float ds=t<time?p*(product[row][col]-delta[at])*0.125f:0.0f;
                probabilities[row][col]=__float2bfloat16_rn(p);
                derivative[row][col]=__float2bfloat16_rn(ds);
            }
            __syncthreads();
            for(int d=0;d<64;d+=16) {
                wmma::fragment<wmma::matrix_a,16,16,16,A,wmma::col_major> da,pa;
                wmma::fragment<wmma::matrix_b,16,16,16,A,wmma::row_major> qb,gb;
                wmma::load_matrix_sync(da,&derivative[d][0],16);
                wmma::load_matrix_sync(pa,&probabilities[d][0],16);
                wmma::load_matrix_sync(qb,&queries[d][warp*16],64);
                wmma::load_matrix_sync(gb,&incoming[d][warp*16],64);
                wmma::mma_sync(gradient_k,da,qb,gradient_k);wmma::mma_sync(gradient_v,pa,gb,gradient_v);
            }
            __syncthreads();
        }
    }
    // Inputs are no longer needed; reuse the query tile for final FP32 stores.
    float* scratch=(float*)queries;
    wmma::store_matrix_sync(scratch+warp*16,gradient_k,64,wmma::mem_row_major);
    __syncthreads();
    I t=start_key+tid/8,feature=tid%8,at=((batch*time+t)*kv_heads+kv)*64;
    if(t<time)for(int d=feature;d<64;d+=8)dk[at+d]=__float2bfloat16_rn(scratch[(tid/8)*64+d]);
    __syncthreads();
    wmma::store_matrix_sync(scratch+warp*16,gradient_v,64,wmma::mem_row_major);
    __syncthreads();
    if(t<time)for(int d=feature;d<64;d+=8)dv[at+d]=__float2bfloat16_rn(scratch[(tid/8)*64+d]);
}

#endif

extern "C" __global__ void lm_attention_delta(const A* g,const A* out,R* delta,I rows,I dim) {
    I row=blockIdx.x,lane=threadIdx.x;
    if(row>=rows)return;
    R value=0;
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)value+=read(g,row*dim+d)*read(out,row*dim+d);}
    value=warp_sum(value);
    if(lane==0)delta[row]=value;
}

extern "C" __global__ void lm_attention_backward_q(const A* g,const A* q,const A* k,const A* v,const R* lse,const R* delta,A* dq,I batch,I time,I heads,I kv_heads,I dim) {
    I row=blockIdx.x,lane=threadIdx.x;
    if(row>=batch*time*heads)return;
    I h=row%heads,t=(row/heads)%time,b=row/(heads*time),kh=h/(heads/kv_heads),qi=row*dim;
    R a[8],go[8],acc[8];
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;a[j]=d<dim?read(q,qi+d):0;go[j]=d<dim?read(g,qi+d):0;acc[j]=0;}
    R scale=rsqrt((R)dim);
    for(I s=0;s<=t;s++){
        I ki=((b*time+s)*kv_heads+kh)*dim;
        R dot=0,dp=0;
        #pragma unroll
        for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim){dot+=a[j]*read(k,ki+d);dp+=go[j]*read(v,ki+d);}}
        dot=warp_sum(dot)*scale;dp=warp_sum(dp);
        R p=exp(dot-lse[row]),ds=p*(dp-delta[row])*scale;
        #pragma unroll
        for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)acc[j]+=ds*read(k,ki+d);}
    }
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)dq[qi+d]=acc[j];}
}

extern "C" __global__ void lm_attention_backward_kv(const A* g,const A* q,const A* k,const A* v,const R* lse,const R* delta,A* dk,A* dv,I batch,I time,I heads,I kv_heads,I dim) {
    I row=blockIdx.x,lane=threadIdx.x;
    if(row>=batch*time*kv_heads)return;
    I kh=row%kv_heads,s=(row/kv_heads)%time,b=row/(kv_heads*time),ki=row*dim,repeats=heads/kv_heads;
    R kval[8],vval[8],acc_k[8],acc_v[8];
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;kval[j]=d<dim?read(k,ki+d):0;vval[j]=d<dim?read(v,ki+d):0;acc_k[j]=0;acc_v[j]=0;}
    R scale=rsqrt((R)dim);
    for(I t=s;t<time;t++){
        for(I h=kh*repeats;h<(kh+1)*repeats;h++){
            I qr=(b*time+t)*heads+h,qi=qr*dim;
            R dot=0,dp=0;
            #pragma unroll
            for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim){dot+=read(q,qi+d)*kval[j];dp+=read(g,qi+d)*vval[j];}}
            dot=warp_sum(dot)*scale;dp=warp_sum(dp);
            R p=exp(dot-lse[qr]),ds=p*(dp-delta[qr])*scale;
            #pragma unroll
            for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim){acc_k[j]+=ds*read(q,qi+d);acc_v[j]+=value_probability(p,dim,time)*read(g,qi+d);}}
        }
    }
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim){dk[ki+d]=acc_k[j];dv[ki+d]=acc_v[j];}}
}

extern "C" __global__ void lm_route(const R* probs,R* weights,I* indices,I n,I experts,I top_k){
    ITEM;R total=0;
    for(I slot=0;slot<top_k;slot++){I best=-1;R peak=-(R)__int_as_float(0x7f800000);
        for(I e=0;e<experts;e++){bool used=false;for(I s=0;s<slot;s++)used|=indices[i*top_k+s]==e;
            if(!used&&(best<0||probs[i*experts+e]>peak)){best=e;peak=probs[i*experts+e];}}
        indices[i*top_k+slot]=best;weights[i*top_k+slot]=peak;total+=peak;}
    for(I slot=0;slot<top_k;slot++)weights[i*top_k+slot]/=total;
}
// Stable GPU compaction: each expert packs increasing (token,slot) positions.
extern "C" __global__ void lm_route_pack(const I* indices,I* counts,I* ids,I* slots,I* locations,I n,I capacity,I top_k,I experts){
    I expert=blockIdx.x,t=threadIdx.x;
    if(expert>=experts)return;
    __shared__ I prefix[256];
    __shared__ I written;
    if(t==0)written=0;
    __syncthreads();
    for(I start=0;start<n;start+=blockDim.x){
        I at=start+t;
        bool selected=at<n&&indices[at]==expert;
        prefix[t]=selected?1:0;
        __syncthreads();
        for(I offset=1;offset<blockDim.x;offset*=2){
            I previous=t>=offset?prefix[t-offset]:0;
            __syncthreads();
            prefix[t]+=previous;
            __syncthreads();
        }
        if(selected){
            I destination=expert*capacity+written+prefix[t]-1;
            ids[destination]=at/top_k;
            slots[destination]=at%top_k;
            locations[at]=written+prefix[t]-1;
        }
        __syncthreads();
        if(t==0)written+=prefix[blockDim.x-1];
        __syncthreads();
    }
    if(t==0)counts[expert]=written;
}
extern "C" __global__ void lm_route_util(const I* counts,R* utilization,I n,I total){ITEM;utilization[i]=(R)counts[i]/total;}
extern "C" __global__ void lm_route_aux(const R* probs,const R* utilization,R* out,I rows,I experts){
    if(blockIdx.x||threadIdx.x)return;R total=0;for(I r=0;r<rows;r++)for(I e=0;e<experts;e++)total+=probs[r*experts+e]*utilization[e];out[0]=experts*total/rows;
}
extern "C" __global__ void lm_route_backward(const R* g,const R* p,const R* w,const I* ids,const R* util,R* out,I n,I rows,I experts,I top_k,I auxiliary){
    ITEM;I r=i/experts,e=i%experts;R value=0,center=0;
    if(auxiliary){for(I j=0;j<experts;j++)center+=p[r*experts+j]*util[j];value=g[0]*experts/rows*p[i]*(util[e]-center);}
    else{for(I s=0;s<top_k;s++)center+=g[r*top_k+s]*w[r*top_k+s];for(I s=0;s<top_k;s++)if(ids[r*top_k+s]==e)value=w[r*top_k+s]*(g[r*top_k+s]-center);}
    out[i]=value;
}
extern "C" __global__ void lm_route_gather(const A* x,const I* ids,A* out,I n,I width,I base,I backward){
    ITEM;I at=ids[base+i/width]*width+i%width;if(backward)out[at]=read(x,i);else out[i]=read(x,at);
}
extern "C" __global__ void lm_combine(const A* y,const R* w,const I* ids,const I* slots,A* out,I n,I width,I base,I top_k){
    ITEM;I route=i/width,row=ids[base+route],slot=slots[base+route];out[row*width+i%width]=read(out,row*width+i%width)+read(y,i)*w[row*top_k+slot];
}
extern "C" __global__ void lm_pointer_table(I* out,I start,I count,I p0,I p1,I p2,I p3,I p4,I p5,I p6,I p7) {
    I index=threadIdx.x;
    if(index>=count)return;
    I values[8]={p0,p1,p2,p3,p4,p5,p6,p7};
    out[start+index]=values[index];
}
// 每个输出只写一次；所有 slot 乘加在 REAL 中完成，不逐专家舍入 BF16。
extern "C" __global__ void lm_combine_all(const unsigned long long* pointers,const R* weights,
        const I* experts,const I* locations,A* out,I n,I width,I top_k) {
    ITEM; I row=i/width,feature=i%width;R total=0;
    for(I slot=0;slot<top_k;slot++) {
        I at=row*top_k+slot;
        const A* values=(const A*)pointers[experts[at]];
        total+=read(values,locations[at]*width+feature)*weights[at];
    }
    out[i]=total;
}
extern "C" __global__ void lm_combine_backward(const A* g,const A* y,const R* w,const I* ids,const I* slots,A* dy,R* dw,I n,I width,I base,I top_k){
    ITEM;I route=i/width,row=ids[base+route],slot=slots[base+route],at=row*top_k+slot;R incoming=read(g,row*width+i%width);dy[i]=incoming*w[at];
}
extern "C" __global__ void lm_combine_weight_backward(const A* g,const A* y,const I* ids,const I* slots,R* dw,I rows,I width,I base,I top_k){
    I route=blockIdx.x,j=threadIdx.x;
    if(route>=rows)return;
    I row=ids[base+route],slot=slots[base+route];
    __shared__ R tmp[256];
    R total=0;
    for(I d=j;d<width;d+=blockDim.x)total+=read(g,row*width+d)*read(y,route*width+d);
    total=block_sum(total,tmp);
    if(j==0)dw[row*top_k+slot]=total;
}
extern "C" __global__ void lm_head_rows(const A* x,A* out,I n,I start,I width,I time,I backward){
    ITEM;I row=start+i/width,at=((row/(time-1))*time+row%(time-1))*width+i%width;if(backward)out[at]=read(out,at)+read(x,i);else out[i]=read(x,at);
}
extern "C" __global__ void lm_head_loss(const A* logits,const I* labels,R* out,I rows,I vocab){
    I row=blockIdx.x,j=threadIdx.x;if(row>=rows)return;__shared__ R tmp[256];
    R peak=-(R)__int_as_float(0x7f800000);for(I c=j;c<vocab;c+=blockDim.x)peak=fmax(peak,read(logits,row*vocab+c));peak=block_max(peak,tmp);
    R total=0;for(I c=j;c<vocab;c+=blockDim.x)total+=exp(read(logits,row*vocab+c)-peak);total=block_sum(total,tmp);
    if(j==0)out[row]=labels[row]==-100?0:log(total)-(read(logits,row*vocab+labels[row])-peak);
}
extern "C" __global__ void lm_head_gradient(A* logits,const I* labels,const R* gradient,I rows,I vocab,R factor){
    I row=blockIdx.x,j=threadIdx.x;if(row>=rows)return;__shared__ R tmp[256];
    R peak=-(R)__int_as_float(0x7f800000);for(I c=j;c<vocab;c+=blockDim.x)peak=fmax(peak,read(logits,row*vocab+c));peak=block_max(peak,tmp);
    R total=0;for(I c=j;c<vocab;c+=blockDim.x)total+=exp(read(logits,row*vocab+c)-peak);total=block_sum(total,tmp);
    for(I c=j;c<vocab;c+=blockDim.x)logits[row*vocab+c]=labels[row]==-100?0:gradient[0]*factor*(exp(read(logits,row*vocab+c)-peak)/total-(c==labels[row]));
}
