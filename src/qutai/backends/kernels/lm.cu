// Dense/MoE decoder kernels. Attention stores O(B*T*H*D), never a T*T matrix.
typedef REAL R;
typedef long long I;
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
extern "C" __global__ void lm_add_into(R* out,const R* x,I n) { ITEM; out[i]+=x[i]; }
extern "C" __global__ void lm_embedding(const R* w,const I* ids,R* out,I n,I width) { ITEM; out[i]=w[ids[i/width]*width+i%width]; }
// IDs are integer metadata grouped in input order; gradients are summed only here.
extern "C" __global__ void lm_embedding_backward(const R* g,const I* tokens,const I* offsets,const I* rows,R* out,I n,I width) {
    ITEM;
    I group=i/width, feature=i%width;
    R total=0;
    for(I p=offsets[group];p<offsets[group+1];p++) total+=g[rows[p]*width+feature];
    out[tokens[group]*width+feature]=total;
}
extern "C" __global__ void lm_rms(const R* x,const R* w,R* out,R* inv,I rows,I width,R eps) {
    I row=blockIdx.x, j=threadIdx.x; if(row>=rows) return; __shared__ R tmp[256];
    R total=0; for(I d=j;d<width;d+=blockDim.x) { R v=x[row*width+d]; total+=v*v; }
    R scale=rsqrt(block_sum(total,tmp)/width+eps);
    if(j==0) inv[row]=scale;
    for(I d=j;d<width;d+=blockDim.x) out[row*width+d]=x[row*width+d]*scale*w[d];
}
extern "C" __global__ void lm_rms_dx(const R* g,const R* x,const R* w,const R* inv,R* out,I rows,I width) {
    I row=blockIdx.x,j=threadIdx.x; if(row>=rows) return; __shared__ R tmp[256];
    R total=0; for(I d=j;d<width;d+=blockDim.x) total+=g[row*width+d]*w[d]*x[row*width+d];
    R projected=block_sum(total,tmp)/width, scale=inv[row];
    for(I d=j;d<width;d+=blockDim.x) {I at=row*width+d;out[at]=scale*(g[at]*w[d]-x[at]*scale*scale*projected);}
}
extern "C" __global__ void lm_rms_dw(const R* g,const R* x,const R* inv,R* out,I n,I rows) { ITEM; R total=0; for(I r=0;r<rows;r++) total+=g[r*n+i]*x[r*n+i]*inv[r]; out[i]=total; }
extern "C" __global__ void lm_silu(const R* x,const R* g,R* out,I n,I backward) { ITEM;R v=x[i],s=v>=0?(R)1/((R)1+exp(-v)):exp(v)/((R)1+exp(v));out[i]=backward?g[i]*s*((R)1+v*((R)1-s)):v*s; }
extern "C" __global__ void lm_rope(const R* x,R* out,I n,I time,I heads,I dim,R theta,I offset,I inverse) {
    ITEM;I at=2*i,d=(i%(dim/2))*2,t=(i/(heads*(dim/2)))%time;
    R a=(offset+t)*pow(theta,-(R)d/dim)*(inverse?-1:1),c=cos(a),s=sin(a);
    out[at]=x[at]*c-x[at+1]*s;out[at+1]=x[at]*s+x[at+1]*c;
}
__device__ R warp_sum(R value) {
    for(int offset=16;offset;offset>>=1) value+=__shfl_down_sync(0xffffffff,value,offset);
    return __shfl_sync(0xffffffff,value,0);
}

extern "C" __global__ void lm_attention(const R* q,const R* k,const R* v,R* out,R* lse,I batch,I time,I heads,I kv_heads,I dim) {
    I row=blockIdx.x,lane=threadIdx.x;
    if(row>=batch*time*heads)return;
    I h=row%heads,t=(row/heads)%time,b=row/(heads*time),kh=h/(heads/kv_heads),qi=row*dim;
    R values[8],acc[8];
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;values[j]=d<dim?q[qi+d]:0;acc[j]=0;}
    R maximum=-(R)__int_as_float(0x7f800000),normalizer=0,scale=rsqrt((R)dim);
    for(I s=0;s<=t;s++){
        I ki=((b*time+s)*kv_heads+kh)*dim;
        R dot=0;
        #pragma unroll
        for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)dot+=values[j]*k[ki+d];}
        dot=warp_sum(dot)*scale;
        R next=fmax(maximum,dot),alpha=exp(maximum-next),beta=exp(dot-next);
        normalizer=normalizer*alpha+beta;maximum=next;
        #pragma unroll
        for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)acc[j]=acc[j]*alpha+beta*v[ki+d];}
    }
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)out[qi+d]=acc[j]/normalizer;}
    if(lane==0)lse[row]=maximum+log(normalizer);
}

extern "C" __global__ void lm_attention_delta(const R* g,const R* out,R* delta,I rows,I dim) {
    I row=blockIdx.x,lane=threadIdx.x;
    if(row>=rows)return;
    R value=0;
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)value+=g[row*dim+d]*out[row*dim+d];}
    value=warp_sum(value);
    if(lane==0)delta[row]=value;
}

extern "C" __global__ void lm_attention_backward_q(const R* g,const R* q,const R* k,const R* v,const R* lse,const R* delta,R* dq,I batch,I time,I heads,I kv_heads,I dim) {
    I row=blockIdx.x,lane=threadIdx.x;
    if(row>=batch*time*heads)return;
    I h=row%heads,t=(row/heads)%time,b=row/(heads*time),kh=h/(heads/kv_heads),qi=row*dim;
    R a[8],go[8],acc[8];
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;a[j]=d<dim?q[qi+d]:0;go[j]=d<dim?g[qi+d]:0;acc[j]=0;}
    R scale=rsqrt((R)dim);
    for(I s=0;s<=t;s++){
        I ki=((b*time+s)*kv_heads+kh)*dim;
        R dot=0,dp=0;
        #pragma unroll
        for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim){dot+=a[j]*k[ki+d];dp+=go[j]*v[ki+d];}}
        dot=warp_sum(dot)*scale;dp=warp_sum(dp);
        R p=exp(dot-lse[row]),ds=p*(dp-delta[row])*scale;
        #pragma unroll
        for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)acc[j]+=ds*k[ki+d];}
    }
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim)dq[qi+d]=acc[j];}
}

extern "C" __global__ void lm_attention_backward_kv(const R* g,const R* q,const R* k,const R* v,const R* lse,const R* delta,R* dk,R* dv,I batch,I time,I heads,I kv_heads,I dim) {
    I row=blockIdx.x,lane=threadIdx.x;
    if(row>=batch*time*kv_heads)return;
    I kh=row%kv_heads,s=(row/kv_heads)%time,b=row/(kv_heads*time),ki=row*dim,repeats=heads/kv_heads;
    R kval[8],vval[8],acc_k[8],acc_v[8];
    #pragma unroll
    for(int j=0;j<8;j++){I d=lane+32*j;kval[j]=d<dim?k[ki+d]:0;vval[j]=d<dim?v[ki+d]:0;acc_k[j]=0;acc_v[j]=0;}
    R scale=rsqrt((R)dim);
    for(I t=s;t<time;t++){
        for(I h=kh*repeats;h<(kh+1)*repeats;h++){
            I qr=(b*time+t)*heads+h,qi=qr*dim;
            R dot=0,dp=0;
            #pragma unroll
            for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim){dot+=q[qi+d]*kval[j];dp+=g[qi+d]*vval[j];}}
            dot=warp_sum(dot)*scale;dp=warp_sum(dp);
            R p=exp(dot-lse[qr]),ds=p*(dp-delta[qr])*scale;
            #pragma unroll
            for(int j=0;j<8;j++){I d=lane+32*j;if(d<dim){acc_k[j]+=ds*q[qi+d];acc_v[j]+=p*g[qi+d];}}
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
extern "C" __global__ void lm_route_pack(const I* indices,I* counts,I* ids,I* slots,I n,I capacity,I top_k,I experts){
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
extern "C" __global__ void lm_route_gather(const R* x,const I* ids,R* out,I n,I width,I base,I backward){
    ITEM;I at=ids[base+i/width]*width+i%width;if(backward)out[at]=x[i];else out[i]=x[at];
}
extern "C" __global__ void lm_combine(const R* y,const R* w,const I* ids,const I* slots,R* out,I n,I width,I base,I top_k){
    ITEM;I route=i/width,row=ids[base+route],slot=slots[base+route];out[row*width+i%width]+=y[i]*w[row*top_k+slot];
}
extern "C" __global__ void lm_combine_backward(const R* g,const R* y,const R* w,const I* ids,const I* slots,R* dy,R* dw,I n,I width,I base,I top_k){
    ITEM;I route=i/width,row=ids[base+route],slot=slots[base+route],at=row*top_k+slot;R incoming=g[row*width+i%width];dy[i]=incoming*w[at];
}
extern "C" __global__ void lm_combine_weight_backward(const R* g,const R* y,const I* ids,const I* slots,R* dw,I rows,I width,I base,I top_k){
    I route=blockIdx.x,j=threadIdx.x;
    if(route>=rows)return;
    I row=ids[base+route],slot=slots[base+route];
    __shared__ R tmp[256];
    R total=0;
    for(I d=j;d<width;d+=blockDim.x)total+=g[row*width+d]*y[route*width+d];
    total=block_sum(total,tmp);
    if(j==0)dw[row*top_k+slot]=total;
}
extern "C" __global__ void lm_head_rows(const R* x,R* out,I n,I start,I width,I time,I backward){
    ITEM;I row=start+i/width,at=((row/(time-1))*time+row%(time-1))*width+i%width;if(backward)out[at]+=x[i];else out[i]=x[at];
}
extern "C" __global__ void lm_head_loss(const R* logits,const I* labels,R* out,I rows,I vocab){
    I row=blockIdx.x,j=threadIdx.x;if(row>=rows)return;__shared__ R tmp[256];
    R peak=-(R)__int_as_float(0x7f800000);for(I c=j;c<vocab;c+=blockDim.x)peak=fmax(peak,logits[row*vocab+c]);peak=block_max(peak,tmp);
    R total=0;for(I c=j;c<vocab;c+=blockDim.x)total+=exp(logits[row*vocab+c]-peak);total=block_sum(total,tmp);
    if(j==0)out[row]=labels[row]==-100?0:log(total)-(logits[row*vocab+labels[row]]-peak);
}
extern "C" __global__ void lm_head_gradient(R* logits,const I* labels,const R* gradient,I rows,I vocab,R factor){
    I row=blockIdx.x,j=threadIdx.x;if(row>=rows)return;__shared__ R tmp[256];
    R peak=-(R)__int_as_float(0x7f800000);for(I c=j;c<vocab;c+=blockDim.x)peak=fmax(peak,logits[row*vocab+c]);peak=block_max(peak,tmp);
    R total=0;for(I c=j;c<vocab;c+=blockDim.x)total+=exp(logits[row*vocab+c]-peak);total=block_sum(total,tmp);
    for(I c=j;c<vocab;c+=blockDim.x)logits[row*vocab+c]=labels[row]==-100?0:gradient[0]*factor*(exp(logits[row*vocab+c]-peak)/total-(c==labels[row]));
}
