// Decoder 专用计算；所有浮点值与路由表均保留在 Metal buffer。
kernel void lm_silu(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    float x=a[i], s=1/(1+exp(-x));
    out[i]=p[1] ? b[i]*s*(1+x*(1-s)) : x*s;
}

kernel void lm_rope(device const float *a [[buffer(0)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], constant float *f [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    uint dim=p[3], j=i%dim, t=(i/(dim*p[2]))%p[1], base=i-j+(j/2)*2;
    float angle=(t+p[4])*pow(f[0],-float(j/2*2)/dim)*(p[5] ? -1:1);
    out[i]=(j%2) ? a[base]*sin(angle)+a[base+1]*cos(angle) : a[base]*cos(angle)-a[base+1]*sin(angle);
}

kernel void lm_route_select(device const float *a [[buffer(0)]], device long *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint experts=p[1], slot=i%p[2], base=(i/p[2])*experts;
    for(uint e=0;e<experts;++e) {
        uint rank=0;
        for(uint j=0;j<experts;++j) rank+=(a[base+j]>a[base+e] || (a[base+j]==a[base+e] && j<e));
        if(rank==slot) { out[i]=e; return; }
    }
}

kernel void lm_route_weights(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint row=i/p[2]; float total=0;
    for(uint s=0;s<p[2];++s) total+=a[row*p[1]+b[row*p[2]+s]];
    out[i]=a[row*p[1]+b[i]]/total;
}

kernel void lm_route_counts(device const long *a [[buffer(0)]], device long *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    long count=0; for(uint j=0;j<p[1];++j) count+=(a[j]==long(i)); out[i]=count;
}

kernel void lm_route_pack(device const long *a [[buffer(0)]], device long *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint count=0; for(uint j=0;j<p[1];++j) if(a[j]==long(i)) out[i*p[1]+count++]=j;
}

kernel void lm_route_util(device const long *a [[buffer(0)]], device float *out [[buffer(3)]], constant float *f [[buffer(5)]], uint i [[thread_position_in_grid]]) { out[i]=float(a[i])*f[0]; }

kernel void lm_route_grad(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const long *c [[buffer(2)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint row=i/p[1], e=i%p[1], base=row*p[2]; float center=0;
    for(uint s=0;s<p[2];++s) center+=a[base+s]*b[base+s];
    float value=0; for(uint s=0;s<p[2];++s) if(c[base+s]==long(e)) value=b[base+s]*(a[base+s]-center);
    out[i]=value;
}

kernel void lm_route_aux_grad(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const float *c [[buffer(2)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint row=i/p[2], e=i%p[2]; float center=0;
    for(uint j=0;j<p[2];++j) center+=b[row*p[2]+j]*c[j];
    out[i]=a[0]*float(p[2])/p[1]*b[i]*(c[e]-center);
}

kernel void lm_route_gather(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint assignment=b[p[1]+i/p[2]], row=assignment/p[3]; out[i]=a[row*p[2]+i%p[2]];
}

kernel void lm_route_scatter(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device atomic_float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint assignment=b[p[1]+i/p[2]], row=assignment/p[3]; atomic_fetch_add_explicit(out+row*p[2]+i%p[2],a[i],memory_order_relaxed);
}

kernel void lm_combine(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const long *c [[buffer(2)]], device atomic_float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint assignment=c[p[1]+i/p[2]], row=assignment/p[3]; atomic_fetch_add_explicit(out+row*p[2]+i%p[2],a[i]*b[assignment],memory_order_relaxed);
}

kernel void lm_combine_grad(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const long *c [[buffer(2)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint assignment=c[p[1]+i/p[2]], row=assignment/p[3]; out[i]=a[row*p[2]+i%p[2]]*b[assignment];
}

kernel void lm_combine_weight_grad(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const long *c [[buffer(2)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint assignment=c[p[1]+i], row=assignment/p[3]; float total=0;
    for(uint j=0;j<p[2];++j) total+=a[row*p[2]+j]*b[i*p[2]+j]; out[assignment]=total;
}

kernel void lm_head_loss(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    long label=b[p[2]+i]; if(label==-100) { out[i]=0; return; }
    uint base=i*p[1]; float peak=-INFINITY,total=0;
    for(uint j=0;j<p[1];++j) peak=max(peak,a[base+j]);
    for(uint j=0;j<p[1];++j) total+=exp(a[base+j]-peak);
    out[i]=log(total)-(a[base+label]-peak);
}

kernel void lm_head_grad(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device const float *c [[buffer(2)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    long label=b[p[2]+i/p[1]]; out[i]=label==-100 ? 0:c[0]/p[3]*(a[i]-(long(i%p[1])==label ? 1:0));
}


kernel void lm_embedding_grad(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint token=i/p[2], column=i%p[2]; float total=0;
    for(uint row=0;row<p[1];++row) if(b[row]==long(token)) total+=a[row*p[2]+column];
    out[i]=total;
}
