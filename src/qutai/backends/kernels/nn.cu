// NCHW kernels. Every arithmetic operation runs on the CUDA device.
typedef REAL real;
typedef long long index_t;

extern "C" __global__ void softmax_forward(real* out, const real* x, index_t groups, index_t width, index_t inner) {
    index_t g = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    index_t base = (g / inner) * width * inner + g % inner;
    real peak = x[base];
    for (index_t k = 1; k < width; ++k) if (x[base+k*inner] > peak) peak = x[base+k*inner];
    real total = 0;
    for (index_t k = 0; k < width; ++k) total += exp(x[base+k*inner]-peak);
    for (index_t k = 0; k < width; ++k) out[base+k*inner] = exp(x[base+k*inner]-peak)/total;
}

extern "C" __global__ void conv_forward(real* out, const real* x, const real* w, const real* bias,
    index_t count, index_t C, index_t H, index_t W, index_t O, index_t KH, index_t KW,
    index_t OH, index_t OW, index_t SH, index_t SW, index_t PH, index_t PW, index_t has_bias) {
    index_t i = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    index_t ox=i%OW, oy=(i/OW)%OH, o=(i/(OW*OH))%O, b=i/(OW*OH*O);
    real total = has_bias ? bias[o] : (real)0;
    for (index_t c=0; c<C; ++c) for (index_t ky=0; ky<KH; ++ky) {
        index_t iy=oy*SH+ky-PH;
        if (iy<0 || iy>=H) continue;
        for (index_t kx=0; kx<KW; ++kx) {
            index_t ix=ox*SW+kx-PW;
            if (ix>=0 && ix<W) total += x[((b*C+c)*H+iy)*W+ix] * w[((o*C+c)*KH+ky)*KW+kx];
        }
    }
    out[i]=total;
}

extern "C" __global__ void conv_input_backward(real* out, const real* g, const real* w,
    index_t count, index_t C, index_t H, index_t W, index_t O, index_t KH, index_t KW,
    index_t OH, index_t OW, index_t SH, index_t SW, index_t PH, index_t PW) {
    index_t i=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    index_t ix=i%W, iy=(i/W)%H, c=(i/(W*H))%C, b=i/(W*H*C);
    real total=0;
    for (index_t o=0; o<O; ++o) for (index_t ky=0; ky<KH; ++ky) {
        index_t ty=iy+PH-ky;
        if (ty<0 || ty%SH) continue;
        index_t oy=ty/SH;
        if (oy>=OH) continue;
        for (index_t kx=0; kx<KW; ++kx) {
            index_t tx=ix+PW-kx;
            if (tx<0 || tx%SW) continue;
            index_t ox=tx/SW;
            if (ox<OW) total += g[((b*O+o)*OH+oy)*OW+ox] * w[((o*C+c)*KH+ky)*KW+kx];
        }
    }
    out[i]=total;
}

extern "C" __global__ void conv_weight_backward(real* out, const real* g, const real* x,
    index_t count, index_t N, index_t C, index_t H, index_t W, index_t O, index_t KH, index_t KW,
    index_t OH, index_t OW, index_t SH, index_t SW, index_t PH, index_t PW) {
    index_t i=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    index_t kx=i%KW, ky=(i/KW)%KH, c=(i/(KW*KH))%C, o=i/(KW*KH*C);
    real total=0;
    for (index_t b=0; b<N; ++b) for (index_t oy=0; oy<OH; ++oy) {
        index_t iy=oy*SH+ky-PH;
        if (iy<0 || iy>=H) continue;
        for (index_t ox=0; ox<OW; ++ox) {
            index_t ix=ox*SW+kx-PW;
            if (ix>=0 && ix<W) total += g[((b*O+o)*OH+oy)*OW+ox] * x[((b*C+c)*H+iy)*W+ix];
        }
    }
    out[i]=total;
}

extern "C" __global__ void conv_bias_backward(real* out, const real* g, index_t N, index_t O, index_t area) {
    index_t o=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (o>=O) return;
    real total=0;
    for (index_t b=0; b<N; ++b) for (index_t p=0; p<area; ++p) total += g[(b*O+o)*area+p];
    out[o]=total;
}

extern "C" __global__ void pool_forward(real* out, index_t* indices, const real* x,
    index_t count, index_t H, index_t W, index_t KH, index_t KW,
    index_t OH, index_t OW, index_t SH, index_t SW, index_t PH, index_t PW) {
    index_t i=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    index_t ox=i%OW, oy=(i/OW)%OH, plane=i/(OW*OH), selected=-1;
    real best=0;
    for (index_t ky=0; ky<KH; ++ky) {
        index_t iy=oy*SH+ky-PH;
        if (iy<0 || iy>=H) continue;
        for (index_t kx=0; kx<KW; ++kx) {
            index_t ix=ox*SW+kx-PW;
            if (ix<0 || ix>=W) continue;
            index_t p=(plane*H+iy)*W+ix;
            if (selected<0 || x[p]>best) { selected=p; best=x[p]; }
        }
    }
    out[i]=best; indices[i]=selected;
}

extern "C" __global__ void pool_backward(real* out, const real* g, const index_t* indices, index_t count) {
    index_t i=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) atomicAdd(out+indices[i],g[i]);
}

extern "C" __global__ void channel_stats_forward(real* averages, real* variances, const real* x,
    index_t N, index_t C, index_t area) {
    index_t c=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=C) return;
    real total=0;
    for (index_t b=0; b<N; ++b) for (index_t p=0; p<area; ++p) total += x[(b*C+c)*area+p];
    real average=total/(N*area), variance=0;
    for (index_t b=0; b<N; ++b) for (index_t p=0; p<area; ++p) {
        real centered=x[(b*C+c)*area+p]-average;
        variance += centered*centered;
    }
    averages[c]=average; variances[c]=variance/(N*area);
}

extern "C" __global__ void batch_norm_inverse(real* inverse, const real* variances, index_t C, real eps) {
    index_t c=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (c<C) inverse[c]=(real)1/sqrt(variances[c]+eps);
}

extern "C" __global__ void batch_norm_forward(real* out, real* normalized, real* inverse,
    const real* x, const real* w, const real* bias, const real* averages, const real* variances,
    index_t count, index_t C, index_t area, real eps) {
    index_t i=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    index_t c=(i/area)%C;
    real scale=inverse[c];
    real value=(x[i]-averages[c])*scale;
    out[i]=value*w[c]+bias[c]; normalized[i]=value;
}

extern "C" __global__ void norm_parameter_backward(real* dw, real* db, const real* g, const real* normalized,
    index_t groups, index_t C, index_t area) {
    index_t c=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=C) return;
    real weight=0, bias=0;
    for (index_t b=0; b<groups; ++b) for (index_t p=0; p<area; ++p) {
        index_t i=(b*C+c)*area+p;
        weight += g[i]*normalized[i]; bias += g[i];
    }
    dw[c]=weight; db[c]=bias;
}

extern "C" __global__ void batch_norm_input_backward(real* out, const real* g, const real* normalized,
    const real* inverse, const real* w, const real* dw, const real* db,
    index_t count, index_t C, index_t area, index_t channel_count, index_t train) {
    index_t i=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    index_t c=(i/area)%C;
    real incoming=g[i];
    if (train) incoming -= (db[c]+normalized[i]*dw[c])/channel_count;
    out[i]=incoming*w[c]*inverse[c];
}

extern "C" __global__ void layer_norm_forward(real* out, real* normalized, real* inverse,
    const real* x, const real* w, const real* bias, index_t groups, index_t width, real eps) {
    index_t g=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (g>=groups) return;
    index_t start=g*width;
    real average=0;
    for (index_t k=0; k<width; ++k) average += x[start+k];
    average /= width;
    real variance=0;
    for (index_t k=0; k<width; ++k) { real c=x[start+k]-average; variance += c*c; }
    real scale=(real)1/sqrt(variance/width+eps);
    inverse[g]=scale;
    for (index_t k=0; k<width; ++k) {
        real value=(x[start+k]-average)*scale;
        normalized[start+k]=value; out[start+k]=value*w[k]+bias[k];
    }
}

extern "C" __global__ void layer_norm_input_backward(real* out, const real* g, const real* normalized,
    const real* inverse, const real* w, index_t groups, index_t width) {
    index_t group=(index_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (group>=groups) return;
    index_t start=group*width;
    real average=0, projected=0;
    for (index_t k=0; k<width; ++k) {
        real weighted=g[start+k]*w[k];
        average += weighted; projected += weighted*normalized[start+k];
    }
    average/=width; projected/=width;
    for (index_t k=0; k<width; ++k)
        out[start+k]=inverse[group]*(g[start+k]*w[k]-average-normalized[start+k]*projected);
}
