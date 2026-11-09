#include <metal_stdlib>
using namespace metal;

// 统一缓冲区位置：a/b/c 输入，out 输出，p 整数元数据，f 浮点系数。
kernel void pointwise(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                      device float *out [[buffer(3)]], constant uint *p [[buffer(4)]],
                      constant float *f [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    switch (p[1]) {
        case 0: out[i] = a[i] + b[i]; break;
        case 1: out[i] = a[i] - b[i]; break;
        case 2: out[i] = a[i] * b[i]; break;
        case 3: out[i] = a[i] / b[i]; break;
        case 4: out[i] = a[i] * f[0]; break;
        case 5: out[i] = -a[i]; break;
        case 6: out[i] = a[i] * a[i]; break;
        case 7: out[i] = max(a[i], 0.0f); break;
        case 8: out[i] = b[i] > 0 ? a[i] : 0; break;
        case 9: out[i] = f[0]; break;
        case 10: out[i] = f[0] * a[i] + (1 - f[0]) * b[i]; break;
    }
}

kernel void reindex(device const float *a [[buffer(0)]], device float *out [[buffer(3)]],
                    constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint rank = p[1], index = i, source = 0;
    for (int axis = int(rank) - 1; axis >= 0; --axis) {
        source += (index % p[2 + axis]) * p[2 + rank + axis];
        index /= p[2 + axis];
    }
    out[i] = a[source];
}

kernel void reduce_sum(device const float *a [[buffer(0)]], device float *out [[buffer(3)]],
                       constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint count = p[1], inner = p[2];
    uint base = (i / inner) * count * inner + i % inner;
    float total = 0;
    for (uint k = 0; k < count; ++k) total += a[base + k * inner];
    out[i] = total;
}

kernel void matrix_multiply(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                            device float *out [[buffer(3)]], constant uint *p [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    uint rows = p[1], columns = p[2], inner = p[3], rank = p[4];
    uint batch = i / (rows * columns), row = (i / columns) % rows, column = i % columns;
    uint left = 0, right = 0;
    for (int axis = int(rank) - 1; axis >= 0; --axis) {
        uint position = batch % p[5 + axis];
        batch /= p[5 + axis];
        left += position * p[5 + rank + axis];
        right += position * p[5 + 2 * rank + axis];
    }
    float total = 0;
    for (uint k = 0; k < inner; ++k) total += a[left + row * inner + k] * b[right + k * columns + column];
    out[i] = total;
}

kernel void adamw_update(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                         device const float *c [[buffer(2)]], device float *out [[buffer(3)]],
                         constant float *f [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    out[i] = a[i] * f[4] - f[0] * (b[i] / f[1]) / (sqrt(c[i] / f[2]) + f[3]);
}

inline float erf_value(float x) {
    float a = abs(x), t = 1.0f / (1.0f + 0.3275911f * a);
    float y = 1 - (((((1.061405429f*t - 1.453152027f)*t + 1.421413741f)*t - 0.284496736f)*t + 0.254829592f)*t)*exp(-a*a);
    return x < 0 ? -y : y;
}

kernel void math_unary(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                       device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    float x = a[i];
    switch (p[1]) {
        case 0: out[i] = exp(x); break;
        case 1: out[i] = log(x); break;
        case 2: out[i] = x * (1 + erf_value(x * 0.7071067811865475f)) * 0.5f; break;
        case 3: out[i] = b[i] * ((1 + erf_value(x * 0.7071067811865475f))*0.5f + x*exp(-x*x*0.5f)*0.3989422804014327f); break;
        case 4: out[i] = rsqrt(x); break;
    }
}

kernel void fill_index(device long *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    out[i] = as_type<long>((ulong(p[2]) << 32) | p[1]);
}

kernel void gather_float(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) { out[i] = a[b[i]]; }
kernel void gather_wide(device const uint2 *a [[buffer(0)]], device const long *b [[buffer(1)]], device uint2 *out [[buffer(3)]], uint i [[thread_position_in_grid]]) { out[i] = a[b[i]]; }

kernel void scatter_float(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]],
                          device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    float total = 0;
    for (uint k=0; k<p[1]; ++k) if (b[k] == long(i)) total += a[k];
    out[i] = total;
}

kernel void concat_float(device const float *a [[buffer(0)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    out[(i/p[1])*p[2] + p[3] + i%p[1]] = a[i];
}
kernel void concat_wide(device const uint2 *a [[buffer(0)]], device uint2 *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    out[(i/p[1])*p[2] + p[3] + i%p[1]] = a[i];
}

kernel void row_softmax(device const float *a [[buffer(0)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint count=p[1], inner=p[2], base=(i/(count*inner))*count*inner+i%inner;
    float largest=-INFINITY, total=0;
    for (uint k=0; k<count; ++k) largest=max(largest,a[base+k*inner]);
    for (uint k=0; k<count; ++k) total+=exp(a[base+k*inner]-largest);
    out[i]=exp(a[i]-largest)/total;
}

kernel void row_argmax(device const float *a [[buffer(0)]], device long *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint count=p[1], inner=p[2], base=(i/inner)*count*inner+i%inner, best=0;
    for (uint k=1; k<count; ++k) if(a[base+k*inner]>a[base+best*inner]) best=k;
    out[i]=best;
}

kernel void cross_entropy_rows(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint count=p[1], base=i*count;
    float largest=-INFINITY, total=0;
    for(uint k=0;k<count;++k) largest=max(largest,a[base+k]);
    for(uint k=0;k<count;++k) total+=exp(a[base+k]-largest);
    out[i]=log(total)+largest-a[base+b[i]];
}

kernel void cross_entropy_grad(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device const float *c [[buffer(2)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    out[i]=(a[i]-(long(i%p[1])==b[i/p[1]] ? 1.0f:0.0f))*c[0]/p[2];
}

kernel void vector_norm(device const float *a [[buffer(0)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    float scale=0, ssq=1;
    for(uint k=0;k<p[1];++k) {
        float v=abs(a[k]);
        if(!isfinite(v)) { out[0]=v; return; }
        if(v) { if(v>scale) { float r=scale/v; ssq=1+ssq*r*r; scale=v; } else { float r=v/scale; ssq+=r*r; } }
    }
    out[0]=scale*sqrt(ssq);
}

// p: output count, N,C,H,W,O,KH,KW,SH,SW,PH,PW,OH,OW,bias
kernel void convolution(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const float *c [[buffer(2)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint ow=i%p[13], oh=(i/p[13])%p[12], oc=(i/(p[12]*p[13]))%p[5], n=i/(p[5]*p[12]*p[13]);
    float total=p[14] ? c[oc]:0;
    for(uint ic=0;ic<p[2];++ic) for(uint ky=0;ky<p[6];++ky) for(uint kx=0;kx<p[7];++kx) {
        int y=int(oh*p[8]+ky)-int(p[10]), x=int(ow*p[9]+kx)-int(p[11]);
        if(y>=0 && y<int(p[3]) && x>=0 && x<int(p[4]))
            total+=a[((n*p[2]+ic)*p[3]+y)*p[4]+x]*b[((oc*p[2]+ic)*p[6]+ky)*p[7]+kx];
    }
    out[i]=total;
}

kernel void convolution_input_grad(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint x=i%p[4], y=(i/p[4])%p[3], ic=(i/(p[3]*p[4]))%p[2], n=i/(p[2]*p[3]*p[4]);
    float total=0;
    for(uint oc=0;oc<p[5];++oc) for(uint ky=0;ky<p[6];++ky) for(uint kx=0;kx<p[7];++kx) {
        int yy=int(y+p[10])-int(ky), xx=int(x+p[11])-int(kx);
        if(yy>=0 && xx>=0 && yy%p[8]==0 && xx%p[9]==0) {
            uint oh=yy/p[8], ow=xx/p[9];
            if(oh<p[12] && ow<p[13]) total+=a[((n*p[5]+oc)*p[12]+oh)*p[13]+ow]*b[((oc*p[2]+ic)*p[6]+ky)*p[7]+kx];
        }
    }
    out[i]=total;
}

kernel void convolution_weight_grad(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint kx=i%p[7], ky=(i/p[7])%p[6], ic=(i/(p[6]*p[7]))%p[2], oc=i/(p[2]*p[6]*p[7]);
    float total=0;
    for(uint n=0;n<p[1];++n) for(uint oh=0;oh<p[12];++oh) for(uint ow=0;ow<p[13];++ow) {
        int y=int(oh*p[8]+ky)-int(p[10]), x=int(ow*p[9]+kx)-int(p[11]);
        if(y>=0 && y<int(p[3]) && x>=0 && x<int(p[4])) total+=a[((n*p[5]+oc)*p[12]+oh)*p[13]+ow]*b[((n*p[2]+ic)*p[3]+y)*p[4]+x];
    }
    out[i]=total;
}

kernel void pool_indices(device const float *a [[buffer(0)]], device long *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint ow=i%p[13], oh=(i/p[13])%p[12], plane=i/(p[12]*p[13]);
    float largest=-INFINITY; long best=-1;
    for(uint ky=0;ky<p[6];++ky) for(uint kx=0;kx<p[7];++kx) {
        int y=int(oh*p[8]+ky)-int(p[10]), x=int(ow*p[9]+kx)-int(p[11]);
        if(y>=0 && y<int(p[3]) && x>=0 && x<int(p[4])) {
            uint index=(plane*p[3]+y)*p[4]+x;
            if(best<0 || a[index]>largest) { largest=a[index]; best=index; }
        }
    }
    out[i]=best;
}

kernel void scatter_add(device const float *a [[buffer(0)]], device const long *b [[buffer(1)]], device atomic_float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&out[b[i]], a[i], memory_order_relaxed);
}

inline float2 cmul(float2 a,float2 b) { return float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
inline float2 cconj(float2 a) { return float2(a.x,-a.y); }

kernel void complex_pointwise(device const float2 *a [[buffer(0)]], device const float2 *b [[buffer(1)]], device float2 *out [[buffer(3)]], constant uint *p [[buffer(4)]], constant float *f [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    switch(p[1]) {
        case 0: out[i]=a[i]+b[i]; break;
        case 1: out[i]=a[i]-b[i]; break;
        case 2: out[i]=cmul(a[i],b[i]); break;
        case 3: out[i]=cmul(a[i],cconj(b[i]))/dot(b[i],b[i]); break;
        case 4: out[i]=a[i]*f[0]; break;
        case 5: out[i]=-a[i]; break;
        case 9: out[i]=float2(f[0],f[1]); break;
    }
}

kernel void zero_state(device float2 *out [[buffer(3)]], uint i [[thread_position_in_grid]]) { out[i]=float2(i==0 ? 1:0,0); }

kernel void quantum_gate(device const float2 *a [[buffer(0)]], device const float2 *b [[buffer(1)]], device float2 *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint wires=p[1], dimension=1u<<wires, row=0, base=i;
    for(uint j=0;j<wires;++j) { row=2*row+((i>>p[2+j])&1u); base&=~(1u<<p[2+j]); }
    float2 total=0;
    for(uint column=0;column<dimension;++column) {
        uint source=base;
        for(uint j=0;j<wires;++j) source|=((column>>(wires-1-j))&1u)<<p[2+j];
        total+=cmul(b[row*dimension+column],a[source]);
    }
    out[i]=total;
}

inline uint outcome(uint index,constant uint *p) {
    uint result=0;
    for(uint j=0;j<p[2];++j) result=2*result+((index>>p[3+j])&1u);
    return result;
}

kernel void quantum_probabilities(device const float2 *a [[buffer(0)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint qubits=p[1], wires=p[2], base=0;
    for(uint j=0;j<wires;++j) base|=((i>>(wires-1-j))&1u)<<p[3+j];
    float total=0;
    for(uint other=0;other<(1u<<(qubits-wires));++other) {
        uint index=base;
        for(uint j=0;j<qubits-wires;++j) index|=((other>>j)&1u)<<p[3+wires+j];
        total+=dot(a[index],a[index]);
    }
    out[i]=total;
}

kernel void quantum_collapse(device const float2 *a [[buffer(0)]], device float2 *out [[buffer(3)]], constant uint *p [[buffer(4)]], constant float *f [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    out[i]=outcome(i,p)==p[3+p[2]] ? a[i]*f[0]:float2(0);
}

kernel void quantum_probability_grad(device const float2 *a [[buffer(0)]], device const float *b [[buffer(1)]], device float2 *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    out[i]=a[i]*b[outcome(i,p)];
}

kernel void inner_real(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], constant float *f [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    float total=0; for(uint k=0;k<p[1];++k) total+=a[k]*b[k]; out[0]=f[0]*total;
}

kernel void inner_complex(device const float2 *a [[buffer(0)]], device const float2 *b [[buffer(1)]], device float2 *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    float2 total=0; for(uint k=0;k<p[1];++k) total+=cmul(cconj(a[k]),b[k]); out[0]=total;
}

kernel void inner_complex_real(device const float2 *a [[buffer(0)]], device const float2 *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], constant float *f [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    float total=0; for(uint k=0;k<p[1];++k) total+=dot(a[k],b[k]); out[0]=f[0]*total;
}

kernel void weighted_complex(device const float2 *a [[buffer(0)]], device const float *b [[buffer(1)]], device float2 *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) { out[i]=a[i]*b[p[1]]; }
kernel void weighted_real(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) { out[i]=a[i]*b[p[1]]; }

// 设备旋转角度：位移、RX/RY/RZ、伴随、导数。
kernel void quantum_rotation(device const float2 *a [[buffer(0)]], device const float *b [[buffer(1)]], device float2 *out [[buffer(3)]], constant uint *p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint bit = (i >> p[1]) & 1u, other = i ^ (1u << p[1]);
    float angle = (b[0] + (p[4] ? 3.14159265358979323846f : 0)) * (p[3] ? -0.5f : 0.5f);
    float c = cos(angle), s = sin(angle), factor = p[4] ? 0.5f : 1.0f;
    if (p[2] == 0) out[i] = factor * (c*a[i] + cmul(float2(0,-s), a[other]));
    else if (p[2] == 1) out[i] = factor * (c*a[i] + (bit ? s : -s)*a[other]);
    else out[i] = factor * cmul(float2(c, bit ? s : -s), a[i]);
}
