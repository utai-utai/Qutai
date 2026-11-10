typedef long long I;
struct C { REAL re, im; };

__device__ C rotation(C a,C b,REAL angle,I axis,I bit) {
    REAL c=cos(angle/(REAL)2),s=sin(angle/(REAL)2),sign=bit?(REAL)1:(REAL)-1;
    if(axis==0) return {c*a.re+s*b.im,c*a.im-s*b.re};
    if(axis==1) return {c*a.re+sign*s*b.re,c*a.im+sign*s*b.im};
    return {c*a.re-sign*s*a.im,c*a.im+sign*s*a.re};
}

__device__ REAL angle_value(const REAL* inputs,const REAL* weights,const I* gate,I row,I q) {
    return gate[2]==0?inputs[row*q+gate[3]]:gate[2]==1?weights[gate[3]]:(REAL)0.78539816339744830962;
}

extern "C" __global__ void qb_basis(C* out,I size,I width) {
    I i=(I)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size) out[i]={(REAL)(i%width==0),0};
}

extern "C" __global__ void qb_gate(const C* in,C* out,const REAL* inputs,const REAL* weights,
        const I* plan,I size,I width,I q,I start,I gate_index,I inverse) {
    I i=(I)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size) return;
    I row=i/width, basis=i%width;
    const I* gate=plan+gate_index*4;
    I mask=1LL<<(q-1-gate[1]);
    if(gate[0]<3) {
        REAL angle=angle_value(inputs,weights,gate,start+row,q);
        if(inverse) angle=-angle;
        out[i]=rotation(in[i],in[row*width+(basis^mask)],angle,gate[0],(basis&mask)!=0);
    } else {
        I control=1LL<<(q-1-gate[2]);
        if(gate[0]==3) out[i]=in[row*width+((basis&control)?(basis^mask):basis)];
        else { REAL sign=((basis&control)&&(basis&mask))?(REAL)-1:(REAL)1;out[i]={sign*in[i].re,sign*in[i].im}; }
    }
}

extern "C" __global__ void qb_readout(const C* values,REAL* out,I width,I q,I start) {
    I group=blockIdx.x,row=group/q,wire=group%q,mask=1LL<<(q-1-wire);
    REAL total=0;
    for(I basis=threadIdx.x;basis<width;basis+=blockDim.x) {
        C a=values[row*width+basis];
        total+=((basis&mask)?(REAL)-1:(REAL)1)*(a.re*a.re+a.im*a.im);
    }
    __shared__ REAL partials[256];partials[threadIdx.x]=total;__syncthreads();
    for(I step=128;step;step/=2) {if(threadIdx.x<step)partials[threadIdx.x]+=partials[threadIdx.x+step];__syncthreads();}
    if(!threadIdx.x) out[(start+row)*q+wire]=partials[0];
}

extern "C" __global__ void qb_bra(const C* values,const REAL* gradient,C* out,I size,I width,I q,I start) {
    I i=(I)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size) return;
    I row=i/width,basis=i%width;REAL coefficient=0;
    for(I wire=0;wire<q;wire++) coefficient+=gradient[(start+row)*q+wire]*((basis&(1LL<<(q-1-wire)))?(REAL)-1:(REAL)1);
    out[i]={coefficient*values[i].re,coefficient*values[i].im};
}

extern "C" __global__ void qb_gradient(const C* bra,const C* previous,const REAL* inputs,const REAL* weights,
        const I* plan,REAL* input_gradient,REAL* weight_gradient,I width,I q,I start,I gate_index,I input_grad,I weight_grad) {
    I row=blockIdx.x;const I* gate=plan+gate_index*4;
    I mask=1LL<<(q-1-gate[1]);
    REAL angle=angle_value(inputs,weights,gate,start+row,q)+(REAL)3.14159265358979323846;
    REAL total=0;
    for(I basis=threadIdx.x;basis<width;basis+=blockDim.x) {
        I i=row*width+basis;
        C d=rotation(previous[i],previous[row*width+(basis^mask)],angle,gate[0],(basis&mask)!=0),b=bra[i];
        // 2 Re(<bra| R(theta+pi)/2 |previous>) cancels the two factors.
        total+=b.re*d.re+b.im*d.im;
    }
    __shared__ REAL partials[256];partials[threadIdx.x]=total;__syncthreads();
    for(I step=128;step;step/=2) {if(threadIdx.x<step)partials[threadIdx.x]+=partials[threadIdx.x+step];__syncthreads();}
    if(!threadIdx.x) {
        if(gate[2]==0 && input_grad) input_gradient[(start+row)*q+gate[3]]=partials[0];
        if(gate[2]==1 && weight_grad) atomicAdd(weight_gradient+gate[3],partials[0]);
    }
}
