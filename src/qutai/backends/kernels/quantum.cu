// Complex values use interleaved REAL real/imaginary components.

struct Complex { REAL real, imag; };

__device__ Complex cadd(Complex a, Complex b) {
    return {a.real + b.real, a.imag + b.imag};
}

__device__ Complex cmul(Complex a, Complex b) {
    return {a.real*b.real - a.imag*b.imag, a.real*b.imag + a.imag*b.real};
}

__device__ long long outcome(long long index, const long long* masks, long long wires) {
    long long result = 0;
    for (long long wire = 0; wire < wires; ++wire)
        result = (result << 1) | ((index & masks[wire]) != 0);
    return result;
}

__device__ long long offset(long long index, const long long* masks, long long wires) {
    long long result = 0;
    for (long long wire = 0; wire < wires; ++wire)
        if ((index >> (wires - 1 - wire)) & 1) result |= masks[wire];
    return result;
}

extern "C" __global__ void q_basis(Complex* result, long long size) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) result[i] = {(REAL)(i == 0), 0};
}

extern "C" __global__ void q_gate(const Complex* values, const Complex* matrix,
        const long long* masks, Complex* result, long long size,
        long long wires, long long selected) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    long long row = outcome(i, masks, wires), base = i & ~selected;
    long long width = 1LL << wires;
    Complex total = {0, 0};
    for (long long column = 0; column < width; ++column)
        total = cadd(total, cmul(matrix[row*width+column], values[base | offset(column, masks, wires)]));
    result[i] = total;
}

extern "C" __global__ void q_probabilities(const Complex* values, const long long* masks,
        const long long* hidden_masks, REAL* result, long long outputs,
        long long wires, long long hidden_wires) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outputs) return;
    long long base = offset(i, masks, wires);
    REAL total = 0;
    for (long long hidden = 0; hidden < (1LL << hidden_wires); ++hidden) {
        Complex a = values[base | offset(hidden, hidden_masks, hidden_wires)];
        total += a.real*a.real + a.imag*a.imag;
    }
    result[i] = total;
}

extern "C" __global__ void q_collapse(const Complex* values, const long long* masks,
        Complex* result, long long size, long long wires, long long selected_outcome, REAL scale) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    Complex a = values[i];
    result[i] = outcome(i, masks, wires) == selected_outcome
        ? Complex{a.real*scale, a.imag*scale} : Complex{0, 0};
}

extern "C" __global__ void q_inner(const Complex* left, const Complex* right,
        Complex* result, long long size) {
    if (blockIdx.x || threadIdx.x) return;
    Complex total = {0, 0};
    for (long long i = 0; i < size; ++i) {
        Complex a = left[i], b = right[i];
        total.real += a.real*b.real + a.imag*b.imag;
        total.imag += a.real*b.imag - a.imag*b.real;
    }
    result[0] = total;
}

extern "C" __global__ void q_real_inner(const REAL* left, const REAL* right,
        REAL* result, long long components, REAL scale) {
    if (blockIdx.x || threadIdx.x) return;
    REAL total = 0;
    for (long long i = 0; i < components; ++i) total += left[i]*right[i];
    result[0] = scale*total;
}

extern "C" __global__ void q_weighted_sum(const REAL* values, const REAL* weights,
        REAL* result, long long components, long long weight_index) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < components) result[i] += weights[weight_index]*values[i];
}

extern "C" __global__ void q_probabilities_backward(const REAL* gradient,
        const Complex* values, const long long* masks, Complex* result,
        long long size, long long wires) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    REAL g = gradient[outcome(i, masks, wires)];
    result[i] = {g*values[i].real, g*values[i].imag};
}

extern "C" __global__ void q_rotation(const Complex* values, const REAL* angle,
        Complex* result, long long size, long long mask, long long axis,
        long long adjoint, long long derivative) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    // d R(theta)/d theta = R(theta + pi) / 2.
    REAL theta = angle[0] + (derivative ? (REAL)3.1415926535897932384626433832795 : (REAL)0);
    if (adjoint) theta = -theta;
    REAL c = cos(theta/(REAL)2), s = sin(theta/(REAL)2);
    REAL scale = derivative ? (REAL)0.5 : (REAL)1;
    Complex a = values[i], b = values[i ^ mask], value;
    if (axis == 0) { // RX: off-diagonal coefficient -i sin(theta/2).
        value = {c*a.real + s*b.imag, c*a.imag - s*b.real};
    } else if (axis == 1) { // RY: row zero has -sin, row one has +sin.
        REAL sign = (i & mask) ? (REAL)1 : (REAL)-1;
        value = {c*a.real + sign*s*b.real, c*a.imag + sign*s*b.imag};
    } else { // RZ: exp(-i theta/2) for zero, exp(+i theta/2) for one.
        REAL sign = (i & mask) ? (REAL)1 : (REAL)-1;
        value = {c*a.real - sign*s*a.imag, c*a.imag + sign*s*a.real};
    }
    result[i] = {scale*value.real, scale*value.imag};
}
