#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstring>
#include <string>

// Python 只接触 C 接口；Metal 对象及其所有权留在桥接层。
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> library;
static NSMutableDictionary *pipelines;
static thread_local std::string error;
static std::string deviceName;
static uint64_t dispatches = 0;
static NSMutableArray *pending;

static int fail(NSString *message) {
    error = [message UTF8String];
    return 0;
}

static int drain() {
    for (id<MTLCommandBuffer> command in pending) {
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return fail(command.error.localizedDescription ?: @"Metal command failed");
    }
    [pending removeAllObjects];
    return 1;
}

static int finish(id<MTLCommandBuffer> command) {
    [command commit];
    [pending addObject:command];
    return pending.count >= 64 ? drain() : 1;
}

extern "C" {
const char *q_error() { return error.c_str(); }
const char *q_device_name() { return deviceName.c_str(); }
uint64_t q_dispatches() { return dispatches; }

int q_init(const char *source) {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device) return fail(@"No Metal device available");
        queue = [device newCommandQueue];
        if (!queue) return fail(@"Cannot create Metal command queue");
        NSError *problem = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                      options:options error:&problem];
        if (!library) return fail(problem.localizedDescription);
        pipelines = [NSMutableDictionary new];
        pending = [NSMutableArray new];
        deviceName = [device.name UTF8String];
        return 1;
    }
}

void *q_alloc(uint64_t bytes) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [device newBufferWithLength:MAX(bytes, 16)
                                                 options:MTLResourceStorageModeShared];
        if (!buffer) { fail(@"Cannot allocate Metal buffer"); return nullptr; }
        return (__bridge_retained void *)buffer;
    }
}

void q_free(void *buffer) {
    if (buffer) { id released = CFBridgingRelease(buffer); (void)released; }
}

void q_write(void *buffer, const void *source, uint64_t bytes) {
    if (bytes) std::memcpy([(__bridge id<MTLBuffer>)buffer contents], source, bytes);
}

int q_read(void *buffer, void *target, uint64_t bytes) {
    @autoreleasepool {
        if (!drain()) return 0;
        if (bytes) std::memcpy(target, [(__bridge id<MTLBuffer>)buffer contents], bytes);
        return 1;
    }
}

int q_read_range(void *buffer, uint64_t offset, void *target, uint64_t bytes) {
    @autoreleasepool {
        if (!drain()) return 0;
        if (bytes) std::memcpy(target, (char *)[(__bridge id<MTLBuffer>)buffer contents] + offset, bytes);
        return 1;
    }
}

int q_write_range(void *buffer, uint64_t offset, const void *source, uint64_t bytes) {
    @autoreleasepool {
        if (!drain()) return 0;
        if (bytes) std::memcpy((char *)[(__bridge id<MTLBuffer>)buffer contents] + offset, source, bytes);
        return 1;
    }
}

int q_copy(void *target, void *source, uint64_t bytes) {
    if (!bytes) return 1;
    @autoreleasepool {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        [encoder copyFromBuffer:(__bridge id<MTLBuffer>)source sourceOffset:0
                       toBuffer:(__bridge id<MTLBuffer>)target destinationOffset:0 size:bytes];
        [encoder endEncoding];
        return finish(command);
    }
}

int q_sync() {
    @autoreleasepool { return drain(); }
}

int q_matmul(void *a, void *b, void *out, uint32_t rows, uint32_t inner, uint32_t columns) {
    @autoreleasepool {
        MPSMatrixDescriptor *ad = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:inner rowBytes:inner*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *bd = [MPSMatrixDescriptor matrixDescriptorWithRows:inner columns:columns rowBytes:columns*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *od = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns rowBytes:columns*4 dataType:MPSDataTypeFloat32];
        MPSMatrix *am = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a descriptor:ad];
        MPSMatrix *bm = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b descriptor:bd];
        MPSMatrix *om = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)out descriptor:od];
        MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:NO resultRows:rows resultColumns:columns interiorColumns:inner alpha:1 beta:0];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        [kernel encodeToCommandBuffer:command leftMatrix:am rightMatrix:bm resultMatrix:om];
        ++dispatches;
        return finish(command);
    }
}

int q_run(const char *name, void **buffers, const uint32_t *meta, uint32_t metaCount,
          const float *scalars, uint32_t scalarCount, uint64_t count) {
    if (!count) return 1;
    @autoreleasepool {
        NSString *key = [NSString stringWithUTF8String:name];
        id<MTLComputePipelineState> pipeline = pipelines[key];
        if (!pipeline) {
            NSError *problem = nil;
            id<MTLFunction> function = [library newFunctionWithName:key];
            if (!function) return fail(@"Unknown Metal kernel");
            pipeline = [device newComputePipelineStateWithFunction:function error:&problem];
            if (!pipeline) return fail(problem.localizedDescription);
            pipelines[key] = pipeline;
        }
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (int index = 0; index < 4; ++index)
            [encoder setBuffer:(__bridge id<MTLBuffer>)(buffers[index] ?: buffers[3]) offset:0 atIndex:index];
        [encoder setBytes:meta length:metaCount * sizeof(uint32_t) atIndex:4];
        [encoder setBytes:scalars length:scalarCount * sizeof(float) atIndex:5];
        NSUInteger width = MIN((NSUInteger)count, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        if (!finish(command)) return 0;
        ++dispatches;
        return 1;
    }
}
}
