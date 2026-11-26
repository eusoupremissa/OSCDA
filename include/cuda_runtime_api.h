#ifndef OSCDA_CUDA_RUNTIME_API_H
#define OSCDA_CUDA_RUNTIME_API_H

#include <stddef.h>

#if defined(_WIN32)
#if defined(OSCDA_BUILD)
#define CUDARTAPI __declspec(dllexport)
#else
#define CUDARTAPI __declspec(dllimport)
#endif
#define CUDART_CB __stdcall
#else
#define CUDARTAPI __attribute__((visibility("default")))
#define CUDART_CB
#endif

typedef struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    constexpr dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1)
        : x(x), y(y), z(z) {}
#endif
} dim3;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorInvalidDevice = 101,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorNotReady = 600,
    cudaErrorNotSupported = 801,
    cudaErrorUnknown = 999
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef struct osc_stream *cudaStream_t;
typedef struct osc_event *cudaEvent_t;
typedef void(CUDART_CB *cudaHostFn_t)(void *userData);

enum {
    cudaStreamDefault = 0,
    cudaStreamNonBlocking = 1,
    cudaEventDefault = 0,
    cudaEventBlockingSync = 1,
    cudaEventDisableTiming = 2,
    cudaEventInterprocess = 4
};

typedef struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    int multiProcessorCount;
    int concurrentKernels;
    int unifiedAddressing;
} cudaDeviceProp;

CUDARTAPI const char *cudaGetErrorName(cudaError_t error);
CUDARTAPI const char *cudaGetErrorString(cudaError_t error);
CUDARTAPI cudaError_t cudaGetLastError(void);
CUDARTAPI cudaError_t cudaPeekAtLastError(void);

CUDARTAPI cudaError_t cudaGetDeviceCount(int *count);
CUDARTAPI cudaError_t cudaGetDevice(int *device);
CUDARTAPI cudaError_t cudaSetDevice(int device);
CUDARTAPI cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
CUDARTAPI cudaError_t cudaDeviceSynchronize(void);

CUDARTAPI cudaError_t cudaMalloc(void **ptr, size_t size);
CUDARTAPI cudaError_t cudaFree(void *ptr);
CUDARTAPI cudaError_t cudaMallocHost(void **ptr, size_t size);
CUDARTAPI cudaError_t cudaFreeHost(void *ptr);
CUDARTAPI cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
CUDARTAPI cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
CUDARTAPI cudaError_t cudaMemset(void *dst, int value, size_t count);
CUDARTAPI cudaError_t cudaMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream);

CUDARTAPI cudaError_t cudaStreamCreate(cudaStream_t *stream);
CUDARTAPI cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
CUDARTAPI cudaError_t cudaStreamDestroy(cudaStream_t stream);
CUDARTAPI cudaError_t cudaStreamSynchronize(cudaStream_t stream);
CUDARTAPI cudaError_t cudaStreamQuery(cudaStream_t stream);
CUDARTAPI cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
CUDARTAPI cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags);
CUDARTAPI cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
CUDARTAPI cudaError_t cudaLaunchKernel(const void *function, dim3 grid, dim3 block,
                                    void **arguments, size_t shared, cudaStream_t stream);
CUDARTAPI unsigned int __cudaPushCallConfiguration(dim3 grid, dim3 block,
#ifdef __cplusplus
                                                   size_t shared = 0, cudaStream_t stream = nullptr);
#else
                                                   size_t shared, cudaStream_t stream);
#endif
CUDARTAPI cudaError_t __cudaPopCallConfiguration(dim3 *grid, dim3 *block,
                                                size_t *shared, void *stream);

CUDARTAPI cudaError_t cudaEventCreate(cudaEvent_t *event);
CUDARTAPI cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
CUDARTAPI cudaError_t cudaEventDestroy(cudaEvent_t event);
CUDARTAPI cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
CUDARTAPI cudaError_t cudaEventSynchronize(cudaEvent_t event);
CUDARTAPI cudaError_t cudaEventQuery(cudaEvent_t event);
CUDARTAPI cudaError_t cudaEventElapsedTime(float *milliseconds, cudaEvent_t start, cudaEvent_t end);

#ifdef __cplusplus
}
#endif

#endif
