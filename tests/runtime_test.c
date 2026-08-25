#include <cuda_runtime.h>

#include <assert.h>
#include <stdio.h>

enum { ELEMENTS = 4096 };

struct vector_job {
    const float *a;
    const float *b;
    float *out;
    size_t count;
};

static void CUDART_CB vector_add(void *data)
{
    struct vector_job *job = data;
    for (size_t i = 0; i < job->count; ++i)
        job->out[i] = job->a[i] + job->b[i];
}

static void check(cudaError_t error)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        assert(error == cudaSuccess);
    }
}

int main(void)
{
    int count = 0;
    cudaDeviceProp prop;
    check(cudaGetDeviceCount(&count));
    check(cudaGetDeviceProperties(&prop, 0));
    assert(count == 1);
    assert(prop.warpSize == 1);

    float a[ELEMENTS];
    float b[ELEMENTS];
    float out[ELEMENTS];
    for (size_t i = 0; i < ELEMENTS; ++i) {
        a[i] = (float)i;
        b[i] = (float)(ELEMENTS - i);
        out[i] = 0.0f;
    }

    float *device_a;
    float *device_b;
    float *device_out;
    check(cudaMalloc((void **)&device_a, sizeof(a)));
    check(cudaMalloc((void **)&device_b, sizeof(b)));
    check(cudaMalloc((void **)&device_out, sizeof(out)));

    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t end;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&end));

    struct vector_job job = {device_a, device_b, device_out, ELEMENTS};
    check(cudaEventRecord(start, stream));
    check(cudaMemcpyAsync(device_a, a, sizeof(a), cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(device_b, b, sizeof(b), cudaMemcpyDefault, stream));
    check(cudaMemsetAsync(device_out, 0, sizeof(out), stream));
    check(cudaLaunchHostFunc(stream, vector_add, &job));
    check(cudaMemcpyAsync(out, device_out, sizeof(out), cudaMemcpyDeviceToHost, stream));
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));

    float elapsed = -1.0f;
    check(cudaEventElapsedTime(&elapsed, start, end));
    assert(elapsed >= 0.0f);
    for (size_t i = 0; i < ELEMENTS; ++i)
        assert(out[i] == (float)ELEMENTS);

    assert(cudaMemcpy(a, b, sizeof(a), cudaMemcpyDeviceToDevice) == cudaErrorInvalidMemcpyDirection);
    assert(cudaGetLastError() == cudaErrorInvalidMemcpyDirection);
    assert(cudaPeekAtLastError() == cudaSuccess);

    check(cudaEventDestroy(end));
    check(cudaEventDestroy(start));
    check(cudaStreamDestroy(stream));
    check(cudaFree(device_out));
    check(cudaFree(device_b));
    check(cudaFree(device_a));
    puts("UniWarp runtime test passed");
    return 0;
}
