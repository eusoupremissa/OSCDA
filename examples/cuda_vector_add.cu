#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

// The device definition is assembled separately in vector_add.ptx.
extern "C" __attribute__((global)) void vector_add(const float *, const float *, float *,
                                                    unsigned int) {}

static void check(cudaError_t error)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    constexpr unsigned int count = 4099;
    float a[count], b[count], output[count];
    for (unsigned int i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 1.0f + (i % 7) * 0.25f;
    }

    float *da, *db, *result;
    check(cudaMalloc(reinterpret_cast<void **>(&da), sizeof(a)));
    check(cudaMalloc(reinterpret_cast<void **>(&db), sizeof(b)));
    check(cudaMalloc(reinterpret_cast<void **>(&result), sizeof(output)));
    check(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));

    for (unsigned int launch = 0; launch < 2; ++launch) {
        check(cudaMemset(result, 0, sizeof(output)));
        vector_add<<<(count + 255) / 256, 256>>>(da, db, result, count);
        check(cudaGetLastError());
        check(cudaMemcpy(output, result, sizeof(output), cudaMemcpyDeviceToHost));
        for (unsigned int i = 0; i < count; ++i) {
            if (output[i] != a[i] + b[i]) {
                fprintf(stderr, "launch %u: incorrect output at %u\n", launch, i);
                return EXIT_FAILURE;
            }
        }
    }
    check(cudaFree(result));
    check(cudaFree(db));
    check(cudaFree(da));
    cudaDeviceProp properties;
    int device;
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    printf("%s: two CUDA launches, %u correct results each\n", properties.name, count);
}
