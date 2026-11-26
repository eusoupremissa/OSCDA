#include <osc/osc.h>

#include <stdio.h>
#include <stdlib.h>

// The device definitions are packaged separately with nvFatbin.
extern "C" __attribute__((global)) void matmul(const float *, const float *, float *,
                                               unsigned int, unsigned int, unsigned int) {}
extern "C" __attribute__((global)) void relu(float *, unsigned int) {}

static void check(cudaError_t error)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

extern "C" int osc_inference_test()
{
    if (oscGetBackend() != oscBackendAmdHip) {
        fputs("AMD HIP backend unavailable; rerun with OSC_HIP_DIAGNOSTICS=1\n", stderr);
        return 77;
    }

    constexpr unsigned int m = 19, n = 23, k = 29, tile = 16, threads = 128;
    float a[m * k], b[k * n];
    float *da, *db, *dc, *output;
    for (unsigned int i = 0; i < m * k; ++i)
        a[i] = static_cast<int>(i % 9) - 4;
    for (unsigned int i = 0; i < k * n; ++i)
        b[i] = static_cast<int>(i % 7) - 3;

    check(cudaMalloc(reinterpret_cast<void **>(&da), sizeof(a)));
    check(cudaMalloc(reinterpret_cast<void **>(&db), sizeof(b)));
    check(cudaMalloc(reinterpret_cast<void **>(&dc), m * n * sizeof(float)));
    check(cudaMallocHost(reinterpret_cast<void **>(&output), m * n * sizeof(float)));
    check(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));

    cudaStream_t producer, consumer;
    cudaEvent_t ready;
    check(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    dim3 grid((n + tile - 1) / tile, (m + tile - 1) / tile), block(tile, tile);

    for (int pass = 0; pass < 2; ++pass) {
        if (pass) {
            for (float &value : a)
                value = -value;
            check(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
        }
        matmul<<<grid, block, 0, producer>>>(da, db, dc, m, n, k);
        check(cudaGetLastError());
        check(cudaEventRecord(ready, producer));
        check(cudaStreamWaitEvent(consumer, ready, 0));
        relu<<<(m * n + threads - 1) / threads, threads, 0, consumer>>>(dc, m * n);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(output, dc, m * n * sizeof(float), cudaMemcpyDeviceToHost, consumer));
        check(cudaStreamSynchronize(consumer));

        for (unsigned int row = 0; row < m; ++row) {
            for (unsigned int col = 0; col < n; ++col) {
                float expected = 0;
                for (unsigned int j = 0; j < k; ++j)
                    expected += a[row * k + j] * b[j * n + col];
                expected = expected < 0 ? 0 : expected;
                if (output[row * n + col] != expected) {
                    fprintf(stderr, "pass %d, output (%u, %u): got %g, expected %g\n",
                            pass + 1, row, col, output[row * n + col], expected);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    check(cudaEventDestroy(ready));
    check(cudaStreamDestroy(consumer));
    check(cudaStreamDestroy(producer));
    check(cudaFreeHost(output));
    check(cudaFree(dc));
    check(cudaFree(db));
    check(cudaFree(da));
    int device;
    char arch[256];
    check(cudaGetDevice(&device));
    check(oscGetDeviceArch(device, arch, sizeof(arch)));
    printf("%s: two MatMul + ReLU passes, %u exact CPU matches each\n", arch, m * n);
    return EXIT_SUCCESS;
}
