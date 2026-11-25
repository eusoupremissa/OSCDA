#include <osc/osc.h>

#include <stdio.h>
#include <stdlib.h>

enum { TILE = 16, MAX_DIM = 48 };

static void check(cudaError_t error)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

static void compare(oscFunction_t kernel, unsigned int m, unsigned int n, unsigned int k)
{
    float a[MAX_DIM * MAX_DIM], b[MAX_DIM * MAX_DIM], result[MAX_DIM * MAX_DIM];
    size_t a_bytes = m * k * sizeof(float), b_bytes = k * n * sizeof(float);
    size_t c_bytes = m * n * sizeof(float);
    for (unsigned int i = 0; i < m * k; ++i)
        a[i] = ((int)((i * 17 + i / k) % 31) - 15) * 0.125f;
    for (unsigned int i = 0; i < k * n; ++i)
        b[i] = ((int)((i * 13 + i / n) % 23) - 11) * 0.0625f;

    void *device_a, *device_b, *device_c;
    check(cudaMalloc(&device_a, a_bytes));
    check(cudaMalloc(&device_b, b_bytes));
    check(cudaMalloc(&device_c, c_bytes));
    check(cudaMemcpy(device_a, a, a_bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_b, b, b_bytes, cudaMemcpyHostToDevice));
    check(cudaMemset(device_c, 0xff, c_bytes));

    void *arguments[] = {&device_a, &device_b, &device_c, &m, &n, &k};
    check(oscLaunchKernel(kernel, (n + TILE - 1) / TILE, (m + TILE - 1) / TILE, 1,
                          TILE, TILE, 1, 0, arguments));
    check(cudaMemcpy(result, device_c, c_bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(device_c));
    check(cudaFree(device_b));
    check(cudaFree(device_a));

    for (unsigned int row = 0; row < m; ++row) {
        for (unsigned int col = 0; col < n; ++col) {
            float expected = 0.0f;
            for (unsigned int i = 0; i < k; ++i)
                expected += a[row * k + i] * b[i * n + col];
            float actual = result[row * n + col];
            // All products and sums in this fixture are exactly representable in FP32.
            if (actual != expected) {
                fprintf(stderr, "C[%u,%u]: GPU %.9g, CPU %.9g\n", row, col, actual, expected);
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("%ux%u * %ux%u: %u outputs matched CPU exactly\n", m, k, k, n, m * n);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s matmul.hsaco\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (oscGetBackend() != oscBackendAmdHip) {
        fputs("OSC could not find a usable AMD HIP device\n", stderr);
        return EXIT_FAILURE;
    }
    int device;
    cudaDeviceProp properties;
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    printf("%s\n", properties.name);

    oscModule_t module;
    oscFunction_t kernel;
    check(oscModuleLoad(&module, argv[1]));
    check(oscModuleGetFunction(&kernel, module, "matmul"));
    compare(kernel, 32, 32, 32);
    compare(kernel, 37, 29, 43);
    check(oscModuleUnload(module));
    return EXIT_SUCCESS;
}
