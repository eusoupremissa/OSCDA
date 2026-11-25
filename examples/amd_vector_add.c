#include <osc/osc.h>

#include <stdio.h>
#include <stdlib.h>

enum { ELEMENTS = 4099, BLOCK_SIZE = 256,
       PADDED = (ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE };

static void check(cudaError_t error)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s vector_add.hsaco\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (oscGetBackend() != oscBackendAmdHip) {
        fputs("OSC could not find a usable AMD HIP device\n", stderr);
        return EXIT_FAILURE;
    }

    float a[ELEMENTS];
    float b[ELEMENTS];
    float output[PADDED] = {0};
    for (unsigned int i = 0; i < ELEMENTS; ++i) {
        a[i] = (float)i;
        b[i] = 1.0f + (float)(i % 7) * 0.25f;
    }

    float *device_a = NULL;
    float *device_b = NULL;
    float *device_output = NULL;
    check(cudaMalloc((void **)&device_a, sizeof(a)));
    check(cudaMalloc((void **)&device_b, sizeof(b)));
    check(cudaMalloc((void **)&device_output, sizeof(output)));
    check(cudaMemcpy(device_a, a, sizeof(a), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_b, b, sizeof(b), cudaMemcpyHostToDevice));
    check(cudaMemset(device_output, 0, sizeof(output)));

    oscModule_t module;
    oscFunction_t function;
    check(oscModuleLoad(&module, argv[1]));
    check(oscModuleGetFunction(&function, module, "vector_add"));
    unsigned int count = ELEMENTS;
    void *arguments[] = {&device_a, &device_b, &device_output, &count};
    check(oscLaunchKernel(function, PADDED / BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, arguments));
    check(cudaMemcpy(output, device_output, sizeof(output), cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < PADDED; ++i) {
        float expected = i < ELEMENTS ? a[i] + b[i] : 0.0f;
        if (output[i] != expected) {
            fprintf(stderr, "wrong result at %u: %f (expected %f)\n", i, output[i], expected);
            return EXIT_FAILURE;
        }
    }

    check(oscModuleUnload(module));
    check(cudaFree(device_output));
    check(cudaFree(device_b));
    check(cudaFree(device_a));
    int device;
    cudaDeviceProp properties;
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    printf("%s: %d results and %d padding values matched\n", properties.name,
           ELEMENTS, PADDED - ELEMENTS);
    return EXIT_SUCCESS;
}
