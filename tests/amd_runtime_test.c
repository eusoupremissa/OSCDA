#include <osc/osc.h>

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void check(cudaError_t error)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        assert(error == cudaSuccess);
    }
}

int main(void)
{
    if (oscGetBackend() != oscBackendAmdHip) {
        puts("AMD HIP backend unavailable");
        return 77;
    }

    int count = 0;
    cudaDeviceProp prop;
    check(cudaGetDeviceCount(&count));
    assert(count > 0);
    check(cudaGetDeviceProperties(&prop, 0));
    assert(prop.name[0] != '\0');
    assert(prop.totalGlobalMem > 0);

    unsigned char input[4096];
    unsigned char output[4096];
    for (size_t i = 0; i < sizeof(input); ++i)
        input[i] = (unsigned char)i;
    memset(output, 0, sizeof(output));

    void *device = NULL;
    check(cudaMalloc(&device, sizeof(input)));
    check(cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice));
    check(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost));
    assert(memcmp(input, output, sizeof(input)) == 0);
    check(cudaMemset(device, 0x5a, sizeof(input)));
    check(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < sizeof(output); ++i)
        assert(output[i] == 0x5a);
    check(cudaFree(device));

    printf("OSC AMD backend: %s (%s)\n", prop.name, oscGetBackendName());
    return 0;
}
