#include <osc/osc.h>

#undef NDEBUG
#include <assert.h>
#include <string.h>

int main(int argc, char **argv)
{
    int integrated = argc == 1;
    (void)argv;
    assert(oscGetBackend() == oscBackendIntelLevelZero);
    assert(strcmp(oscGetBackendName(), "Intel Level Zero") == 0);

    cudaDeviceProp properties;
    assert(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    assert(strcmp(properties.name, "Mock Intel integrated GPU") == 0);
    assert(properties.warpSize == 16);

    oscIntelMemoryInfo_t info;
    assert(oscIntelGetMemoryInfo(&info) == cudaSuccess);
    assert(info.physicalBytes == (size_t)1024 * 1024 * 1024);
    assert(info.budgetBytes == (size_t)64 * 1024 * 1024);
    assert(info.allocatedBytes == 0);
    assert(info.integrated == (unsigned int)integrated);
    assert(info.timeoutMs == 25);

    unsigned char input[4096], output[4096];
    for (size_t i = 0; i < sizeof(input); ++i)
        input[i] = (unsigned char)i;
    void *device = NULL;
    assert(cudaMalloc(&device, sizeof(input)) == cudaSuccess);
    assert(oscIntelGetMemoryInfo(&info) == cudaSuccess);
    assert(info.allocatedBytes == sizeof(input));
    assert(oscIntelSetMemoryBudget(sizeof(input) - 1) == cudaErrorInvalidValue);
    assert(cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(memcmp(input, output, sizeof(input)) == 0);
    assert(cudaMemset(device, 0x5a, sizeof(input)) == cudaSuccess);
    assert(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t i = 0; i < sizeof(output); ++i)
        assert(output[i] == 0x5a);
    assert(cudaFree(device) == cudaSuccess);
    assert(oscIntelGetMemoryInfo(&info) == cudaSuccess);
    assert(info.allocatedBytes == 0);
    return 0;
}
