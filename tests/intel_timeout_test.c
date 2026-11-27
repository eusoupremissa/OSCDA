#include <osc/osc.h>

#undef NDEBUG
#include <assert.h>

int main(void)
{
    void *device = NULL;
    assert(oscGetBackend() == oscBackendIntelLevelZero);
    assert(cudaMalloc(&device, 4096) == cudaSuccess);
    assert(cudaMemset(device, 0, 4096) == cudaErrorNotReady);

    oscIntelMemoryInfo_t info;
    assert(oscIntelGetMemoryInfo(&info) == cudaSuccess);
    assert(info.stalled == 1);
    assert(cudaDeviceSynchronize() == cudaErrorNotReady);
    return 0;
}
