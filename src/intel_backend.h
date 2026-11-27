#ifndef OSCDA_INTEL_BACKEND_H
#define OSCDA_INTEL_BACKEND_H

#include <cuda_runtime_api.h>
#include <osc/osc.h>

int osc_intel_available(void);
cudaError_t osc_intel_get_properties(cudaDeviceProp *properties);
cudaError_t osc_intel_synchronize(void);
cudaError_t osc_intel_malloc(void **pointer, size_t size);
cudaError_t osc_intel_free(void *pointer, size_t size);
cudaError_t osc_intel_memcpy(void *destination, const void *source, size_t size);
cudaError_t osc_intel_memset(void *destination, int value, size_t size);
cudaError_t osc_intel_get_memory_info(oscIntelMemoryInfo_t *info);
cudaError_t osc_intel_set_memory_budget(size_t bytes);

#endif
