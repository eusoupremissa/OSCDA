#ifndef OSCDA_OSC_H
#define OSCDA_OSC_H

#include <cuda_runtime_api.h>

#define OSCDA_VERSION_MAJOR 0
#define OSCDA_VERSION_MINOR 3
#define OSCDA_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

typedef enum oscBackend {
    oscBackendCpu = 0,
    oscBackendIntelLevelZero = 1
} oscBackend_t;

typedef struct oscIntelMemoryInfo {
    size_t physicalBytes;
    size_t budgetBytes;
    size_t allocatedBytes;
    unsigned int integrated;
    unsigned int timeoutMs;
    unsigned int stalled;
} oscIntelMemoryInfo_t;

typedef struct osc_module *oscModule_t;
typedef struct osc_function *oscFunction_t;

CUDARTAPI const char *oscGetRuntimeName(void);
CUDARTAPI oscBackend_t oscGetBackend(void);
CUDARTAPI const char *oscGetBackendName(void);
CUDARTAPI cudaError_t oscIntelGetMemoryInfo(oscIntelMemoryInfo_t *info);
CUDARTAPI cudaError_t oscIntelSetMemoryBudget(size_t bytes);

/* Kernel loading is reserved for the upcoming Intel SPIR-V compiler path. */
CUDARTAPI cudaError_t oscModuleLoad(oscModule_t *module, const char *path);
CUDARTAPI cudaError_t oscModuleUnload(oscModule_t module);
CUDARTAPI cudaError_t oscModuleGetFunction(oscFunction_t *function, oscModule_t module,
                                            const char *name);
CUDARTAPI cudaError_t oscLaunchKernel(oscFunction_t function,
                                      unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                                      unsigned int block_x, unsigned int block_y, unsigned int block_z,
                                      unsigned int shared_memory_bytes, void **arguments);

#ifdef __cplusplus
}
#endif

#endif
