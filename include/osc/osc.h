#ifndef OSCDA_OSC_H
#define OSCDA_OSC_H

#include <cuda_runtime_api.h>

#define OSCDA_VERSION_MAJOR 0
#define OSCDA_VERSION_MINOR 2
#define OSCDA_VERSION_PATCH 2

#ifdef __cplusplus
extern "C" {
#endif

typedef enum oscBackend {
    oscBackendCpu = 0,
    oscBackendAmdHip = 1
} oscBackend_t;

typedef struct osc_module *oscModule_t;
typedef struct osc_function *oscFunction_t;

CUDARTAPI const char *oscGetRuntimeName(void);
CUDARTAPI oscBackend_t oscGetBackend(void);
CUDARTAPI const char *oscGetBackendName(void);

/* Loads AMDGPU code objects accepted by HIP (normally .hsaco files). */
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
