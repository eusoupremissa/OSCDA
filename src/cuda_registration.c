#define _GNU_SOURCE

#include <osc/osc.h>

#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <link.h>
#endif

cudaError_t osc_result(cudaError_t error);

struct fatbin_wrapper {
    uint32_t magic, version;
    const unsigned char *image;
    const void *unused;
};

struct registered_fatbin {
    const unsigned char *image;
    size_t size;
    struct registered_fatbin *next;
};

struct registered_kernel {
    const void *host_stub;
    const char *name;
    struct registered_fatbin *owner;
    struct registered_kernel *next;
};

struct launch_config {
    dim3 grid, block;
    size_t shared;
    cudaStream_t stream;
    struct launch_config *previous;
};

static pthread_mutex_t registry_lock = PTHREAD_MUTEX_INITIALIZER;
static struct registered_fatbin *fatbins;
static struct registered_kernel *kernels;
static _Thread_local struct launch_config *launch_config;

#ifndef _WIN32
struct image_span {
    uintptr_t address;
    size_t size;
};

static int find_image_span(struct dl_phdr_info *object, size_t size, void *data)
{
    (void)size;
    struct image_span *span = data;
    for (size_t i = 0; i < object->dlpi_phnum; ++i) {
        const ElfW(Phdr) *segment = &object->dlpi_phdr[i];
        uintptr_t base = object->dlpi_addr + segment->p_vaddr;
        if (segment->p_type == PT_LOAD && (segment->p_flags & PF_R) &&
            span->address >= base && span->address - base < segment->p_filesz) {
            span->size = segment->p_filesz - (span->address - base);
            return 1;
        }
    }
    return 0;
}
#endif

static size_t mapped_image_size(const void *image)
{
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION region;
    if (!VirtualQuery(image, &region, sizeof(region)) || region.Type != MEM_IMAGE ||
        region.State != MEM_COMMIT || (region.Protect & (PAGE_GUARD | PAGE_NOACCESS)))
        return 0;
    return region.RegionSize - ((uintptr_t)image - (uintptr_t)region.BaseAddress);
#else
    struct image_span span = {(uintptr_t)image, 0};
    dl_iterate_phdr(find_image_span, &span);
    return span.size;
#endif
}

CUDARTAPI void **__cudaRegisterFatBinary(const struct fatbin_wrapper *wrapper)
{
    if (!wrapper || wrapper->magic != 0x466243b1 || wrapper->version != 1)
        return NULL;
    const unsigned char *image = wrapper->image;
    size_t available = mapped_image_size(image);
    if (available < 16 || memcmp(image, "\x50\xed\x55\xba\x01\x00\x10\x00", 8))
        return NULL;
    uint64_t body_size = 0;
    for (unsigned int i = 0; i < 8; ++i)
        body_size |= (uint64_t)image[8 + i] << (8 * i);
    if (!body_size || body_size > available - 16 || body_size > 64 * 1024 * 1024)
        return NULL;
    struct registered_fatbin *fatbin = calloc(1, sizeof(*fatbin));
    if (!fatbin)
        return NULL;
    fatbin->image = image;
    fatbin->size = 16 + (size_t)body_size;
    pthread_mutex_lock(&registry_lock);
    fatbin->next = fatbins;
    fatbins = fatbin;
    pthread_mutex_unlock(&registry_lock);
    return (void **)fatbin;
}

CUDARTAPI void __cudaRegisterFatBinaryEnd(void **handle)
{
    (void)handle;
}

CUDARTAPI void __cudaRegisterFunction(void **handle, const char *host_stub, char *device_function,
                                     const char *device_name, int thread_limit, void *thread_id,
                                     void *block_id, void *block_dim, void *grid_dim, int *warp_size)
{
    (void)thread_limit;
    (void)thread_id;
    (void)block_id;
    (void)block_dim;
    (void)grid_dim;
    (void)warp_size;
    const char *name = device_name ? device_name : device_function;
    if (!handle || !host_stub || !name)
        return;
    pthread_mutex_lock(&registry_lock);
    for (struct registered_fatbin *fatbin = fatbins; fatbin; fatbin = fatbin->next) {
        if ((void **)fatbin != handle)
            continue;
        struct registered_kernel *kernel = malloc(sizeof(*kernel));
        if (kernel) {
            *kernel = (struct registered_kernel){host_stub, name, fatbin, kernels};
            kernels = kernel;
        }
        break;
    }
    pthread_mutex_unlock(&registry_lock);
}

CUDARTAPI void __cudaUnregisterFatBinary(void **handle)
{
    pthread_mutex_lock(&registry_lock);
    struct registered_fatbin **owner = &fatbins;
    while (*owner && (void **)*owner != handle)
        owner = &(*owner)->next;
    if (*owner) {
        struct registered_fatbin *fatbin = *owner;
        *owner = fatbin->next;
        struct registered_kernel **kernel = &kernels;
        while (*kernel) {
            if ((*kernel)->owner == fatbin) {
                struct registered_kernel *removed = *kernel;
                *kernel = removed->next;
                free(removed);
            } else {
                kernel = &(*kernel)->next;
            }
        }
        free(fatbin);
    }
    pthread_mutex_unlock(&registry_lock);
}

cudaError_t cudaLaunchKernel(const void *function, dim3 grid, dim3 block, void **arguments,
                             size_t shared, cudaStream_t stream)
{
    if (!function || !grid.x || !grid.y || !grid.z || !block.x || !block.y || !block.z ||
        shared > UINT_MAX)
        return osc_result(cudaErrorInvalidValue);
    (void)arguments;
    (void)stream;
    return osc_result(cudaErrorNotSupported);
}

unsigned int __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shared, cudaStream_t stream)
{
    struct launch_config *config = malloc(sizeof(*config));
    if (!config)
        return osc_result(cudaErrorMemoryAllocation);
    *config = (struct launch_config){grid, block, shared, stream, launch_config};
    launch_config = config;
    return cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shared, void *stream)
{
    struct launch_config *config = launch_config;
    if (!config || !grid || !block || !shared || !stream)
        return osc_result(cudaErrorInvalidValue);
    *grid = config->grid;
    *block = config->block;
    *shared = config->shared;
    *(cudaStream_t *)stream = config->stream;
    launch_config = config->previous;
    free(config);
    return cudaSuccess;
}
