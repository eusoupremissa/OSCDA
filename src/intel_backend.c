#define _POSIX_C_SOURCE 200809L

#include "intel_backend.h"

#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Minimal stable Level Zero ABI used through the loader; no SDK headers required. */
typedef void *ze_driver_handle_t;
typedef void *ze_device_handle_t;
typedef void *ze_context_handle_t;
typedef void *ze_command_list_handle_t;
typedef void *ze_event_handle_t;
typedef uint32_t ze_result_t;

enum {
    ZE_RESULT_SUCCESS = 0,
    ZE_RESULT_NOT_READY = 1,
    ZE_RESULT_ERROR_DEVICE_LOST = 0x70000001u,
    ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY = 0x70000002u,
    ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 0x70000003u,
    ZE_INIT_FLAG_GPU_ONLY = 1,
    ZE_DEVICE_TYPE_GPU = 1,
    ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 1,
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = 1,
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 3,
    ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES = 4,
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 6,
    ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES = 7,
    ZE_STRUCTURE_TYPE_CONTEXT_DESC = 13,
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 14,
    ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 21,
    ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 22,
    ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 1,
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0
};

struct ze_device_properties {
    uint32_t stype;
    void *next;
    uint32_t type;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t flags;
    uint32_t subdevice_id;
    uint32_t core_clock_rate;
    uint64_t max_mem_alloc_size;
    uint32_t max_hardware_contexts;
    uint32_t max_command_queue_priority;
    uint32_t num_threads_per_eu;
    uint32_t physical_eu_simd_width;
    uint32_t num_eus_per_subslice;
    uint32_t num_subslices_per_slice;
    uint32_t num_slices;
    uint64_t timer_resolution;
    uint32_t timestamp_valid_bits;
    uint32_t kernel_timestamp_valid_bits;
    unsigned char uuid[16];
    char name[256];
};

struct ze_device_compute_properties {
    uint32_t stype;
    void *next;
    uint32_t max_total_group_size;
    uint32_t max_group_size_x;
    uint32_t max_group_size_y;
    uint32_t max_group_size_z;
    uint32_t max_group_count_x;
    uint32_t max_group_count_y;
    uint32_t max_group_count_z;
    uint32_t max_shared_local_memory;
    uint32_t num_subgroup_sizes;
    uint32_t subgroup_sizes[8];
};

struct ze_device_memory_properties {
    uint32_t stype;
    void *next;
    uint32_t flags;
    uint32_t max_clock_rate;
    uint32_t max_bus_width;
    uint64_t total_size;
    char name[256];
};

struct ze_command_queue_group_properties {
    uint32_t stype;
    void *next;
    uint32_t flags;
    size_t max_memory_fill_pattern_size;
    uint32_t num_queues;
};

struct ze_context_desc {
    uint32_t stype;
    const void *next;
    uint32_t flags;
};

struct ze_command_queue_desc {
    uint32_t stype;
    const void *next;
    uint32_t ordinal;
    uint32_t index;
    uint32_t flags;
    uint32_t mode;
    uint32_t priority;
};

struct ze_device_mem_alloc_desc {
    uint32_t stype;
    const void *next;
    uint32_t flags;
    uint32_t ordinal;
};

struct ze_host_mem_alloc_desc {
    uint32_t stype;
    const void *next;
    uint32_t flags;
};

_Static_assert(sizeof(struct ze_device_properties) == 368, "Level Zero device ABI mismatch");
_Static_assert(sizeof(struct ze_device_compute_properties) == 88, "Level Zero compute ABI mismatch");
_Static_assert(sizeof(struct ze_device_memory_properties) == 296, "Level Zero memory ABI mismatch");
_Static_assert(sizeof(struct ze_command_queue_group_properties) == 40, "Level Zero queue ABI mismatch");
_Static_assert(sizeof(struct ze_context_desc) == 24, "Level Zero context ABI mismatch");
_Static_assert(sizeof(struct ze_command_queue_desc) == 40, "Level Zero command ABI mismatch");
_Static_assert(sizeof(struct ze_device_mem_alloc_desc) == 24, "Level Zero allocation ABI mismatch");
_Static_assert(sizeof(struct ze_host_mem_alloc_desc) == 24, "Level Zero host ABI mismatch");

struct osc_level_zero_api {
    void *library;
    ze_result_t (*init)(uint32_t);
    ze_result_t (*driver_get)(uint32_t *, ze_driver_handle_t *);
    ze_result_t (*device_get)(ze_driver_handle_t, uint32_t *, ze_device_handle_t *);
    ze_result_t (*device_get_properties)(ze_device_handle_t, struct ze_device_properties *);
    ze_result_t (*device_get_compute_properties)(ze_device_handle_t,
                                                  struct ze_device_compute_properties *);
    ze_result_t (*device_get_memory_properties)(ze_device_handle_t, uint32_t *,
                                                 struct ze_device_memory_properties *);
    ze_result_t (*device_get_queue_properties)(ze_device_handle_t, uint32_t *,
                                                struct ze_command_queue_group_properties *);
    ze_result_t (*context_create)(ze_driver_handle_t, const struct ze_context_desc *,
                                  ze_context_handle_t *);
    ze_result_t (*context_destroy)(ze_context_handle_t);
    ze_result_t (*command_list_create_immediate)(ze_context_handle_t, ze_device_handle_t,
                                                  const struct ze_command_queue_desc *,
                                                  ze_command_list_handle_t *);
    ze_result_t (*command_list_destroy)(ze_command_list_handle_t);
    ze_result_t (*command_list_append_copy)(ze_command_list_handle_t, void *, const void *, size_t,
                                             ze_event_handle_t, uint32_t, ze_event_handle_t *);
    ze_result_t (*command_list_append_fill)(ze_command_list_handle_t, void *, const void *, size_t,
                                             size_t, ze_event_handle_t, uint32_t,
                                             ze_event_handle_t *);
    ze_result_t (*command_list_synchronize)(ze_command_list_handle_t, uint64_t);
    ze_result_t (*mem_alloc_shared)(ze_context_handle_t,
                                    const struct ze_device_mem_alloc_desc *,
                                    const struct ze_host_mem_alloc_desc *, size_t, size_t,
                                    ze_device_handle_t, void **);
    ze_result_t (*mem_alloc_device)(ze_context_handle_t,
                                    const struct ze_device_mem_alloc_desc *, size_t, size_t,
                                    ze_device_handle_t, void **);
    ze_result_t (*mem_free)(ze_context_handle_t, void *);
};

struct osc_intel_state {
    struct osc_level_zero_api api;
    ze_driver_handle_t driver;
    ze_device_handle_t device;
    ze_context_handle_t context;
    ze_command_list_handle_t command_list;
    struct ze_device_properties properties;
    struct ze_device_compute_properties compute;
    size_t physical_bytes;
    size_t budget_bytes;
    size_t allocated_bytes;
    unsigned int timeout_ms;
    int integrated;
    int available;
    int stalled;
};

static pthread_once_t osc_intel_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t osc_intel_lock = PTHREAD_MUTEX_INITIALIZER;
static struct osc_intel_state osc_intel;

static int osc_level_zero_symbol(void **target, const char *name)
{
    *target = dlsym(osc_intel.api.library, name);
    return *target != NULL;
}

#define OSC_ZE_LOAD(field, symbol) \
    do { if (!osc_level_zero_symbol((void **)&osc_intel.api.field, symbol)) goto unavailable; } while (0)

static size_t osc_size_from_env(const char *name, size_t fallback)
{
    const char *text = getenv(name);
    if (!text || !*text)
        return fallback;
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || value > SIZE_MAX)
        return fallback;
    return (size_t)value;
}

static size_t osc_available_host_memory(void)
{
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0 || (uint64_t)pages > SIZE_MAX / (uint64_t)page_size)
        return 0;
    return (size_t)pages * (size_t)page_size;
}

static size_t osc_default_budget(size_t physical, int integrated)
{
    if (!physical)
        return 0;
    size_t reserve = integrated ? physical / 4 : physical / 10;
    size_t minimum = (size_t)(integrated ? 512 : 256) * 1024 * 1024;
    if (reserve < minimum)
        reserve = minimum;
    size_t budget = reserve < physical ? physical - reserve : physical / 2;
    if (integrated) {
        size_t available = osc_available_host_memory();
        if (available && budget > available / 2)
            budget = available / 2;
    }
    return budget;
}

static void osc_intel_init_once(void)
{
    const char *library = getenv("OSC_LEVEL_ZERO_LIBRARY");
    osc_intel.api.library = dlopen(library && *library ? library : "libze_loader.so.1",
                                   RTLD_NOW | RTLD_LOCAL);
    if (!osc_intel.api.library)
        return;

    OSC_ZE_LOAD(init, "zeInit");
    OSC_ZE_LOAD(driver_get, "zeDriverGet");
    OSC_ZE_LOAD(device_get, "zeDeviceGet");
    OSC_ZE_LOAD(device_get_properties, "zeDeviceGetProperties");
    OSC_ZE_LOAD(device_get_compute_properties, "zeDeviceGetComputeProperties");
    OSC_ZE_LOAD(device_get_memory_properties, "zeDeviceGetMemoryProperties");
    OSC_ZE_LOAD(device_get_queue_properties, "zeDeviceGetCommandQueueGroupProperties");
    OSC_ZE_LOAD(context_create, "zeContextCreate");
    OSC_ZE_LOAD(context_destroy, "zeContextDestroy");
    OSC_ZE_LOAD(command_list_create_immediate, "zeCommandListCreateImmediate");
    OSC_ZE_LOAD(command_list_destroy, "zeCommandListDestroy");
    OSC_ZE_LOAD(command_list_append_copy, "zeCommandListAppendMemoryCopy");
    OSC_ZE_LOAD(command_list_append_fill, "zeCommandListAppendMemoryFill");
    OSC_ZE_LOAD(command_list_synchronize, "zeCommandListHostSynchronize");
    OSC_ZE_LOAD(mem_alloc_shared, "zeMemAllocShared");
    OSC_ZE_LOAD(mem_alloc_device, "zeMemAllocDevice");
    OSC_ZE_LOAD(mem_free, "zeMemFree");

    if (osc_intel.api.init(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS)
        goto unavailable;

    uint32_t driver_count = 0;
    if (osc_intel.api.driver_get(&driver_count, NULL) != ZE_RESULT_SUCCESS || !driver_count)
        goto unavailable;
    ze_driver_handle_t *drivers = calloc(driver_count, sizeof(*drivers));
    if (!drivers || osc_intel.api.driver_get(&driver_count, drivers) != ZE_RESULT_SUCCESS) {
        free(drivers);
        goto unavailable;
    }

    size_t requested = osc_size_from_env("OSC_INTEL_DEVICE", 0);
    size_t seen = 0;
    for (uint32_t i = 0; i < driver_count && !osc_intel.device; ++i) {
        uint32_t device_count = 0;
        if (osc_intel.api.device_get(drivers[i], &device_count, NULL) != ZE_RESULT_SUCCESS)
            continue;
        ze_device_handle_t *devices = calloc(device_count, sizeof(*devices));
        if (!devices || osc_intel.api.device_get(drivers[i], &device_count, devices) != ZE_RESULT_SUCCESS) {
            free(devices);
            continue;
        }
        for (uint32_t j = 0; j < device_count; ++j) {
            struct ze_device_properties properties = {0};
            properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (osc_intel.api.device_get_properties(devices[j], &properties) == ZE_RESULT_SUCCESS &&
                properties.type == ZE_DEVICE_TYPE_GPU && properties.vendor_id == 0x8086 &&
                seen++ == requested) {
                osc_intel.driver = drivers[i];
                osc_intel.device = devices[j];
                osc_intel.properties = properties;
                break;
            }
        }
        free(devices);
    }
    free(drivers);
    if (!osc_intel.device)
        goto unavailable;

    osc_intel.compute.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
    if (osc_intel.api.device_get_compute_properties(osc_intel.device, &osc_intel.compute) !=
        ZE_RESULT_SUCCESS)
        goto unavailable;

    uint32_t memory_count = 0;
    if (osc_intel.api.device_get_memory_properties(osc_intel.device, &memory_count, NULL) !=
        ZE_RESULT_SUCCESS || !memory_count)
        goto unavailable;
    struct ze_device_memory_properties *memory = calloc(memory_count, sizeof(*memory));
    if (!memory)
        goto unavailable;
    for (uint32_t i = 0; i < memory_count; ++i)
        memory[i].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    if (osc_intel.api.device_get_memory_properties(osc_intel.device, &memory_count, memory) !=
        ZE_RESULT_SUCCESS) {
        free(memory);
        goto unavailable;
    }
    for (uint32_t i = 0; i < memory_count; ++i) {
        if (memory[i].total_size > SIZE_MAX - osc_intel.physical_bytes) {
            osc_intel.physical_bytes = SIZE_MAX;
            break;
        }
        osc_intel.physical_bytes += (size_t)memory[i].total_size;
    }
    free(memory);

    uint32_t queue_count = 0;
    if (osc_intel.api.device_get_queue_properties(osc_intel.device, &queue_count, NULL) !=
        ZE_RESULT_SUCCESS || !queue_count)
        goto unavailable;
    struct ze_command_queue_group_properties *queues = calloc(queue_count, sizeof(*queues));
    if (!queues)
        goto unavailable;
    for (uint32_t i = 0; i < queue_count; ++i)
        queues[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    if (osc_intel.api.device_get_queue_properties(osc_intel.device, &queue_count, queues) !=
        ZE_RESULT_SUCCESS) {
        free(queues);
        goto unavailable;
    }
    uint32_t ordinal = UINT32_MAX;
    for (uint32_t i = 0; i < queue_count; ++i) {
        if (queues[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            ordinal = i;
            break;
        }
    }
    free(queues);
    if (ordinal == UINT32_MAX)
        goto unavailable;

    struct ze_context_desc context = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0};
    if (osc_intel.api.context_create(osc_intel.driver, &context, &osc_intel.context) !=
        ZE_RESULT_SUCCESS)
        goto unavailable;
    struct ze_command_queue_desc queue = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, ordinal, 0, 0,
        ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    if (osc_intel.api.command_list_create_immediate(osc_intel.context, osc_intel.device, &queue,
                                                     &osc_intel.command_list) != ZE_RESULT_SUCCESS)
        goto unavailable;

    osc_intel.integrated = !!(osc_intel.properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
    osc_intel.budget_bytes = osc_default_budget(osc_intel.physical_bytes, osc_intel.integrated);
    size_t budget_mb = osc_size_from_env("OSC_INTEL_MEMORY_BUDGET_MB", 0);
    if (budget_mb && budget_mb <= SIZE_MAX / (1024 * 1024)) {
        size_t requested_budget = budget_mb * 1024 * 1024;
        if (requested_budget <= osc_intel.physical_bytes)
            osc_intel.budget_bytes = requested_budget;
    }
    size_t timeout = osc_size_from_env("OSC_INTEL_TIMEOUT_MS", 30000);
    osc_intel.timeout_ms = timeout > UINT_MAX ? UINT_MAX : (unsigned int)timeout;
    osc_intel.available = osc_intel.budget_bytes > 0;
    if (osc_intel.available)
        return;

unavailable:
    if (osc_intel.command_list)
        osc_intel.api.command_list_destroy(osc_intel.command_list);
    if (osc_intel.context)
        osc_intel.api.context_destroy(osc_intel.context);
    if (osc_intel.api.library)
        dlclose(osc_intel.api.library);
    memset(&osc_intel, 0, sizeof(osc_intel));
}

#undef OSC_ZE_LOAD

static cudaError_t osc_from_level_zero(ze_result_t result)
{
    if (result == ZE_RESULT_SUCCESS)
        return cudaSuccess;
    if (result == ZE_RESULT_NOT_READY)
        return cudaErrorNotReady;
    if (result == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY ||
        result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
        return cudaErrorMemoryAllocation;
    if (result == ZE_RESULT_ERROR_DEVICE_LOST)
        return cudaErrorUnknown;
    return cudaErrorUnknown;
}

int osc_intel_available(void)
{
    pthread_once(&osc_intel_once, osc_intel_init_once);
    return osc_intel.available;
}

cudaError_t osc_intel_get_properties(cudaDeviceProp *properties)
{
    if (!osc_intel_available())
        return cudaErrorInitializationError;
    memset(properties, 0, sizeof(*properties));
    snprintf(properties->name, sizeof(properties->name), "%s", osc_intel.properties.name);
    properties->totalGlobalMem = osc_intel.budget_bytes;
    properties->sharedMemPerBlock = osc_intel.compute.max_shared_local_memory;
    properties->warpSize = osc_intel.compute.num_subgroup_sizes ?
        (int)osc_intel.compute.subgroup_sizes[0] : 1;
    properties->maxThreadsPerBlock = (int)osc_intel.compute.max_total_group_size;
    properties->maxThreadsDim[0] = (int)osc_intel.compute.max_group_size_x;
    properties->maxThreadsDim[1] = (int)osc_intel.compute.max_group_size_y;
    properties->maxThreadsDim[2] = (int)osc_intel.compute.max_group_size_z;
    properties->maxGridSize[0] = (int)osc_intel.compute.max_group_count_x;
    properties->maxGridSize[1] = (int)osc_intel.compute.max_group_count_y;
    properties->maxGridSize[2] = (int)osc_intel.compute.max_group_count_z;
    properties->clockRate = (int)osc_intel.properties.core_clock_rate * 1000;
    properties->major = 1;
    properties->minor = 0;
    properties->multiProcessorCount = (int)(osc_intel.properties.num_slices *
        osc_intel.properties.num_subslices_per_slice * osc_intel.properties.num_eus_per_subslice);
    properties->concurrentKernels = 1;
    properties->unifiedAddressing = 1;
    return cudaSuccess;
}

static cudaError_t osc_intel_wait_locked(void)
{
    if (osc_intel.stalled)
        return cudaErrorNotReady;
    uint64_t timeout_ns = (uint64_t)osc_intel.timeout_ms * 1000000u;
    ze_result_t result = osc_intel.api.command_list_synchronize(osc_intel.command_list, timeout_ns);
    if (result == ZE_RESULT_NOT_READY || result == ZE_RESULT_ERROR_DEVICE_LOST)
        osc_intel.stalled = 1;
    return osc_from_level_zero(result);
}

cudaError_t osc_intel_synchronize(void)
{
    if (!osc_intel_available())
        return cudaErrorInitializationError;
    pthread_mutex_lock(&osc_intel_lock);
    cudaError_t error = osc_intel_wait_locked();
    pthread_mutex_unlock(&osc_intel_lock);
    return error;
}

cudaError_t osc_intel_malloc(void **pointer, size_t size)
{
    if (!osc_intel_available())
        return cudaErrorInitializationError;
    pthread_mutex_lock(&osc_intel_lock);
    if (osc_intel.stalled || size > osc_intel.properties.max_mem_alloc_size ||
        size > osc_intel.budget_bytes - osc_intel.allocated_bytes) {
        pthread_mutex_unlock(&osc_intel_lock);
        return osc_intel.stalled ? cudaErrorNotReady : cudaErrorMemoryAllocation;
    }
    struct ze_device_mem_alloc_desc device = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, NULL, 0, 0
    };
    ze_result_t result;
    if (osc_intel.integrated) {
        struct ze_host_mem_alloc_desc host = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL, 0};
        result = osc_intel.api.mem_alloc_shared(osc_intel.context, &device, &host, size, 0,
                                                osc_intel.device, pointer);
    } else {
        result = osc_intel.api.mem_alloc_device(osc_intel.context, &device, size, 0,
                                                osc_intel.device, pointer);
    }
    if (result == ZE_RESULT_SUCCESS)
        osc_intel.allocated_bytes += size;
    pthread_mutex_unlock(&osc_intel_lock);
    return osc_from_level_zero(result);
}

cudaError_t osc_intel_free(void *pointer, size_t size)
{
    if (!osc_intel_available())
        return cudaErrorInitializationError;
    pthread_mutex_lock(&osc_intel_lock);
    ze_result_t result = osc_intel.api.mem_free(osc_intel.context, pointer);
    if (result == ZE_RESULT_SUCCESS)
        osc_intel.allocated_bytes -= size;
    pthread_mutex_unlock(&osc_intel_lock);
    return osc_from_level_zero(result);
}

cudaError_t osc_intel_memcpy(void *destination, const void *source, size_t size)
{
    if (!osc_intel_available())
        return cudaErrorInitializationError;
    pthread_mutex_lock(&osc_intel_lock);
    ze_result_t result = osc_intel.stalled ? ZE_RESULT_NOT_READY :
        osc_intel.api.command_list_append_copy(osc_intel.command_list, destination, source, size,
                                                NULL, 0, NULL);
    cudaError_t error = osc_from_level_zero(result);
    if (error == cudaSuccess)
        error = osc_intel_wait_locked();
    pthread_mutex_unlock(&osc_intel_lock);
    return error;
}

cudaError_t osc_intel_memset(void *destination, int value, size_t size)
{
    if (!osc_intel_available())
        return cudaErrorInitializationError;
    unsigned char pattern = (unsigned char)value;
    pthread_mutex_lock(&osc_intel_lock);
    ze_result_t result = osc_intel.stalled ? ZE_RESULT_NOT_READY :
        osc_intel.api.command_list_append_fill(osc_intel.command_list, destination, &pattern, 1,
                                                size, NULL, 0, NULL);
    cudaError_t error = osc_from_level_zero(result);
    if (error == cudaSuccess)
        error = osc_intel_wait_locked();
    pthread_mutex_unlock(&osc_intel_lock);
    return error;
}

cudaError_t osc_intel_get_memory_info(oscIntelMemoryInfo_t *info)
{
    if (!info)
        return cudaErrorInvalidValue;
    if (!osc_intel_available())
        return cudaErrorInitializationError;
    pthread_mutex_lock(&osc_intel_lock);
    info->physicalBytes = osc_intel.physical_bytes;
    info->budgetBytes = osc_intel.budget_bytes;
    info->allocatedBytes = osc_intel.allocated_bytes;
    info->integrated = (unsigned int)osc_intel.integrated;
    info->timeoutMs = osc_intel.timeout_ms;
    info->stalled = (unsigned int)osc_intel.stalled;
    pthread_mutex_unlock(&osc_intel_lock);
    return cudaSuccess;
}

cudaError_t osc_intel_set_memory_budget(size_t bytes)
{
    if (!osc_intel_available())
        return cudaErrorInitializationError;
    pthread_mutex_lock(&osc_intel_lock);
    cudaError_t result = cudaSuccess;
    if (bytes < osc_intel.allocated_bytes || bytes > osc_intel.physical_bytes)
        result = cudaErrorInvalidValue;
    else
        osc_intel.budget_bytes = bytes;
    pthread_mutex_unlock(&osc_intel_lock);
    return result;
}
