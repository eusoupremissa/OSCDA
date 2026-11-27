#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef void *handle_t;

struct device_properties {
    uint32_t stype;
    void *next;
    uint32_t type, vendor_id, device_id, flags, subdevice_id, core_clock_rate;
    uint64_t max_mem_alloc_size;
    uint32_t max_hardware_contexts, max_command_queue_priority, num_threads_per_eu;
    uint32_t physical_eu_simd_width, num_eus_per_subslice, num_subslices_per_slice, num_slices;
    uint64_t timer_resolution;
    uint32_t timestamp_valid_bits, kernel_timestamp_valid_bits;
    unsigned char uuid[16];
    char name[256];
};

struct compute_properties {
    uint32_t stype;
    void *next;
    uint32_t max_total_group_size, max_group_size_x, max_group_size_y, max_group_size_z;
    uint32_t max_group_count_x, max_group_count_y, max_group_count_z;
    uint32_t max_shared_local_memory, num_subgroup_sizes, subgroup_sizes[8];
};

struct memory_properties {
    uint32_t stype;
    void *next;
    uint32_t flags, max_clock_rate, max_bus_width;
    uint64_t total_size;
    char name[256];
};

struct queue_properties {
    uint32_t stype;
    void *next;
    uint32_t flags;
    size_t max_memory_fill_pattern_size;
    uint32_t num_queues;
};

uint32_t zeInit(uint32_t flags) { return flags == 1 ? 0 : 1; }

uint32_t zeDriverGet(uint32_t *count, handle_t *drivers)
{
    if (!drivers) *count = 1;
    else drivers[0] = (handle_t)1;
    return 0;
}

uint32_t zeDeviceGet(handle_t driver, uint32_t *count, handle_t *devices)
{
    (void)driver;
    if (!devices) *count = 1;
    else devices[0] = (handle_t)2;
    return 0;
}

uint32_t zeDeviceGetProperties(handle_t device, struct device_properties *properties)
{
    (void)device;
    properties->type = 1;
    properties->vendor_id = 0x8086;
    properties->device_id = 0x1234;
    properties->flags = getenv("MOCK_ZE_DISCRETE") ? 0 : 1;
    properties->core_clock_rate = 1500;
    properties->max_mem_alloc_size = 512u * 1024 * 1024;
    properties->num_threads_per_eu = 8;
    properties->physical_eu_simd_width = 16;
    properties->num_eus_per_subslice = 8;
    properties->num_subslices_per_slice = 4;
    properties->num_slices = 1;
    strcpy(properties->name, "Mock Intel integrated GPU");
    return 0;
}

uint32_t zeDeviceGetComputeProperties(handle_t device, struct compute_properties *properties)
{
    (void)device;
    properties->max_total_group_size = 1024;
    properties->max_group_size_x = 1024;
    properties->max_group_size_y = 1024;
    properties->max_group_size_z = 64;
    properties->max_group_count_x = 65535;
    properties->max_group_count_y = 65535;
    properties->max_group_count_z = 65535;
    properties->max_shared_local_memory = 65536;
    properties->num_subgroup_sizes = 1;
    properties->subgroup_sizes[0] = 16;
    return 0;
}

uint32_t zeDeviceGetMemoryProperties(handle_t device, uint32_t *count,
                                     struct memory_properties *properties)
{
    (void)device;
    if (!properties) {
        *count = 1;
    } else {
        properties[0].total_size = 1024u * 1024 * 1024;
        strcpy(properties[0].name, "mock shared memory");
    }
    return 0;
}

uint32_t zeDeviceGetCommandQueueGroupProperties(handle_t device, uint32_t *count,
                                                struct queue_properties *properties)
{
    (void)device;
    if (!properties) {
        *count = 1;
    } else {
        properties[0].flags = 1;
        properties[0].max_memory_fill_pattern_size = 8;
        properties[0].num_queues = 1;
    }
    return 0;
}

uint32_t zeContextCreate(handle_t driver, const void *description, handle_t *context)
{
    (void)driver; (void)description; *context = (handle_t)3; return 0;
}
uint32_t zeContextDestroy(handle_t context) { (void)context; return 0; }
uint32_t zeCommandListCreateImmediate(handle_t context, handle_t device,
                                      const void *description, handle_t *list)
{
    (void)context; (void)device; (void)description; *list = (handle_t)4; return 0;
}
uint32_t zeCommandListDestroy(handle_t list) { (void)list; return 0; }
uint32_t zeCommandListHostSynchronize(handle_t list, uint64_t timeout)
{
    (void)list; (void)timeout; return getenv("MOCK_ZE_TIMEOUT") ? 1 : 0;
}
uint32_t zeCommandListAppendMemoryCopy(handle_t list, void *destination, const void *source,
                                       size_t size, handle_t event, uint32_t wait_count,
                                       handle_t *wait_events)
{
    (void)list; (void)event; (void)wait_count; (void)wait_events;
    memcpy(destination, source, size);
    return 0;
}
uint32_t zeCommandListAppendMemoryFill(handle_t list, void *destination, const void *pattern,
                                       size_t pattern_size, size_t size, handle_t event,
                                       uint32_t wait_count, handle_t *wait_events)
{
    (void)list; (void)event; (void)wait_count; (void)wait_events;
    if (pattern_size != 1) return 1;
    memset(destination, *(const unsigned char *)pattern, size);
    return 0;
}
uint32_t zeMemAllocShared(handle_t context, const void *device_description,
                          const void *host_description, size_t size, size_t alignment,
                          handle_t device, void **pointer)
{
    (void)context; (void)device_description; (void)host_description;
    (void)alignment; (void)device;
    if (getenv("MOCK_ZE_DISCRETE")) return 0x70000003u;
    *pointer = malloc(size); return *pointer ? 0 : 0x70000002u;
}
uint32_t zeMemAllocDevice(handle_t context, const void *description, size_t size,
                          size_t alignment, handle_t device, void **pointer)
{
    (void)context; (void)description; (void)alignment; (void)device;
    *pointer = malloc(size); return *pointer ? 0 : 0x70000003u;
}
uint32_t zeMemFree(handle_t context, void *pointer)
{
    (void)context; free(pointer); return 0;
}
