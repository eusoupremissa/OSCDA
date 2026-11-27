#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime_api.h>
#include <osc/osc.h>

#include "intel_backend.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

enum osc_backend_kind {
    OSC_BACKEND_CPU,
    OSC_BACKEND_INTEL_LEVEL_ZERO
};

static int osc_uses_intel(void)
{
    const char *requested = getenv("OSC_BACKEND");
    return (!requested || strcmp(requested, "cpu") != 0) && osc_intel_available();
}

typedef cudaError_t (*osc_job_fn)(void *);

struct osc_job {
    osc_job_fn run;
    void *data;
    struct osc_job *next;
};

struct osc_stream {
    pthread_t worker;
    pthread_mutex_t lock;
    pthread_cond_t wake;
    pthread_cond_t idle;
    struct osc_job *head;
    struct osc_job *tail;
    size_t pending;
    unsigned int flags;
    cudaError_t error;
    int closing;
    struct osc_stream *next;
};

struct osc_event {
    pthread_mutex_t lock;
    pthread_cond_t ready;
    struct timespec timestamp;
    unsigned long generation;
    unsigned long completed;
    unsigned int flags;
    int recorded;
    struct osc_event *next;
};

struct osc_allocation {
    void *base;
    size_t size;
    enum osc_backend_kind backend;
    struct osc_allocation *next;
};

struct osc_copy {
    void *dst;
    const void *src;
    size_t size;
};

struct osc_fill {
    void *dst;
    size_t size;
    int value;
};

struct osc_callback {
    cudaHostFn_t fn;
    void *data;
};

struct osc_event_record {
    cudaEvent_t event;
    unsigned long generation;
};

static pthread_mutex_t osc_runtime_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t osc_memory_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t osc_default_once = PTHREAD_ONCE_INIT;
static struct osc_stream osc_default_stream;
static struct osc_stream *osc_streams;
static struct osc_event *osc_events;
static struct osc_allocation *osc_allocations;
static int osc_default_status = cudaErrorInitializationError;
static _Thread_local cudaError_t osc_last_error;

cudaError_t osc_result(cudaError_t error)
{
    osc_last_error = error;
    return error;
}

static void *osc_worker(void *data)
{
    struct osc_stream *stream = data;

    pthread_mutex_lock(&stream->lock);
    for (;;) {
        while (!stream->head && !stream->closing)
            pthread_cond_wait(&stream->wake, &stream->lock);

        if (!stream->head && stream->closing)
            break;

        struct osc_job *job = stream->head;
        stream->head = job->next;
        if (!stream->head)
            stream->tail = NULL;

        pthread_mutex_unlock(&stream->lock);
        cudaError_t error = job->run(job->data);
        free(job);
        pthread_mutex_lock(&stream->lock);

        if (stream->error == cudaSuccess && error != cudaSuccess)
            stream->error = error;
        if (--stream->pending == 0)
            pthread_cond_broadcast(&stream->idle);
    }
    pthread_mutex_unlock(&stream->lock);
    return NULL;
}

static int osc_stream_init(struct osc_stream *stream, unsigned int flags)
{
    memset(stream, 0, sizeof(*stream));
    stream->flags = flags;
    if (pthread_mutex_init(&stream->lock, NULL) != 0)
        return 0;
    if (pthread_cond_init(&stream->wake, NULL) != 0) {
        pthread_mutex_destroy(&stream->lock);
        return 0;
    }
    if (pthread_cond_init(&stream->idle, NULL) != 0) {
        pthread_cond_destroy(&stream->wake);
        pthread_mutex_destroy(&stream->lock);
        return 0;
    }
    if (pthread_create(&stream->worker, NULL, osc_worker, stream) != 0) {
        pthread_cond_destroy(&stream->idle);
        pthread_cond_destroy(&stream->wake);
        pthread_mutex_destroy(&stream->lock);
        return 0;
    }
    return 1;
}

static void osc_default_init(void)
{
    if (osc_stream_init(&osc_default_stream, cudaStreamDefault))
        osc_default_status = cudaSuccess;
}

static struct osc_stream *osc_stream_find(cudaStream_t handle)
{
    pthread_once(&osc_default_once, osc_default_init);
    if (!handle)
        return osc_default_status == cudaSuccess ? &osc_default_stream : NULL;

    struct osc_stream *stream = osc_streams;
    while (stream && stream != handle)
        stream = stream->next;
    return stream;
}

static cudaError_t osc_submit(cudaStream_t handle, osc_job_fn run, void *data)
{
    struct osc_job *job = malloc(sizeof(*job));
    if (!job) {
        free(data);
        return cudaErrorMemoryAllocation;
    }
    job->run = run;
    job->data = data;
    job->next = NULL;

    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_stream *stream = osc_stream_find(handle);
    if (!stream) {
        pthread_mutex_unlock(&osc_runtime_lock);
        free(job);
        free(data);
        return cudaErrorInvalidResourceHandle;
    }

    pthread_mutex_lock(&stream->lock);
    pthread_mutex_unlock(&osc_runtime_lock);
    if (stream->closing) {
        pthread_mutex_unlock(&stream->lock);
        free(job);
        free(data);
        return cudaErrorInvalidResourceHandle;
    }
    if (stream->tail)
        stream->tail->next = job;
    else
        stream->head = job;
    stream->tail = job;
    ++stream->pending;
    pthread_cond_signal(&stream->wake);
    pthread_mutex_unlock(&stream->lock);
    return cudaSuccess;
}

static cudaError_t osc_stream_wait(struct osc_stream *stream)
{
    pthread_mutex_lock(&stream->lock);
    while (stream->pending)
        pthread_cond_wait(&stream->idle, &stream->lock);
    cudaError_t error = stream->error;
    stream->error = cudaSuccess;
    pthread_mutex_unlock(&stream->lock);
    return error;
}

static int osc_contains(const struct osc_allocation *allocation, const void *ptr, size_t size)
{
    uintptr_t base = (uintptr_t)allocation->base;
    uintptr_t address = (uintptr_t)ptr;
    return address >= base && address - base <= allocation->size &&
           size <= allocation->size - (address - base);
}

/* A tracked pointer with an invalid extent must never be treated as host memory. */
static int osc_device_range(const void *ptr, size_t size)
{
    int found = 0;
    pthread_mutex_lock(&osc_memory_lock);
    for (struct osc_allocation *allocation = osc_allocations; allocation; allocation = allocation->next) {
        if (osc_contains(allocation, ptr, size)) {
            found = 1;
            break;
        }
        if (osc_contains(allocation, ptr, 1))
            found = -1;
    }
    pthread_mutex_unlock(&osc_memory_lock);
    return found;
}

static enum osc_backend_kind osc_range_backend(const void *ptr, size_t size)
{
    enum osc_backend_kind backend = OSC_BACKEND_CPU;
    pthread_mutex_lock(&osc_memory_lock);
    for (struct osc_allocation *allocation = osc_allocations; allocation; allocation = allocation->next) {
        if (osc_contains(allocation, ptr, size)) {
            backend = allocation->backend;
            break;
        }
    }
    pthread_mutex_unlock(&osc_memory_lock);
    return backend;
}

static cudaError_t osc_validate_copy(void *dst, const void *src, size_t size, cudaMemcpyKind *kind)
{
    if (!dst || !src)
        return cudaErrorInvalidValue;
    if (!size)
        return cudaSuccess;

    int dst_device = osc_device_range(dst, size);
    int src_device = osc_device_range(src, size);
    if (dst_device < 0 || src_device < 0)
        return cudaErrorInvalidValue;
    if (*kind == cudaMemcpyDefault) {
        if (dst_device && src_device)
            *kind = cudaMemcpyDeviceToDevice;
        else if (dst_device)
            *kind = cudaMemcpyHostToDevice;
        else if (src_device)
            *kind = cudaMemcpyDeviceToHost;
        else
            *kind = cudaMemcpyHostToHost;
    }

    switch (*kind) {
    case cudaMemcpyHostToHost:
        return !dst_device && !src_device ? cudaSuccess : cudaErrorInvalidMemcpyDirection;
    case cudaMemcpyHostToDevice:
        return dst_device && !src_device ? cudaSuccess : cudaErrorInvalidMemcpyDirection;
    case cudaMemcpyDeviceToHost:
        return !dst_device && src_device ? cudaSuccess : cudaErrorInvalidMemcpyDirection;
    case cudaMemcpyDeviceToDevice:
        return dst_device && src_device ? cudaSuccess : cudaErrorInvalidMemcpyDirection;
    default:
        return cudaErrorInvalidMemcpyDirection;
    }
}

static cudaError_t osc_copy_run(void *data)
{
    struct osc_copy *copy = data;
    cudaError_t error = cudaSuccess;
    if (osc_range_backend(copy->dst, copy->size) == OSC_BACKEND_INTEL_LEVEL_ZERO ||
        osc_range_backend(copy->src, copy->size) == OSC_BACKEND_INTEL_LEVEL_ZERO)
        error = osc_intel_memcpy(copy->dst, copy->src, copy->size);
    else
        memmove(copy->dst, copy->src, copy->size);
    free(copy);
    return error;
}

static cudaError_t osc_fill_run(void *data)
{
    struct osc_fill *fill = data;
    cudaError_t error = cudaSuccess;
    if (osc_range_backend(fill->dst, fill->size) == OSC_BACKEND_INTEL_LEVEL_ZERO)
        error = osc_intel_memset(fill->dst, fill->value, fill->size);
    else
        memset(fill->dst, fill->value, fill->size);
    free(fill);
    return error;
}

static cudaError_t osc_callback_run(void *data)
{
    struct osc_callback *callback = data;
    callback->fn(callback->data);
    free(callback);
    return cudaSuccess;
}

static cudaError_t osc_event_record_run(void *data)
{
    struct osc_event_record *record = data;
    struct timespec now;
#ifdef _WIN32
    LARGE_INTEGER ticks, frequency;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&frequency);
    now.tv_sec = (time_t)(ticks.QuadPart / frequency.QuadPart);
    now.tv_nsec = (long)((ticks.QuadPart % frequency.QuadPart) * 1e9 / frequency.QuadPart);
#else
    clock_gettime(CLOCK_MONOTONIC, &now);
#endif

    pthread_mutex_lock(&record->event->lock);
    if (record->generation == record->event->generation)
        record->event->timestamp = now;
    if (record->generation > record->event->completed)
        record->event->completed = record->generation;
    pthread_cond_broadcast(&record->event->ready);
    pthread_mutex_unlock(&record->event->lock);
    free(record);
    return cudaSuccess;
}

const char *cudaGetErrorName(cudaError_t error)
{
    switch (error) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError: return "cudaErrorInitializationError";
    case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady: return "cudaErrorNotReady";
    case cudaErrorNotSupported: return "cudaErrorNotSupported";
    default: return "cudaErrorUnknown";
    }
}

const char *cudaGetErrorString(cudaError_t error)
{
    switch (error) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "memory allocation failed";
    case cudaErrorInitializationError: return "runtime initialization failed";
    case cudaErrorInvalidMemcpyDirection: return "invalid memory copy direction";
    case cudaErrorInvalidDevice: return "invalid device ordinal";
    case cudaErrorInvalidResourceHandle: return "invalid resource handle";
    case cudaErrorNotReady: return "asynchronous operation is not complete";
    case cudaErrorNotSupported: return "operation is not supported";
    default: return "unknown error";
    }
}

cudaError_t cudaGetLastError(void)
{
    cudaError_t error = osc_last_error;
    osc_last_error = cudaSuccess;
    return error;
}

cudaError_t cudaPeekAtLastError(void)
{
    return osc_last_error;
}

cudaError_t cudaGetDeviceCount(int *count)
{
    if (!count)
        return osc_result(cudaErrorInvalidValue);
    if (osc_uses_intel()) {
        *count = 1;
        return osc_result(cudaSuccess);
    }
    *count = 1;
    return osc_result(cudaSuccess);
}

cudaError_t cudaGetDevice(int *device)
{
    if (!device)
        return osc_result(cudaErrorInvalidValue);
    if (osc_uses_intel()) {
        *device = 0;
        return osc_result(cudaSuccess);
    }
    *device = 0;
    return osc_result(cudaSuccess);
}

cudaError_t cudaSetDevice(int device)
{
    if (osc_uses_intel())
        return osc_result(device == 0 ? cudaSuccess : cudaErrorInvalidDevice);
    return osc_result(device == 0 ? cudaSuccess : cudaErrorInvalidDevice);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
{
    if (!prop)
        return osc_result(cudaErrorInvalidValue);
    memset(prop, 0, sizeof(*prop));
    if (osc_uses_intel())
        return osc_result(device == 0 ? osc_intel_get_properties(prop) : cudaErrorInvalidDevice);
    if (device != 0)
        return osc_result(cudaErrorInvalidDevice);

    memcpy(prop->name, "OSC deterministic CPU", sizeof("OSC deterministic CPU"));
    prop->totalGlobalMem = (size_t)-1;
    prop->warpSize = 1;
    prop->maxThreadsPerBlock = 1;
    prop->maxThreadsDim[0] = 1;
    prop->maxThreadsDim[1] = 1;
    prop->maxThreadsDim[2] = 1;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->major = 0;
    prop->minor = 1;
#ifdef _WIN32
    long processors = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
#else
    long processors = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    prop->multiProcessorCount = processors > 0 ? (int)processors : 1;
    prop->concurrentKernels = 1;
    prop->unifiedAddressing = 1;
    return osc_result(cudaSuccess);
}

cudaError_t cudaDeviceSynchronize(void)
{
    pthread_once(&osc_default_once, osc_default_init);
    if (osc_default_status != cudaSuccess)
        return osc_result((cudaError_t)osc_default_status);

    cudaError_t error = osc_stream_wait(&osc_default_stream);
    pthread_mutex_lock(&osc_runtime_lock);
    for (struct osc_stream *stream = osc_streams; stream; stream = stream->next) {
        cudaError_t current = osc_stream_wait(stream);
        if (error == cudaSuccess)
            error = current;
    }
    pthread_mutex_unlock(&osc_runtime_lock);
    if (error == cudaSuccess && osc_uses_intel())
        error = osc_intel_synchronize();
    return osc_result(error);
}

cudaError_t cudaMalloc(void **ptr, size_t size)
{
    if (!ptr || !size)
        return osc_result(cudaErrorInvalidValue);

    void *memory = NULL;
    enum osc_backend_kind backend = osc_uses_intel() ? OSC_BACKEND_INTEL_LEVEL_ZERO : OSC_BACKEND_CPU;
    if (backend == OSC_BACKEND_INTEL_LEVEL_ZERO) {
        cudaError_t error = osc_intel_malloc(&memory, size);
        if (error != cudaSuccess) {
            *ptr = NULL;
            return osc_result(error);
        }
    } else {
        memory = malloc(size);
    }
    struct osc_allocation *allocation = malloc(sizeof(*allocation));
    if (!memory || !allocation) {
        if (memory && backend == OSC_BACKEND_INTEL_LEVEL_ZERO)
            osc_intel_free(memory, size);
        else
            free(memory);
        free(allocation);
        *ptr = NULL;
        return osc_result(cudaErrorMemoryAllocation);
    }

    allocation->base = memory;
    allocation->size = size;
    allocation->backend = backend;
    pthread_mutex_lock(&osc_memory_lock);
    allocation->next = osc_allocations;
    osc_allocations = allocation;
    pthread_mutex_unlock(&osc_memory_lock);
    *ptr = memory;
    return osc_result(cudaSuccess);
}

cudaError_t cudaFree(void *ptr)
{
    if (!ptr)
        return osc_result(cudaSuccess);

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        return error;

    pthread_mutex_lock(&osc_memory_lock);
    struct osc_allocation **cursor = &osc_allocations;
    while (*cursor && (*cursor)->base != ptr)
        cursor = &(*cursor)->next;
    if (!*cursor) {
        pthread_mutex_unlock(&osc_memory_lock);
        return osc_result(cudaErrorInvalidValue);
    }
    struct osc_allocation *allocation = *cursor;
    *cursor = allocation->next;
    pthread_mutex_unlock(&osc_memory_lock);

    if (allocation->backend == OSC_BACKEND_INTEL_LEVEL_ZERO)
        error = osc_intel_free(allocation->base, allocation->size);
    else
        free(allocation->base);
    free(allocation);
    return osc_result(error);
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    if (!ptr || !size)
        return osc_result(cudaErrorInvalidValue);
    *ptr = malloc(size);
    return osc_result(*ptr ? cudaSuccess : cudaErrorMemoryAllocation);
}

cudaError_t cudaFreeHost(void *ptr)
{
    free(ptr);
    return osc_result(cudaSuccess);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t error = osc_validate_copy(dst, src, count, &kind);
    if (error != cudaSuccess)
        return osc_result(error);
    error = cudaDeviceSynchronize();
    if (error == cudaSuccess && count) {
        if (osc_range_backend(dst, count) == OSC_BACKEND_INTEL_LEVEL_ZERO ||
            osc_range_backend(src, count) == OSC_BACKEND_INTEL_LEVEL_ZERO)
            error = osc_intel_memcpy(dst, src, count);
        else
            memmove(dst, src, count);
    }
    return osc_result(error);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t error = osc_validate_copy(dst, src, count, &kind);
    if (error != cudaSuccess)
        return osc_result(error);
    if (!count)
        return osc_result(cudaSuccess);

    struct osc_copy *copy = malloc(sizeof(*copy));
    if (!copy)
        return osc_result(cudaErrorMemoryAllocation);
    copy->dst = dst;
    copy->src = src;
    copy->size = count;
    return osc_result(osc_submit(stream, osc_copy_run, copy));
}

cudaError_t cudaMemset(void *dst, int value, size_t count)
{
    if (!dst || osc_device_range(dst, count) != 1)
        return osc_result(cudaErrorInvalidValue);
    cudaError_t error = cudaDeviceSynchronize();
    if (error == cudaSuccess && count) {
        if (osc_range_backend(dst, count) == OSC_BACKEND_INTEL_LEVEL_ZERO)
            error = osc_intel_memset(dst, value, count);
        else
            memset(dst, value, count);
    }
    return osc_result(error);
}

cudaError_t cudaMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream)
{
    if (!dst || osc_device_range(dst, count) != 1)
        return osc_result(cudaErrorInvalidValue);
    if (!count)
        return osc_result(cudaSuccess);

    struct osc_fill *fill = malloc(sizeof(*fill));
    if (!fill)
        return osc_result(cudaErrorMemoryAllocation);
    fill->dst = dst;
    fill->value = value;
    fill->size = count;
    return osc_result(osc_submit(stream, osc_fill_run, fill));
}

cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
    return cudaStreamCreateWithFlags(stream, cudaStreamDefault);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
{
    if (!stream || (flags & ~cudaStreamNonBlocking))
        return osc_result(cudaErrorInvalidValue);

    struct osc_stream *created = malloc(sizeof(*created));
    if (!created)
        return osc_result(cudaErrorMemoryAllocation);
    if (!osc_stream_init(created, flags)) {
        free(created);
        return osc_result(cudaErrorInitializationError);
    }

    pthread_mutex_lock(&osc_runtime_lock);
    created->next = osc_streams;
    osc_streams = created;
    pthread_mutex_unlock(&osc_runtime_lock);
    *stream = created;
    return osc_result(cudaSuccess);
}

cudaError_t cudaStreamDestroy(cudaStream_t handle)
{
    if (!handle)
        return osc_result(cudaErrorInvalidResourceHandle);

    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_stream **cursor = &osc_streams;
    while (*cursor && *cursor != handle)
        cursor = &(*cursor)->next;
    if (!*cursor) {
        pthread_mutex_unlock(&osc_runtime_lock);
        return osc_result(cudaErrorInvalidResourceHandle);
    }
    struct osc_stream *stream = *cursor;
    cudaError_t error = osc_stream_wait(stream);
    *cursor = stream->next;
    pthread_mutex_unlock(&osc_runtime_lock);

    pthread_mutex_lock(&stream->lock);
    stream->closing = 1;
    pthread_cond_signal(&stream->wake);
    pthread_mutex_unlock(&stream->lock);
    pthread_join(stream->worker, NULL);
    pthread_cond_destroy(&stream->idle);
    pthread_cond_destroy(&stream->wake);
    pthread_mutex_destroy(&stream->lock);
    free(stream);
    return osc_result(error);
}

cudaError_t cudaStreamSynchronize(cudaStream_t handle)
{
    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_stream *stream = osc_stream_find(handle);
    if (!stream) {
        pthread_mutex_unlock(&osc_runtime_lock);
        return osc_result(cudaErrorInvalidResourceHandle);
    }
    cudaError_t error = osc_stream_wait(stream);
    pthread_mutex_unlock(&osc_runtime_lock);
    return osc_result(error);
}

cudaError_t cudaStreamQuery(cudaStream_t handle)
{
    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_stream *stream = osc_stream_find(handle);
    if (!stream) {
        pthread_mutex_unlock(&osc_runtime_lock);
        return osc_result(cudaErrorInvalidResourceHandle);
    }
    pthread_mutex_lock(&stream->lock);
    cudaError_t error = stream->pending ? cudaErrorNotReady : stream->error;
    pthread_mutex_unlock(&stream->lock);
    pthread_mutex_unlock(&osc_runtime_lock);
    return osc_result(error);
}

cudaError_t cudaStreamGetFlags(cudaStream_t handle, unsigned int *flags)
{
    if (!flags)
        return osc_result(cudaErrorInvalidValue);
    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_stream *stream = osc_stream_find(handle);
    if (stream)
        *flags = stream->flags;
    pthread_mutex_unlock(&osc_runtime_lock);
    return osc_result(stream ? cudaSuccess : cudaErrorInvalidResourceHandle);
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
    if (!fn)
        return osc_result(cudaErrorInvalidValue);
    struct osc_callback *callback = malloc(sizeof(*callback));
    if (!callback)
        return osc_result(cudaErrorMemoryAllocation);
    callback->fn = fn;
    callback->data = userData;
    return osc_result(osc_submit(stream, osc_callback_run, callback));
}

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    if (!event || (flags & ~(cudaEventBlockingSync | cudaEventDisableTiming)))
        return osc_result(cudaErrorInvalidValue);

    struct osc_event *created = calloc(1, sizeof(*created));
    if (!created)
        return osc_result(cudaErrorMemoryAllocation);
    created->flags = flags;
    if (pthread_mutex_init(&created->lock, NULL) != 0) {
        free(created);
        return osc_result(cudaErrorInitializationError);
    }
    if (pthread_cond_init(&created->ready, NULL) != 0) {
        pthread_mutex_destroy(&created->lock);
        free(created);
        return osc_result(cudaErrorInitializationError);
    }

    pthread_mutex_lock(&osc_runtime_lock);
    created->next = osc_events;
    osc_events = created;
    pthread_mutex_unlock(&osc_runtime_lock);
    *event = created;
    return osc_result(cudaSuccess);
}

static struct osc_event *osc_event_find(cudaEvent_t handle)
{
    struct osc_event *event = osc_events;
    while (event && event != handle)
        event = event->next;
    return event;
}

cudaError_t cudaEventDestroy(cudaEvent_t handle)
{
    if (!handle)
        return osc_result(cudaErrorInvalidResourceHandle);
    cudaError_t error = cudaEventSynchronize(handle);
    if (error != cudaSuccess)
        return error;

    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_event **cursor = &osc_events;
    while (*cursor && *cursor != handle)
        cursor = &(*cursor)->next;
    if (!*cursor) {
        pthread_mutex_unlock(&osc_runtime_lock);
        return osc_result(cudaErrorInvalidResourceHandle);
    }
    struct osc_event *event = *cursor;
    *cursor = event->next;
    pthread_mutex_unlock(&osc_runtime_lock);

    pthread_cond_destroy(&event->ready);
    pthread_mutex_destroy(&event->lock);
    free(event);
    return osc_result(cudaSuccess);
}

cudaError_t cudaEventRecord(cudaEvent_t handle, cudaStream_t stream)
{
    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_event *event = osc_event_find(handle);
    pthread_mutex_unlock(&osc_runtime_lock);
    if (!event)
        return osc_result(cudaErrorInvalidResourceHandle);

    struct osc_event_record *record = malloc(sizeof(*record));
    if (!record)
        return osc_result(cudaErrorMemoryAllocation);
    pthread_mutex_lock(&event->lock);
    event->recorded = 1;
    record->generation = ++event->generation;
    pthread_mutex_unlock(&event->lock);
    record->event = event;

    cudaError_t error = osc_submit(stream, osc_event_record_run, record);
    if (error != cudaSuccess) {
        pthread_mutex_lock(&event->lock);
        event->completed = event->generation;
        pthread_cond_broadcast(&event->ready);
        pthread_mutex_unlock(&event->lock);
    }
    return osc_result(error);
}

cudaError_t cudaEventSynchronize(cudaEvent_t handle)
{
    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_event *event = osc_event_find(handle);
    if (!event) {
        pthread_mutex_unlock(&osc_runtime_lock);
        return osc_result(cudaErrorInvalidResourceHandle);
    }
    pthread_mutex_lock(&event->lock);
    pthread_mutex_unlock(&osc_runtime_lock);
    unsigned long target = event->generation;
    while (event->completed < target)
        pthread_cond_wait(&event->ready, &event->lock);
    pthread_mutex_unlock(&event->lock);
    return osc_result(cudaSuccess);
}

cudaError_t cudaEventQuery(cudaEvent_t handle)
{
    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_event *event = osc_event_find(handle);
    if (!event) {
        pthread_mutex_unlock(&osc_runtime_lock);
        return osc_result(cudaErrorInvalidResourceHandle);
    }
    pthread_mutex_lock(&event->lock);
    cudaError_t error = event->completed < event->generation ? cudaErrorNotReady : cudaSuccess;
    pthread_mutex_unlock(&event->lock);
    pthread_mutex_unlock(&osc_runtime_lock);
    return osc_result(error);
}

cudaError_t cudaEventElapsedTime(float *milliseconds, cudaEvent_t start_handle, cudaEvent_t end_handle)
{
    if (!milliseconds)
        return osc_result(cudaErrorInvalidValue);

    pthread_mutex_lock(&osc_runtime_lock);
    struct osc_event *start = osc_event_find(start_handle);
    struct osc_event *end = osc_event_find(end_handle);
    if (!start || !end) {
        pthread_mutex_unlock(&osc_runtime_lock);
        return osc_result(cudaErrorInvalidResourceHandle);
    }

    pthread_mutex_lock(&start->lock);
    int start_recorded = start->recorded;
    int start_ready = start->completed >= start->generation;
    unsigned int start_flags = start->flags;
    struct timespec start_time = start->timestamp;
    pthread_mutex_unlock(&start->lock);

    pthread_mutex_lock(&end->lock);
    int end_recorded = end->recorded;
    int end_ready = end->completed >= end->generation;
    unsigned int end_flags = end->flags;
    struct timespec end_time = end->timestamp;
    pthread_mutex_unlock(&end->lock);
    pthread_mutex_unlock(&osc_runtime_lock);

    if (!start_recorded || !end_recorded || !start_ready || !end_ready)
        return osc_result(cudaErrorNotReady);
    if ((start_flags | end_flags) & cudaEventDisableTiming)
        return osc_result(cudaErrorInvalidResourceHandle);

    time_t seconds = end_time.tv_sec - start_time.tv_sec;
    long nanoseconds = end_time.tv_nsec - start_time.tv_nsec;
    *milliseconds = (float)seconds * 1000.0f + (float)nanoseconds / 1000000.0f;
    return osc_result(cudaSuccess);
}

const char *oscGetRuntimeName(void)
{
    return "OSCDA";
}

oscBackend_t oscGetBackend(void)
{
    return osc_uses_intel() ? oscBackendIntelLevelZero : oscBackendCpu;
}

const char *oscGetBackendName(void)
{
    return osc_uses_intel() ? "Intel Level Zero" : "CPU";
}

cudaError_t oscIntelGetMemoryInfo(oscIntelMemoryInfo_t *info)
{
    return osc_result(osc_intel_get_memory_info(info));
}

cudaError_t oscIntelSetMemoryBudget(size_t bytes)
{
    return osc_result(osc_intel_set_memory_budget(bytes));
}

cudaError_t oscModuleLoad(oscModule_t *module, const char *path)
{
    if (!module || !path)
        return osc_result(cudaErrorInvalidValue);
    return osc_result(cudaErrorNotSupported);
}

cudaError_t oscModuleUnload(oscModule_t module)
{
    if (!module)
        return osc_result(cudaErrorInvalidValue);
    return osc_result(cudaErrorNotSupported);
}

cudaError_t oscModuleGetFunction(oscFunction_t *function, oscModule_t module, const char *name)
{
    if (!function || !module || !name)
        return osc_result(cudaErrorInvalidValue);
    return osc_result(cudaErrorNotSupported);
}

cudaError_t oscLaunchKernel(oscFunction_t function,
                            unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                            unsigned int block_x, unsigned int block_y, unsigned int block_z,
                            unsigned int shared_memory_bytes, void **arguments)
{
    if (!function || !grid_x || !grid_y || !grid_z || !block_x || !block_y || !block_z)
        return osc_result(cudaErrorInvalidValue);
    (void)shared_memory_bytes;
    (void)arguments;
    return osc_result(cudaErrorNotSupported);
}
