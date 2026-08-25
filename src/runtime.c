#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime_api.h>

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef cudaError_t (*uw_job_fn)(void *);

struct uw_job {
    uw_job_fn run;
    void *data;
    struct uw_job *next;
};

struct uw_stream {
    pthread_t worker;
    pthread_mutex_t lock;
    pthread_cond_t wake;
    pthread_cond_t idle;
    struct uw_job *head;
    struct uw_job *tail;
    size_t pending;
    unsigned int flags;
    cudaError_t error;
    int closing;
    struct uw_stream *next;
};

struct uw_event {
    pthread_mutex_t lock;
    pthread_cond_t ready;
    struct timespec timestamp;
    unsigned long generation;
    unsigned long completed;
    unsigned int flags;
    int recorded;
    struct uw_event *next;
};

struct uw_allocation {
    void *base;
    size_t size;
    struct uw_allocation *next;
};

struct uw_copy {
    void *dst;
    const void *src;
    size_t size;
};

struct uw_fill {
    void *dst;
    size_t size;
    int value;
};

struct uw_callback {
    cudaHostFn_t fn;
    void *data;
};

struct uw_event_record {
    cudaEvent_t event;
    unsigned long generation;
};

static pthread_mutex_t uw_runtime_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t uw_memory_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t uw_default_once = PTHREAD_ONCE_INIT;
static struct uw_stream uw_default_stream;
static struct uw_stream *uw_streams;
static struct uw_event *uw_events;
static struct uw_allocation *uw_allocations;
static int uw_default_status = cudaErrorInitializationError;
static _Thread_local cudaError_t uw_last_error;

static cudaError_t uw_result(cudaError_t error)
{
    uw_last_error = error;
    return error;
}

static void *uw_worker(void *data)
{
    struct uw_stream *stream = data;

    pthread_mutex_lock(&stream->lock);
    for (;;) {
        while (!stream->head && !stream->closing)
            pthread_cond_wait(&stream->wake, &stream->lock);

        if (!stream->head && stream->closing)
            break;

        struct uw_job *job = stream->head;
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

static int uw_stream_init(struct uw_stream *stream, unsigned int flags)
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
    if (pthread_create(&stream->worker, NULL, uw_worker, stream) != 0) {
        pthread_cond_destroy(&stream->idle);
        pthread_cond_destroy(&stream->wake);
        pthread_mutex_destroy(&stream->lock);
        return 0;
    }
    return 1;
}

static void uw_default_init(void)
{
    if (uw_stream_init(&uw_default_stream, cudaStreamDefault))
        uw_default_status = cudaSuccess;
}

static struct uw_stream *uw_stream_find(cudaStream_t handle)
{
    pthread_once(&uw_default_once, uw_default_init);
    if (!handle)
        return uw_default_status == cudaSuccess ? &uw_default_stream : NULL;

    struct uw_stream *stream = uw_streams;
    while (stream && stream != handle)
        stream = stream->next;
    return stream;
}

static cudaError_t uw_submit(cudaStream_t handle, uw_job_fn run, void *data)
{
    struct uw_job *job = malloc(sizeof(*job));
    if (!job) {
        free(data);
        return cudaErrorMemoryAllocation;
    }
    job->run = run;
    job->data = data;
    job->next = NULL;

    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_stream *stream = uw_stream_find(handle);
    if (!stream) {
        pthread_mutex_unlock(&uw_runtime_lock);
        free(job);
        free(data);
        return cudaErrorInvalidResourceHandle;
    }

    pthread_mutex_lock(&stream->lock);
    pthread_mutex_unlock(&uw_runtime_lock);
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

static cudaError_t uw_stream_wait(struct uw_stream *stream)
{
    pthread_mutex_lock(&stream->lock);
    while (stream->pending)
        pthread_cond_wait(&stream->idle, &stream->lock);
    cudaError_t error = stream->error;
    stream->error = cudaSuccess;
    pthread_mutex_unlock(&stream->lock);
    return error;
}

static int uw_contains(const struct uw_allocation *allocation, const void *ptr, size_t size)
{
    uintptr_t base = (uintptr_t)allocation->base;
    uintptr_t address = (uintptr_t)ptr;
    return address >= base && address - base <= allocation->size &&
           size <= allocation->size - (address - base);
}

static int uw_is_device_range(const void *ptr, size_t size)
{
    int found = 0;
    pthread_mutex_lock(&uw_memory_lock);
    for (struct uw_allocation *allocation = uw_allocations; allocation; allocation = allocation->next) {
        if (uw_contains(allocation, ptr, size)) {
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&uw_memory_lock);
    return found;
}

static cudaError_t uw_validate_copy(void *dst, const void *src, size_t size, cudaMemcpyKind *kind)
{
    if (!dst || !src)
        return cudaErrorInvalidValue;
    if (!size)
        return cudaSuccess;

    int dst_device = uw_is_device_range(dst, size);
    int src_device = uw_is_device_range(src, size);
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

static cudaError_t uw_copy_run(void *data)
{
    struct uw_copy *copy = data;
    memmove(copy->dst, copy->src, copy->size);
    free(copy);
    return cudaSuccess;
}

static cudaError_t uw_fill_run(void *data)
{
    struct uw_fill *fill = data;
    memset(fill->dst, fill->value, fill->size);
    free(fill);
    return cudaSuccess;
}

static cudaError_t uw_callback_run(void *data)
{
    struct uw_callback *callback = data;
    callback->fn(callback->data);
    free(callback);
    return cudaSuccess;
}

static cudaError_t uw_event_record_run(void *data)
{
    struct uw_event_record *record = data;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

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
    cudaError_t error = uw_last_error;
    uw_last_error = cudaSuccess;
    return error;
}

cudaError_t cudaPeekAtLastError(void)
{
    return uw_last_error;
}

cudaError_t cudaGetDeviceCount(int *count)
{
    if (!count)
        return uw_result(cudaErrorInvalidValue);
    *count = 1;
    return uw_result(cudaSuccess);
}

cudaError_t cudaGetDevice(int *device)
{
    if (!device)
        return uw_result(cudaErrorInvalidValue);
    *device = 0;
    return uw_result(cudaSuccess);
}

cudaError_t cudaSetDevice(int device)
{
    return uw_result(device == 0 ? cudaSuccess : cudaErrorInvalidDevice);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
{
    if (!prop)
        return uw_result(cudaErrorInvalidValue);
    if (device != 0)
        return uw_result(cudaErrorInvalidDevice);

    memset(prop, 0, sizeof(*prop));
    memcpy(prop->name, "UniWarp deterministic CPU", sizeof("UniWarp deterministic CPU"));
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
    long processors = sysconf(_SC_NPROCESSORS_ONLN);
    prop->multiProcessorCount = processors > 0 ? (int)processors : 1;
    prop->concurrentKernels = 1;
    prop->unifiedAddressing = 1;
    return uw_result(cudaSuccess);
}

cudaError_t cudaDeviceSynchronize(void)
{
    pthread_once(&uw_default_once, uw_default_init);
    if (uw_default_status != cudaSuccess)
        return uw_result((cudaError_t)uw_default_status);

    cudaError_t error = uw_stream_wait(&uw_default_stream);
    pthread_mutex_lock(&uw_runtime_lock);
    for (struct uw_stream *stream = uw_streams; stream; stream = stream->next) {
        cudaError_t current = uw_stream_wait(stream);
        if (error == cudaSuccess)
            error = current;
    }
    pthread_mutex_unlock(&uw_runtime_lock);
    return uw_result(error);
}

cudaError_t cudaMalloc(void **ptr, size_t size)
{
    if (!ptr || !size)
        return uw_result(cudaErrorInvalidValue);

    void *memory = malloc(size);
    struct uw_allocation *allocation = malloc(sizeof(*allocation));
    if (!memory || !allocation) {
        free(memory);
        free(allocation);
        *ptr = NULL;
        return uw_result(cudaErrorMemoryAllocation);
    }

    allocation->base = memory;
    allocation->size = size;
    pthread_mutex_lock(&uw_memory_lock);
    allocation->next = uw_allocations;
    uw_allocations = allocation;
    pthread_mutex_unlock(&uw_memory_lock);
    *ptr = memory;
    return uw_result(cudaSuccess);
}

cudaError_t cudaFree(void *ptr)
{
    if (!ptr)
        return uw_result(cudaSuccess);

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        return error;

    pthread_mutex_lock(&uw_memory_lock);
    struct uw_allocation **cursor = &uw_allocations;
    while (*cursor && (*cursor)->base != ptr)
        cursor = &(*cursor)->next;
    if (!*cursor) {
        pthread_mutex_unlock(&uw_memory_lock);
        return uw_result(cudaErrorInvalidValue);
    }
    struct uw_allocation *allocation = *cursor;
    *cursor = allocation->next;
    pthread_mutex_unlock(&uw_memory_lock);

    free(allocation->base);
    free(allocation);
    return uw_result(cudaSuccess);
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    if (!ptr || !size)
        return uw_result(cudaErrorInvalidValue);
    *ptr = malloc(size);
    return uw_result(*ptr ? cudaSuccess : cudaErrorMemoryAllocation);
}

cudaError_t cudaFreeHost(void *ptr)
{
    free(ptr);
    return uw_result(cudaSuccess);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t error = uw_validate_copy(dst, src, count, &kind);
    if (error != cudaSuccess)
        return uw_result(error);
    error = cudaDeviceSynchronize();
    if (error == cudaSuccess && count)
        memmove(dst, src, count);
    return uw_result(error);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t error = uw_validate_copy(dst, src, count, &kind);
    if (error != cudaSuccess)
        return uw_result(error);
    if (!count)
        return uw_result(cudaSuccess);

    struct uw_copy *copy = malloc(sizeof(*copy));
    if (!copy)
        return uw_result(cudaErrorMemoryAllocation);
    copy->dst = dst;
    copy->src = src;
    copy->size = count;
    return uw_result(uw_submit(stream, uw_copy_run, copy));
}

cudaError_t cudaMemset(void *dst, int value, size_t count)
{
    if (!dst || !uw_is_device_range(dst, count))
        return uw_result(cudaErrorInvalidValue);
    cudaError_t error = cudaDeviceSynchronize();
    if (error == cudaSuccess && count)
        memset(dst, value, count);
    return uw_result(error);
}

cudaError_t cudaMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream)
{
    if (!dst || !uw_is_device_range(dst, count))
        return uw_result(cudaErrorInvalidValue);
    if (!count)
        return uw_result(cudaSuccess);

    struct uw_fill *fill = malloc(sizeof(*fill));
    if (!fill)
        return uw_result(cudaErrorMemoryAllocation);
    fill->dst = dst;
    fill->value = value;
    fill->size = count;
    return uw_result(uw_submit(stream, uw_fill_run, fill));
}

cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
    return cudaStreamCreateWithFlags(stream, cudaStreamDefault);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
{
    if (!stream || (flags & ~cudaStreamNonBlocking))
        return uw_result(cudaErrorInvalidValue);

    struct uw_stream *created = malloc(sizeof(*created));
    if (!created)
        return uw_result(cudaErrorMemoryAllocation);
    if (!uw_stream_init(created, flags)) {
        free(created);
        return uw_result(cudaErrorInitializationError);
    }

    pthread_mutex_lock(&uw_runtime_lock);
    created->next = uw_streams;
    uw_streams = created;
    pthread_mutex_unlock(&uw_runtime_lock);
    *stream = created;
    return uw_result(cudaSuccess);
}

cudaError_t cudaStreamDestroy(cudaStream_t handle)
{
    if (!handle)
        return uw_result(cudaErrorInvalidResourceHandle);

    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_stream **cursor = &uw_streams;
    while (*cursor && *cursor != handle)
        cursor = &(*cursor)->next;
    if (!*cursor) {
        pthread_mutex_unlock(&uw_runtime_lock);
        return uw_result(cudaErrorInvalidResourceHandle);
    }
    struct uw_stream *stream = *cursor;
    cudaError_t error = uw_stream_wait(stream);
    *cursor = stream->next;
    pthread_mutex_unlock(&uw_runtime_lock);

    pthread_mutex_lock(&stream->lock);
    stream->closing = 1;
    pthread_cond_signal(&stream->wake);
    pthread_mutex_unlock(&stream->lock);
    pthread_join(stream->worker, NULL);
    pthread_cond_destroy(&stream->idle);
    pthread_cond_destroy(&stream->wake);
    pthread_mutex_destroy(&stream->lock);
    free(stream);
    return uw_result(error);
}

cudaError_t cudaStreamSynchronize(cudaStream_t handle)
{
    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_stream *stream = uw_stream_find(handle);
    if (!stream) {
        pthread_mutex_unlock(&uw_runtime_lock);
        return uw_result(cudaErrorInvalidResourceHandle);
    }
    cudaError_t error = uw_stream_wait(stream);
    pthread_mutex_unlock(&uw_runtime_lock);
    return uw_result(error);
}

cudaError_t cudaStreamQuery(cudaStream_t handle)
{
    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_stream *stream = uw_stream_find(handle);
    if (!stream) {
        pthread_mutex_unlock(&uw_runtime_lock);
        return uw_result(cudaErrorInvalidResourceHandle);
    }
    pthread_mutex_lock(&stream->lock);
    cudaError_t error = stream->pending ? cudaErrorNotReady : stream->error;
    pthread_mutex_unlock(&stream->lock);
    pthread_mutex_unlock(&uw_runtime_lock);
    return uw_result(error);
}

cudaError_t cudaStreamGetFlags(cudaStream_t handle, unsigned int *flags)
{
    if (!flags)
        return uw_result(cudaErrorInvalidValue);
    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_stream *stream = uw_stream_find(handle);
    if (stream)
        *flags = stream->flags;
    pthread_mutex_unlock(&uw_runtime_lock);
    return uw_result(stream ? cudaSuccess : cudaErrorInvalidResourceHandle);
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
    if (!fn)
        return uw_result(cudaErrorInvalidValue);
    struct uw_callback *callback = malloc(sizeof(*callback));
    if (!callback)
        return uw_result(cudaErrorMemoryAllocation);
    callback->fn = fn;
    callback->data = userData;
    return uw_result(uw_submit(stream, uw_callback_run, callback));
}

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    if (!event || (flags & ~(cudaEventBlockingSync | cudaEventDisableTiming)))
        return uw_result(cudaErrorInvalidValue);

    struct uw_event *created = calloc(1, sizeof(*created));
    if (!created)
        return uw_result(cudaErrorMemoryAllocation);
    created->flags = flags;
    if (pthread_mutex_init(&created->lock, NULL) != 0) {
        free(created);
        return uw_result(cudaErrorInitializationError);
    }
    if (pthread_cond_init(&created->ready, NULL) != 0) {
        pthread_mutex_destroy(&created->lock);
        free(created);
        return uw_result(cudaErrorInitializationError);
    }

    pthread_mutex_lock(&uw_runtime_lock);
    created->next = uw_events;
    uw_events = created;
    pthread_mutex_unlock(&uw_runtime_lock);
    *event = created;
    return uw_result(cudaSuccess);
}

static struct uw_event *uw_event_find(cudaEvent_t handle)
{
    struct uw_event *event = uw_events;
    while (event && event != handle)
        event = event->next;
    return event;
}

cudaError_t cudaEventDestroy(cudaEvent_t handle)
{
    if (!handle)
        return uw_result(cudaErrorInvalidResourceHandle);
    cudaError_t error = cudaEventSynchronize(handle);
    if (error != cudaSuccess)
        return error;

    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_event **cursor = &uw_events;
    while (*cursor && *cursor != handle)
        cursor = &(*cursor)->next;
    if (!*cursor) {
        pthread_mutex_unlock(&uw_runtime_lock);
        return uw_result(cudaErrorInvalidResourceHandle);
    }
    struct uw_event *event = *cursor;
    *cursor = event->next;
    pthread_mutex_unlock(&uw_runtime_lock);

    pthread_cond_destroy(&event->ready);
    pthread_mutex_destroy(&event->lock);
    free(event);
    return uw_result(cudaSuccess);
}

cudaError_t cudaEventRecord(cudaEvent_t handle, cudaStream_t stream)
{
    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_event *event = uw_event_find(handle);
    pthread_mutex_unlock(&uw_runtime_lock);
    if (!event)
        return uw_result(cudaErrorInvalidResourceHandle);

    struct uw_event_record *record = malloc(sizeof(*record));
    if (!record)
        return uw_result(cudaErrorMemoryAllocation);
    pthread_mutex_lock(&event->lock);
    event->recorded = 1;
    record->generation = ++event->generation;
    pthread_mutex_unlock(&event->lock);
    record->event = event;

    cudaError_t error = uw_submit(stream, uw_event_record_run, record);
    if (error != cudaSuccess) {
        pthread_mutex_lock(&event->lock);
        event->completed = event->generation;
        pthread_cond_broadcast(&event->ready);
        pthread_mutex_unlock(&event->lock);
    }
    return uw_result(error);
}

cudaError_t cudaEventSynchronize(cudaEvent_t handle)
{
    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_event *event = uw_event_find(handle);
    if (!event) {
        pthread_mutex_unlock(&uw_runtime_lock);
        return uw_result(cudaErrorInvalidResourceHandle);
    }
    pthread_mutex_lock(&event->lock);
    pthread_mutex_unlock(&uw_runtime_lock);
    unsigned long target = event->generation;
    while (event->completed < target)
        pthread_cond_wait(&event->ready, &event->lock);
    pthread_mutex_unlock(&event->lock);
    return uw_result(cudaSuccess);
}

cudaError_t cudaEventQuery(cudaEvent_t handle)
{
    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_event *event = uw_event_find(handle);
    if (!event) {
        pthread_mutex_unlock(&uw_runtime_lock);
        return uw_result(cudaErrorInvalidResourceHandle);
    }
    pthread_mutex_lock(&event->lock);
    cudaError_t error = event->completed < event->generation ? cudaErrorNotReady : cudaSuccess;
    pthread_mutex_unlock(&event->lock);
    pthread_mutex_unlock(&uw_runtime_lock);
    return uw_result(error);
}

cudaError_t cudaEventElapsedTime(float *milliseconds, cudaEvent_t start_handle, cudaEvent_t end_handle)
{
    if (!milliseconds)
        return uw_result(cudaErrorInvalidValue);

    pthread_mutex_lock(&uw_runtime_lock);
    struct uw_event *start = uw_event_find(start_handle);
    struct uw_event *end = uw_event_find(end_handle);
    if (!start || !end) {
        pthread_mutex_unlock(&uw_runtime_lock);
        return uw_result(cudaErrorInvalidResourceHandle);
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
    pthread_mutex_unlock(&uw_runtime_lock);

    if (!start_recorded || !end_recorded || !start_ready || !end_ready)
        return uw_result(cudaErrorNotReady);
    if ((start_flags | end_flags) & cudaEventDisableTiming)
        return uw_result(cudaErrorInvalidResourceHandle);

    time_t seconds = end_time.tv_sec - start_time.tv_sec;
    long nanoseconds = end_time.tv_nsec - start_time.tv_nsec;
    *milliseconds = (float)seconds * 1000.0f + (float)nanoseconds / 1000000.0f;
    return uw_result(cudaSuccess);
}
