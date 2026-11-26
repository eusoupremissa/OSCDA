#define _GNU_SOURCE

#include <osc/osc.h>

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <link.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

cudaError_t osc_result(cudaError_t error);
cudaError_t osc_launch_kernel(oscFunction_t function, dim3 grid, dim3 block,
                              unsigned int shared, void **arguments, cudaStream_t stream);

struct fatbin_wrapper {
    uint32_t magic, version;
    const unsigned char *image;
    const void *unused;
};

struct loaded_module {
    int device;
    oscModule_t module;
    struct loaded_module *next;
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
    struct loaded_module *modules;
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
static int module_cleanup_installed;

static void unload_modules(struct registered_kernel *kernel)
{
    if (!kernel->modules)
        return;
    int current;
    int restore = cudaGetDevice(&current) == cudaSuccess;
    while (kernel->modules) {
        struct loaded_module *loaded = kernel->modules;
        kernel->modules = loaded->next;
        if (cudaSetDevice(loaded->device) == cudaSuccess)
            oscModuleUnload(loaded->module);
        free(loaded);
    }
    if (restore)
        cudaSetDevice(current);
}

static void unload_modules_at_exit(void)
{
    pthread_mutex_lock(&registry_lock);
    for (struct registered_kernel *kernel = kernels; kernel; kernel = kernel->next)
        unload_modules(kernel);
    pthread_mutex_unlock(&registry_lock);
}

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
            *kernel = (struct registered_kernel){host_stub, name, fatbin, NULL, kernels};
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
                unload_modules(removed);
                free(removed);
            } else {
                kernel = &(*kernel)->next;
            }
        }
        free(fatbin);
    }
    pthread_mutex_unlock(&registry_lock);
}

static int run_translator(const char *input, const char *output, const char *arch,
                          const char *kernel)
{
    const char *python = getenv("OSC_PYTHON");
    const char *program =
        "import os,runpy,shutil,sys;"
        "p=os.environ.get('OSC_PTX_TOOL') or shutil.which('osc-ptx');"
        "p or sys.exit('OSC: install osc-ptx or set OSC_PTX_TOOL');"
        "sys.argv[0]=p;runpy.run_path(p,run_name='__main__')";
#ifdef _WIN32
    if (!python || !*python)
        python = "python.exe";
    char code[512], in_arg[MAX_PATH + 3], out_arg[MAX_PATH + 3];
    snprintf(code, sizeof(code), "\"%s\"", program);
    snprintf(in_arg, sizeof(in_arg), "\"%s\"", input);
    snprintf(out_arg, sizeof(out_arg), "\"%s\"", output);
    const char *args[] = {"python", "-c", code, in_arg, "--arch", arch, "--kernel", kernel, "-o", out_arg, NULL};
    return _spawnvp(_P_WAIT, python, args) == 0;
#else
    if (!python || !*python)
        python = "python3";
    char *args[] = {(char *)python, "-c", (char *)program, (char *)input, "--arch",
                    (char *)arch, "--kernel", (char *)kernel, "-o", (char *)output, NULL};
    pid_t child;
    int error = posix_spawnp(&child, python, NULL, NULL, args, environ);
    if (error) {
        fprintf(stderr, "OSC: cannot start PTX translator with %s: %s\n", python,
                strerror(error));
        return 0;
    }
    int status;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    return waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

static cudaError_t translate_module(const struct registered_kernel *kernel, int device,
                                    oscModule_t *module)
{
    const struct registered_fatbin *fatbin = kernel->owner;
    const char *arch = getenv("OSC_GPU_ARCH");
    char detected[256];
    if (!arch || !*arch) {
        cudaError_t result = oscGetDeviceArch(device, detected, sizeof(detected));
        if (result != cudaSuccess) {
            fprintf(stderr, "OSC: cannot detect AMD target for device %d (%s); set OSC_GPU_ARCH\n",
                    device, cudaGetErrorName(result));
            return result;
        }
        arch = detected;
    }
#ifdef _WIN32
    char directory[MAX_PATH], input[MAX_PATH], output[MAX_PATH];
    DWORD length = GetTempPathA(sizeof(directory), directory);
    if (!length || length >= sizeof(directory) || !GetTempFileNameA(directory, "osc", 0, input))
        return cudaErrorInitializationError;
    if (!GetTempFileNameA(directory, "osc", 0, output)) {
        remove(input);
        return cudaErrorInitializationError;
    }
#else
    char directory[] = "/tmp/osc-cuda-XXXXXX";
    if (!mkdtemp(directory))
        return cudaErrorInitializationError;
    char input[sizeof(directory) + 16], output[sizeof(directory) + 16];
    snprintf(input, sizeof(input), "%s/input.fatbin", directory);
    snprintf(output, sizeof(output), "%s/kernel.hsaco", directory);
#endif
    FILE *file = fopen(input, "wb");
    int saved = file && fwrite(fatbin->image, 1, fatbin->size, file) == fatbin->size;
    if (file && fclose(file))
        saved = 0;
    cudaError_t result = cudaErrorInitializationError;
    if (!saved) {
        fprintf(stderr, "OSC: failed to stage the fatbin for %s\n", kernel->name);
    } else if (!run_translator(input, output, arch, kernel->name)) {
        fprintf(stderr, "OSC: PTX translation failed for %s on %s\n", kernel->name, arch);
    } else {
        result = oscModuleLoad(module, output);
        if (result != cudaSuccess)
            fprintf(stderr, "OSC: HIP could not load %s for %s: %s\n", kernel->name,
                    arch, cudaGetErrorName(result));
    }
    remove(input);
    remove(output);
#ifndef _WIN32
    rmdir(directory);
#endif
    return result;
}

cudaError_t cudaLaunchKernel(const void *function, dim3 grid, dim3 block, void **arguments,
                             size_t shared, cudaStream_t stream)
{
    if (!function || !grid.x || !grid.y || !grid.z || !block.x || !block.y || !block.z ||
        shared > UINT_MAX)
        return osc_result(cudaErrorInvalidValue);
    if (oscGetBackend() != oscBackendAmdHip)
        return osc_result(cudaErrorNotSupported);

    int device;
    cudaError_t result = cudaGetDevice(&device);
    if (result != cudaSuccess)
        return result;
    pthread_mutex_lock(&registry_lock);
    struct registered_kernel *kernel = kernels;
    while (kernel && kernel->host_stub != function)
        kernel = kernel->next;
    result = cudaErrorInvalidResourceHandle;
    if (kernel) {
        struct loaded_module *loaded = kernel->modules;
        while (loaded && loaded->device != device)
            loaded = loaded->next;
        result = cudaSuccess;
        if (!loaded) {
            loaded = malloc(sizeof(*loaded));
            result = loaded ? translate_module(kernel, device, &loaded->module) : cudaErrorMemoryAllocation;
            if (result == cudaSuccess) {
                loaded->device = device;
                loaded->next = kernel->modules;
                kernel->modules = loaded;
            } else {
                free(loaded);
            }
        }
        if (result == cudaSuccess) {
            oscFunction_t native;
            result = oscModuleGetFunction(&native, loaded->module, kernel->name);
            if (result == cudaSuccess)
                result = osc_launch_kernel(native, grid, block, (unsigned int)shared, arguments, stream);
        }
        if (kernel->modules && !module_cleanup_installed) {
            // Constructors register CUDA exit hooks before HIP initializes.
            if (atexit(unload_modules_at_exit) == 0) {
                module_cleanup_installed = 1;
            } else {
                unload_modules(kernel);
                result = cudaErrorInitializationError;
            }
        }
    }
    pthread_mutex_unlock(&registry_lock);
    if (!stream && result == cudaSuccess)
        return cudaStreamSynchronize(NULL);
    return osc_result(result);
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
