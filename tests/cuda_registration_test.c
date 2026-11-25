#include <cuda_runtime.h>

#undef NDEBUG
#include <assert.h>
#include <stdint.h>

struct fatbin_wrapper {
    uint32_t magic, version;
    const unsigned char *image;
    const void *unused;
};

CUDARTAPI void **__cudaRegisterFatBinary(const struct fatbin_wrapper *wrapper);
CUDARTAPI void __cudaUnregisterFatBinary(void **handle);

int main(void)
{
    static const unsigned char image[] = {
        0x50, 0xed, 0x55, 0xba, 1, 0, 16, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
    };
    static const unsigned char truncated[] = {
        0x50, 0xed, 0x55, 0xba, 1, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 1
    };
    struct fatbin_wrapper wrapper = {0x466243b1, 1, image, NULL};
    void **handle = __cudaRegisterFatBinary(&wrapper);
    assert(handle != NULL);
    __cudaUnregisterFatBinary(handle);
    wrapper.image = truncated;
    assert(__cudaRegisterFatBinary(&wrapper) == NULL);

    dim3 grid = {3, 2, 1}, block = {8, 4, 1}, actual_grid, actual_block;
    size_t shared;
    cudaStream_t stream;
    assert(__cudaPushCallConfiguration(grid, block, 16, NULL) == cudaSuccess);
    grid.x = 7;
    assert(__cudaPushCallConfiguration(grid, block, 32, NULL) == cudaSuccess);
    assert(__cudaPopCallConfiguration(&actual_grid, &actual_block, &shared, &stream) == cudaSuccess);
    assert(actual_grid.x == 7 && actual_block.y == 4 && shared == 32 && stream == NULL);
    assert(__cudaPopCallConfiguration(&actual_grid, &actual_block, &shared, &stream) == cudaSuccess);
    assert(actual_grid.x == 3 && actual_block.y == 4 && shared == 16);
    assert(__cudaPopCallConfiguration(&actual_grid, &actual_block, &shared, &stream) ==
           cudaErrorInvalidValue);
    return 0;
}
