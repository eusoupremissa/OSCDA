# OSCDA HIP interop for Rust

Load any named entry from an AMDGPU code object with `oscda_hip::Kernel::load`.
Call `Kernel::launch` with grid and block dimensions, the kernel's argument
array, optional dynamic shared memory, and a borrowed HIP stream. Arguments
use HIP's ABI, with a pointer to the host storage for each argument. The
framework or caller owns device allocations and streams. The kernel owns the
loaded module and unloads it on drop. The code object, buffers, stream, and
current device must belong to the same HIP runtime and device.

Native callers of the bridge can use `oscda_kernel_load`,
`oscda_kernel_launch`, and `oscda_kernel_unload` with the corresponding
`Kernel` handle and C-compatible `Dim3` (three unsigned integers). The original
ReLU symbols remain exported for callers using the 0.2.4 example.

## Rust example

From the OSCDA checkout, build the runtime, translate the existing vector
kernel with LLVM, and run it from Rust. Replace `gfx1032` with your GPU's target.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
OSC_CLANG=/path/to/rocm/llvm/bin/clang++ \
  python3 tools/osc_ptx.py examples/vector_add.ptx --arch gfx1032 -o build/vector_add.hsaco
LD_LIBRARY_PATH="$PWD/build" OSCDA_LIB_DIR="$PWD/build" \
  cargo run --locked --release --manifest-path rust/oscda-hip/Cargo.toml \
  --example vector_add -- "$PWD/build/vector_add.hsaco"
```

The example uses Rust, OSCDA and HIP. The [Applio validation](../../docs/applio-validation.md)
exercises the same interface with tensors and a stream owned by PyTorch ROCm.
