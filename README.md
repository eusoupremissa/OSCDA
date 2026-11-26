# OSCDA

![OSCDA banner](docs/oscda-banner.png)

OSCDA (Open Source Compute Device Architecture), formerly UniWarp, runs a
subset of CUDA kernels on AMD GPUs through HIP on Linux. The current project
version is **0.2.2**. This is an early compatibility runtime and PTX translator.

## Platform policy

**Linux is the project's sole development and validation target.** The current
hardware reference is the Radeon RX 6600 (`gfx1032`). Compatibility with other
AMD GPUs depends on the ROCm runtime, compiler, and kernel features.

Windows code and an experimental probe remain available for independent
testing. A Windows cross-build or a Wine/VM pass does not establish native GPU
compatibility. Windows results are unverified and are not release criteria.
Contributors are welcome to audit the code, run individual tests on their own
Windows systems, and submit pull requests with reproducible results.

## Changes

**0.2.2** submits CUDA kernel launches on explicit streams to native HIP
queues. Copies, fills, callbacks, and events share the same queue ordering.
The Linux RX 6600 test verifies that one stream can finish while another is
held, and checks 4,099 results from each stream.

**0.2.1** supports LZ4- and Zstandard-compressed PTX in CUDA fatbins, including
the automatic launch path for CUDA executables with embedded PTX on Linux.

**0.2.0** introduced the following features.

- Added the HIP backend for AMD device discovery, memory, and AMDGPU module
  launches. Retained a deterministic CPU backend for host-side tests.
- Added `osc-ptx` for text PTX, PTX inside CUDA fatbins, and
  little-endian ELF64 files with a `.nv_fatbin` section.
- Added translation for scalar arithmetic, direct branches, static shared FP32
  memory, and full-workgroup `bar.sync 0` barriers.
- Added automatic registration of compiler-generated CUDA fatbins and launch
  through `cudaLaunchKernel`. OSC translates on first use and caches the
  loaded module per device for that process.
- Added optional HIP diagnostics and regression coverage for invalid tracked
  memory ranges.

## Build and test on Linux

Use CMake 3.20+, a C11 compiler, and a build tool. Python 3 enables the PTX
tests. GPU execution requires the AMDGPU driver, access to `/dev/kfd` and the
AMD render node under `/dev/dri`, and a working ROCm HIP runtime. Translation
also requires ROCm LLVM Clang. Install `liblz4.so.1` or `libzstd.so.1`
for the compression formats you use. The reference toolchain is HIP 6.4 and LLVM 19.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The GPU test is skipped when HIP reports no usable device. Use
`OSC_HIP_DIAGNOSTICS=1 ./build/amd_runtime_test` to see the native HIP status.
Set `OSC_HIP_LIBRARY` to test a specific HIP shared library, or
`OSC_BACKEND=cpu` to force the CPU backend.

Use `#include <osc/osc.h>` for native OSC APIs or `#include <cuda_runtime.h>`
for the compatibility subset. Link with `-losc`. To install the library,
headers, and `osc-ptx`:

```sh
cmake --install build --prefix /path/to/prefix
```

## Run a translated kernel

Set `OSC_CLANG` to the ROCm compiler if it is outside PATH. Replace
`gfx1032` with the target reported by `amdgpu-arch` for your GPU.

```sh
export OSC_CLANG=/path/to/rocm/llvm/bin/clang++
python3 tools/osc_ptx.py examples/vector_add.ptx --arch gfx1032 -o build/vector_add.hsaco
./build/amd_vector_add build/vector_add.hsaco
python3 tools/osc_ptx.py examples/matmul.ptx --arch gfx1032 -o build/matmul.hsaco
./build/amd_matmul build/matmul.hsaco
```

The vector test compares 4,099 results and checks the last block's padding.
The tiled matrix test compares 2,097 FP32 outputs against CPU results across
aligned and irregular dimensions. Both were validated on the RX 6600.

The translator accepts one textual PTX `.entry` per image with 64-bit
addressing and scalar arguments. Kernel-local static FP32 arrays map to AMD
LDS. Every thread in a workgroup must reach each barrier instance. The frontend
validates branch targets and definite register initialization. Unsupported
instructions, unsupported compression formats, cubin-only images, and ambiguous fatbins
fail explicitly. See the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
and [LLVM AMDGPU guide](https://llvm.org/docs/AMDGPUUsage.html).

## Run a CUDA executable with embedded PTX

This example embeds the supplied PTX with NVIDIA's `nvFatbin` and compiles the
CUDA host stub with Clang. NVIDIA's library is required only to build this
fixture. At runtime OSC extracts and translates the embedded PTX automatically.

```sh
python3 examples/pack_cuda_ptx.py /path/to/libnvfatbin.so.12 build/vector_add.fatbin --compression speed
"$OSC_CLANG" -x cuda --cuda-host-only -nocudainc -nocudalib -std=c++17 -Iinclude \
  -Xclang -target-sdk-version=12.0 -Xclang -fcuda-include-gpubinary \
  -Xclang build/vector_add.fatbin -c examples/cuda_vector_add.cu -o build/cuda_vector_add.o
c++ build/cuda_vector_add.o -Lbuild -Wl,-rpath,"$PWD/build" -losc -o build/cuda_vector_add
OSC_GPU_ARCH=gfx1032 OSC_PTX_TOOL="$PWD/tools/osc_ptx.py" ./build/cuda_vector_add
```

Expected on the RX 6600: two CUDA launches with 4,099 correct results each.
The LZ4 example above uses nvFatbin 12.9. For Zstandard, use nvFatbin 13.5
(`libnvfatbin.so.13`) with `--compression balance`.
Default-stream launches remain synchronous. With an explicit stream, the
launch returns after submission. The first launch can still spend time
translating and loading the embedded module. Use `cudaStreamSynchronize`
or an event to wait for completion. On HIP, `cudaMallocHost` provides pinned
buffers for asynchronous host and device transfers.
`OSC_PYTHON` selects the Python executable when needed. An installed `osc-ptx`
on PATH can replace `OSC_PTX_TOOL`.

OSC implements the registration path observed in NVIDIA's CUDA runtime and
in [Clang's CUDA code generation](https://clang.llvm.org/doxygen/CGCUDANV_8cpp_source.html).
For a manual launch, use `oscModuleLoad`, `oscModuleGetFunction`, and
`oscLaunchKernel` with a compiled AMDGPU code object.

### Validate two CUDA streams

Reuse the compressed `build/vector_add.fatbin` from the previous example:

```sh
"$OSC_CLANG" -x cuda --cuda-host-only -nocudainc -nocudalib -std=c++17 -Iinclude \
  -Xclang -target-sdk-version=12.0 -Xclang -fcuda-include-gpubinary \
  -Xclang build/vector_add.fatbin -c examples/cuda_two_streams.cu -o build/cuda_two_streams.o
c++ build/cuda_two_streams.o -Lbuild -Wl,-rpath,"$PWD/build" -losc -pthread -o build/cuda_two_streams
OSC_GPU_ARCH=gfx1032 OSC_PTX_TOOL="$PWD/tools/osc_ptx.py" ./build/cuda_two_streams
```

The program holds a non-blocking stream with a host callback and waits for
the second stream's event before releasing the first. It then checks both
outputs. This verifies independent progress and ordering, without requiring
that the GPU execute two kernels simultaneously.

## Compatibility limits

The runtime implements a subset of CUDA host APIs. CUDA C++ device compilation,
SASS translation, general CUDA binary compatibility, CUDA graphs, and NVIDIA
libraries such as cuBLAS and cuDNN are outside the current scope. PyTorch CUDA
wheels and applications such as RVC have not been validated with OSCDA.

## Contributing

Use a pull request for changes. For GPU results, include the GPU and target ID,
OS and driver versions, HIP and Clang versions, exact commands, and logs.
Keep platform-specific results separate from Linux validation.

OSCDA is independent of AMD and NVIDIA. CUDA is an NVIDIA trademark.
Released under the [MIT License](LICENSE).
