# UniWarp

UniWarp is an independent, open source GPU compute runtime. Its first compatibility surface follows the CUDA Runtime API. The execution model is meant to run on NVIDIA, AMD, CPUs, and whatever accelerator turns up next with a usable toolchain.

The project is young, which is the honest way of saying that version 0.1 does one thing and tries to do it correctly. It provides a deterministic CPU runtime for host-side CUDA programs, with device discovery, tracked device memory, synchronous and asynchronous copies, memory fills, streams, events, elapsed timing, host callbacks, device synchronization, and thread-local error state.

UniWarp does not compile CUDA kernels yet, and it does not claim binary compatibility with NVIDIA libraries. Pretending otherwise would not make either statement true. The boundaries are deliberate, documented, and visible.

## Build

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Applications can include `cuda_runtime.h` and link against `libuniwarp`:

```sh
cc program.c -Iinclude -Lbuild -luniwarp -Wl,-rpath,build
```

## Current surface

The runtime currently supports exactly this:

* One deterministic CPU device
* `cudaMalloc`, `cudaFree`, `cudaMallocHost`, and `cudaFreeHost`
* Tracked pointer ranges and copy direction validation
* `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, and `cudaMemsetAsync`
* Ordered streams backed by independent worker threads
* Events, queries, synchronization, and elapsed time
* Ordered host callbacks
* Device wide synchronization
* Per-thread last error state

The CPU device is a correctness backend, not a benchmark prop. Device allocations remain opaque through the public API but are host allocations internally. Host callbacks make stream ordering and memory semantics testable before a kernel compiler exists, because debugging all of that after adding a compiler would be a remarkably avoidable mistake.

## Architecture

UniWarp follows four boundaries:

```text
CUDA source and runtime calls
            |
    CUDA compatible frontend
            |
      UniWarp module IR
            |
   Backend execution contract
       /       |       \
     CPU     ROCm    NVIDIA
```

Only the runtime and CPU execution core exist today. The backend interface will be extracted when the ROCm backend lands. Creating it earlier would merely promote guesses to architecture, a surprisingly popular way to make software worse.

The compiler path will accept CUDA C++ through Clang, retain PTX as an input format, and lower kernels into a versioned UniWarp IR. Native backends can then select LLVM AMDGPU, SPIR-V, PTX, or another machine-specific route. The IR must preserve address spaces, barriers, warp operations, memory ordering, and floating-point intent before it gets clever with optimization. Fast nonsense is still nonsense.

## Engineering rules

1. Correctness precedes coverage.
2. Compatibility claims require executable tests.
3. Public interfaces are stable only after two real backends use them.
4. The runtime owns ordering, lifetime, and error semantics.
5. Backends own code generation, queues, and device memory.
6. Vendor quirks remain isolated and named.
7. Unsupported behavior fails explicitly.
8. Dependencies must earn their place in the hot path.

## Prior art

UniWarp studies existing systems instead of pretending GPU computing began with this repository:

* [ROCm HIP](https://github.com/ROCm/HIP) demonstrates a practical CUDA shaped portability API and a clean route to AMD hardware.
* [chipStar](https://github.com/CHIP-SPV/chipStar) demonstrates CUDA and HIP execution through SPIR-V with OpenCL and Level Zero backends.
* [GPUOcelot](https://github.com/gpuocelot/gpuocelot) demonstrates PTX parsing, a formal intermediate representation, CPU emulation, dynamic translation, and correctness tooling.
* [ZLUDA](https://github.com/vosen/ZLUDA) demonstrates drop-in CUDA driver translation and the operational difficulty of mapping vendor specific behavior.
* [LLVM](https://github.com/llvm/llvm-project) provides the compiler infrastructure needed for serious CUDA, AMDGPU, and SPIR-V work.

No source code from the NVIDIA CUDA SDK is included. Code from other projects enters UniWarp only through deliberate reuse with compatible licenses, attribution, and tests. Copy-paste is not an architecture and copyright is not an optional compiler warning.

## Roadmap

The next milestones are ordered by dependency, not by which one looks best in an announcement:

1. Lock down runtime semantics with adversarial stream and memory tests.
2. Add a real backend contract and a ROCm implementation.
3. Load versioned modules and launch precompiled AMDGPU kernels.
4. Parse a strict, useful PTX subset into a typed control flow graph.
5. Execute that subset in the deterministic CPU backend.
6. Lower the same IR to LLVM AMDGPU.
7. Add a Clang based CUDA compiler driver.
8. Grow API and library coverage from real applications.

## Independence

UniWarp is not affiliated with, sponsored by, or endorsed by NVIDIA or AMD. CUDA is a trademark of NVIDIA Corporation. Other names belong to their respective owners.
