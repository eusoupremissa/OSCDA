#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

// The device definition is assembled separately in vector_add.ptx.
extern "C" __attribute__((global)) void vector_add(const float *, const float *, float *,
                                                    unsigned int) {}

static void check(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        std::abort();
    }
}

struct Gate {
    std::atomic_bool entered{false};
    std::atomic_bool release{false};
    std::atomic_bool expired{false};
};

static void CUDART_CB hold_stream(void *data)
{
    auto &gate = *static_cast<Gate *>(data);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    gate.entered.store(true);
    while (!gate.release.load()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            gate.expired.store(true);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int main()
{
    constexpr unsigned int count = 4099;
    constexpr size_t bytes = count * sizeof(float);
    struct Pipeline {
        cudaStream_t stream;
        cudaEvent_t done;
        float *a, *b, *out;
        float *da, *db, *result;
    } pipes[2]{};

    for (unsigned int lane = 0; lane < 2; ++lane) {
        auto &p = pipes[lane];
        if (lane == 0)
            check(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
        else
            check(cudaStreamCreate(&p.stream));
        check(cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming));
        check(cudaMallocHost(reinterpret_cast<void **>(&p.a), bytes));
        check(cudaMallocHost(reinterpret_cast<void **>(&p.b), bytes));
        check(cudaMallocHost(reinterpret_cast<void **>(&p.out), bytes));
        check(cudaMalloc(reinterpret_cast<void **>(&p.da), bytes));
        check(cudaMalloc(reinterpret_cast<void **>(&p.db), bytes));
        check(cudaMalloc(reinterpret_cast<void **>(&p.result), bytes));
        for (unsigned int i = 0; i < count; ++i) {
            p.a[i] = static_cast<float>(i + lane * count);
            p.b[i] = 1.0f + (i % 7) * 0.25f;
        }
    }

    Gate gate;
    check(cudaLaunchHostFunc(pipes[0].stream, hold_stream, &gate));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!gate.entered.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!gate.entered.load())
        std::abort();

    for (auto &p : pipes) {
        check(cudaMemcpyAsync(p.da, p.a, bytes, cudaMemcpyHostToDevice, p.stream));
        check(cudaMemcpyAsync(p.db, p.b, bytes, cudaMemcpyHostToDevice, p.stream));
        check(cudaMemsetAsync(p.result, 0, bytes, p.stream));
        vector_add<<<(count + 255) / 256, 256, 0, p.stream>>>(p.da, p.db, p.result, count);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(p.out, p.result, bytes, cudaMemcpyDeviceToHost, p.stream));
        check(cudaEventRecord(p.done, p.stream));
    }

    check(cudaEventSynchronize(pipes[1].done));
    check(cudaStreamSynchronize(pipes[1].stream));
    const bool independent = cudaStreamQuery(pipes[0].stream) == cudaErrorNotReady &&
                             cudaEventQuery(pipes[0].done) == cudaErrorNotReady;
    gate.release.store(true);
    check(cudaStreamSynchronize(pipes[0].stream));
    if (!independent || gate.expired.load()) {
        std::fputs("Independent stream did not finish while the first was held\n", stderr);
        return EXIT_FAILURE;
    }
    check(cudaEventQuery(pipes[0].done));

    for (auto &p : pipes) {
        for (unsigned int i = 0; i < count; ++i) {
            if (p.out[i] != p.a[i] + p.b[i]) {
                std::fprintf(stderr, "Incorrect result at index %u\n", i);
                return EXIT_FAILURE;
            }
        }
        check(cudaEventDestroy(p.done));
        check(cudaStreamDestroy(p.stream));
        check(cudaFree(p.result));
        check(cudaFree(p.db));
        check(cudaFree(p.da));
        check(cudaFreeHost(p.out));
        check(cudaFreeHost(p.b));
        check(cudaFreeHost(p.a));
    }
    int device;
    cudaDeviceProp properties;
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    std::printf("%s: 2 independent CUDA streams, %u correct results each\n",
                properties.name, count);
}
