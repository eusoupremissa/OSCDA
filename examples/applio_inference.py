"""Compare native Applio inference with OSCDA running its FP32 FFN ReLU on HIP."""

import argparse
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def use_torch_hip_library():
    # Load exactly the HIP library that owns PyTorch's allocations and streams.
    libraries = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            path = Path(fields[5])
            if path.is_absolute() and path.name.startswith("libamdhip64.so"):
                libraries.add(path.resolve())
    if len(libraries) != 1:
        raise RuntimeError(f"expected one loaded PyTorch HIP runtime, found {sorted(libraries)}")
    library, = libraries
    override = os.environ.get("OSC_HIP_LIBRARY")
    if override and Path(override).resolve() != library:
        raise RuntimeError("OSC_HIP_LIBRARY differs from PyTorch's loaded HIP runtime")
    os.environ["OSC_HIP_LIBRARY"] = str(library)
    return library


class Dim3(c.Structure):
    _fields_ = [(axis, c.c_uint) for axis in "xyz"]


class OscRelu:
    def __init__(self, torch, library, bridge, device, build_directory):
        self.torch = torch
        self.launches = 0
        self.runtime = c.CDLL(str(library), mode=c.RTLD_GLOBAL)
        self.bridge = c.CDLL(str(bridge))
        ptr = c.c_void_p
        signatures = [
            (self.runtime.cudaGetErrorName, [c.c_int], c.c_char_p),
            (self.runtime.oscGetDeviceArch, [c.c_int, c.c_char_p, c.c_size_t], c.c_int),
            (self.bridge.oscda_kernel_load, [c.c_char_p, c.c_char_p, c.POINTER(ptr)], c.c_int),
            (self.bridge.oscda_kernel_launch, [ptr, Dim3, Dim3, c.c_uint, c.POINTER(ptr), ptr], c.c_int),
            (self.bridge.oscda_kernel_unload, [ptr], c.c_int),
        ]
        for function, arguments, result in signatures:
            function.argtypes = arguments
            function.restype = result
        arch = c.create_string_buffer(256)
        self.check(self.runtime.oscGetDeviceArch, device, arch, len(arch))
        self.arch = arch.value.decode()
        code = build_directory / "applio_relu.hsaco"
        subprocess.run(
            [sys.executable, str(ROOT / "tools/osc_ptx.py"), str(ROOT / "examples/applio_relu.ptx"),
             "--arch", self.arch, "-o", str(code)], check=True,
        )
        self.kernel = ptr()
        self.check(self.bridge.oscda_kernel_load, os.fsencode(code), b"applio_relu", c.byref(self.kernel))

    def check(self, function, *args):
        status = function(*args)
        if status:
            error = self.runtime.cudaGetErrorName(status).decode()
            raise RuntimeError(f"{function.__name__} returned {error} ({status})")

    def close(self):
        self.check(self.bridge.oscda_kernel_unload, self.kernel)

    def __call__(self, x):
        if x.dtype != self.torch.float32 or not x.is_cuda or not x.is_contiguous():
            raise RuntimeError("OSCDA ReLU requires a contiguous FP32 HIP tensor")
        if not 0 < x.numel() <= 0xFFFFFFFF:
            raise RuntimeError("OSCDA ReLU requires a nonempty tensor with at most 2**32 - 1 elements")
        y = self.torch.empty_like(x)
        stream = self.torch.cuda.current_stream(x.device)
        x.record_stream(stream)
        values = (c.c_uint64(x.data_ptr()), c.c_uint64(y.data_ptr()), c.c_uint32(x.numel()))
        arguments = (c.c_void_p * len(values))(*(c.addressof(value) for value in values))
        threads = 256
        grid = Dim3((x.numel() + threads - 1) // threads, 1, 1)
        self.check(self.bridge.oscda_kernel_launch, self.kernel, grid, Dim3(threads, 1, 1),
                   0, arguments, stream.cuda_stream)
        self.launches += 1
        return y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applio", required=True, type=Path, help="path to the Applio checkout")
    parser.add_argument("--osc-library", required=True, type=Path, help="path to libosc.so")
    parser.add_argument("--rust-bridge", required=True, type=Path, help="path to liboscda_hip.so")
    parser.add_argument("--model", required=True, type=Path, help="Applio inference .pth checkpoint")
    parser.add_argument("--input", required=True, type=Path, help="input audio")
    parser.add_argument("--output", required=True, type=Path, help="new WAV output path")
    args = parser.parse_args()
    applio = args.applio.resolve(strict=True)
    library, bridge, model, audio = (path.resolve(strict=True) for path in
                                    (args.osc_library, args.rust_bridge, args.model, args.input))
    if model.suffix.lower() != ".pth":
        parser.error("--model must be an Applio PyTorch inference .pth checkpoint")
    output = args.output.resolve()
    if output.suffix.lower() != ".wav" or output.exists():
        parser.error("--output must name a new .wav file")

    os.chdir(applio)
    sys.path.insert(0, str(applio))
    import numpy as np
    import soundfile as sf
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("Applio validation requires a GPU-enabled PyTorch ROCm build")
    print(f"PyTorch {torch.__version__}, HIP {torch.version.hip}, CUDA {torch.version.cuda}")
    print(f"HIP runtime: {use_torch_hip_library()}")

    from rvc.infer.infer import VoiceConverter
    from rvc.infer.pipeline import Pipeline
    from rvc.lib.algorithm.attentions import FFN

    converter = VoiceConverter()
    device = torch.device(converter.config.device)
    if device.type != "cuda":
        raise RuntimeError(f"Applio selected {device}, expected HIP")
    stream = torch.cuda.Stream(device=device)
    with tempfile.TemporaryDirectory() as directory, torch.cuda.device(device), torch.cuda.stream(stream):
        work = Path(directory)
        relu = OscRelu(torch, library, bridge, torch.cuda.current_device(), work)
        try:
            print(f"{torch.cuda.get_device_name(device)} ({relu.arch})")
            x = torch.tensor([-1.0, 0.0, 1.0, float("nan")], device=device).repeat(257)
            torch.testing.assert_close(relu(x), torch.relu(x), rtol=0, atol=0, equal_nan=True)
            original = FFN._apply_activation

            def activate(module, tensor):
                return original(module, tensor) if module.activation == "gelu" else relu(tensor)

            def convert(path):
                torch.manual_seed(17)
                converter.convert_audio(
                    audio_input_path=str(audio), audio_output_path=str(path),
                    model_path=str(model), index_path="", index_rate=0,
                    inference_engine="torch", f0_method="rmvpe",
                )
                stream.synchronize()

            voice_conversion = Pipeline.voice_conversion

            def verify_device(pipeline, encoder, synthesizer, *args, **kwargs):
                for network in (encoder, synthesizer):
                    if next(network.parameters()).device != device:
                        raise RuntimeError(f"Applio moved {type(network).__name__} off {device}")
                return voice_conversion(pipeline, encoder, synthesizer, *args, **kwargs)

            with torch.inference_mode(), patch.object(Pipeline, "voice_conversion", verify_device):
                convert(work / "warmup.wav")
                reference = work / "rocm.wav"
                convert(reference)
                accelerated = work / "oscda.wav"
                before = relu.launches
                with patch.object(FFN, "_apply_activation", activate):
                    convert(accelerated)
                count = relu.launches - before

            expected, rate = sf.read(reference, dtype="float32")
            actual, actual_rate = sf.read(accelerated, dtype="float32")
            if not count or rate != actual_rate or expected.shape != actual.shape or not actual.size:
                raise RuntimeError("inference failed to produce comparable audio with OSCDA activations")
            if not np.isfinite(expected).all() or not np.isfinite(actual).all() or not np.any(actual):
                raise RuntimeError("inference output is silent or non-finite")
            error = float(np.max(np.abs(expected - actual)))
            if error > 1 / 32768:
                raise RuntimeError(f"inference diverged from native ROCm (maximum error {error:g})")
            with output.open("xb") as destination:
                destination.write(accelerated.read_bytes())
            print(f"OSCDA ReLU launches during inference: {count}")
            print(f"{len(actual)} samples at {rate} Hz; maximum PCM error {error:g}")
        finally:
            stream.synchronize()
            relu.close()


if __name__ == "__main__":
    main()
