# Applio inference on the RX 6600

OSCDA runs the FP32 ReLU inside Applio's RVC encoder feed-forward network.
The Rust bridge owns the compiled kernel module and submits the kernel on the
current PyTorch HIP stream using the existing device buffers. It uses the same
HIP shared library as PyTorch. The PTX frontend and LLVM produce the AMDGPU code
object. The rest of the model runs on native PyTorch ROCm.

## RX 6600 result

On the RX 6600 (`gfx1032`), OSCDA handled all six tested ReLU calls in the
RVC encoder. The output contained 281,600 PCM16 samples at 40 kHz, with zero
sample differences against native ROCm. The input was a speech excerpt from
LibriSpeech. The test used PyTorch 2.9.1 and the AMD TheRock ROCm 7.13 build
for `gfx103X-dgpu`, without an architecture override. The ROCm build emitted
MIOpen warnings about missing device tuning data and an optional grouped
convolution library, and the inference completed. [Input source](https://docs.pytorch.org/audio/2.8/tutorials/asr_inference_with_cuda_ctc_decoder_tutorial.html)

## CUDA dependencies

- The Applio Linux installer adds the NVIDIA `cu128` wheel index. The offline
  PyTorch inference code also supports a ROCm build, which exposes HIP through
  `torch.cuda`. [Installer](https://github.com/IAHispano/Applio/blob/b04a24baec709cbfac1e4bb9e960dde730906b62/run-install.sh#L140-L148) · [Applio device selection](https://github.com/IAHispano/Applio/blob/b04a24baec709cbfac1e4bb9e960dde730906b62/rvc/configs/config.py#L30-L95)
- The ONNX wrapper selects NVIDIA's CUDA execution provider and its graph path
  only for CUDA builds. With ROCm it uses its CPU provider. This validation
  selects the Torch engine and the RMVPE pitch predictor. [ONNX provider selection](https://github.com/IAHispano/Applio/blob/b04a24baec709cbfac1e4bb9e960dde730906b62/rvc/infer/wrapper.py#L73-L125)
- The integration replaces the RVC encoder's ReLU for one offline inference
  run. It leaves GELU and the other PyTorch operations on their existing path.
  [Applio FFN implementation](https://github.com/IAHispano/Applio/blob/b04a24baec709cbfac1e4bb9e960dde730906b62/rvc/lib/algorithm/attentions.py#L221-L231)

## Reproduce

Use 64-bit Linux, Python 3.14, Rust 1.85+, and LLVM Clang with the HIP frontend
and AMDGPU backend. The RX 6600 test used ROCm LLVM Clang 19. Use a filesystem
with several GiB free for the ROCm wheels and model weights.

From the OSCDA checkout, build the C runtime and Rust bridge:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
OSCDA_LIB_DIR="$PWD/build" cargo build --locked --release \
  --manifest-path rust/oscda-hip/Cargo.toml --target-dir build/rust
LD_LIBRARY_PATH="$PWD/build" OSCDA_LIB_DIR="$PWD/build" cargo test --locked \
  --manifest-path rust/oscda-hip/Cargo.toml --target-dir build/rust
```

In a writable directory outside the OSCDA checkout, prepare Applio and the
tested gfx1032-capable AMD TheRock packages:

```sh
git clone https://github.com/IAHispano/Applio.git applio
git -C applio checkout --detach b04a24baec709cbfac1e4bb9e960dde730906b62
python3.14 -m venv venv
cat > torch-constraints.txt <<'CONSTRAINTS'
torch==2.9.1+rocm7.13.0a20260421
torchaudio==2.9.0+rocm7.13.0a20260421
CONSTRAINTS
venv/bin/python -m pip install --no-cache-dir \
  --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/ \
  -r torch-constraints.txt 'rocm-sdk-libraries-gfx103x-dgpu==7.13.0a20260421'
sed -E '/^(torch|torchaudio|webrtcvad-wheels)==/d' applio/requirements-engine.txt \
  > requirements-offline.txt
venv/bin/python -m pip install --no-cache-dir -c torch-constraints.txt -r requirements-offline.txt
```

The offline path does not import the real-time WebRTC VAD extension. Confirm
on your GPU that `torch.cuda.get_arch_list()` includes its target and
`torch.version.hip` is set.

Use a trusted Applio V2 inference checkpoint and a WAV input. Place ContentVec
and RMVPE at the paths below. To reproduce with Applio's published 40 kHz
training checkpoint, export it using Applio's own extraction function.

```sh
assets=https://huggingface.co/IAHispano/Applio/resolve/main/Resources
embedder=applio/rvc/models/embedders/contentvec
predictor=applio/rvc/models/predictors
mkdir -p "$embedder" "$predictor" fixture
curl -fL "$assets/embedders/contentvec/config.json" -o "$embedder/config.json"
curl -fL "$assets/embedders/contentvec/pytorch_model.bin?download=true" -o "$embedder/pytorch_model.bin"
curl -fL "$assets/predictors/rmvpe.pt?download=true" -o "$predictor/rmvpe.pt"
curl -fL "$assets/pretrained_v2/f0G40k.pth?download=true" -o fixture/f0G40k.pth
curl -fL https://download.pytorch.org/torchaudio/tutorial-assets/ctc-decoding/1688-142285-0007.wav \
  -o fixture/input.wav
```

```sh
(
  cd applio
  APPLIO_CONFIG_DIR="$PWD/../fixture/settings" ../venv/bin/python - <<'PY'
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from rvc.train.process.extract_model import extract_model

fixture = Path("../fixture")
(fixture / "model_info.json").write_text(json.dumps({"embedder_model": "contentvec"}))
checkpoint = torch.load(fixture / "f0G40k.pth", map_location="cpu", weights_only=True)
hps = json.loads(Path("rvc/configs/40000.json").read_text(),
                 object_hook=lambda values: SimpleNamespace(**values))
extract_model(checkpoint["model"], "40k", "Applio pretrained generator",
              str(fixture / "applio-40k.pth"), 0, 0, hps, "HiFi-GAN")
PY
)
```

Set `OSCDA` to the absolute path of the OSCDA checkout and `OSC_CLANG` to
the LLVM Clang executable used for PTX compilation:

```sh
APPLIO_CONFIG_DIR="$PWD/fixture/settings" \
OSC_CLANG=/path/to/rocm/llvm/bin/clang++ \
venv/bin/python "$OSCDA/examples/applio_inference.py" \
  --applio applio --osc-library "$OSCDA/build/libosc.so" \
  --rust-bridge "$OSCDA/build/rust/release/liboscda_hip.so" \
  --model fixture/applio-40k.pth --input fixture/input.wav --output fixture/oscda.wav
```

The runner checks the operator against PyTorch on the GPU, warms up the Torch
engine, and runs with a fixed random seed for both audio comparisons. It checks
that the OSCDA operator ran inside the model and allows at most one PCM16 step
of absolute sample error. Set `OSC_HIP_DIAGNOSTICS=1` to log kernel loading
and dispatch. This test covers offline inference through the Torch engine.
