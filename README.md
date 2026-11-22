# LittleBit on a Steam Deck memory budget

**Target: Llama-2-7B. Executed CPU experiment: TinyStories-260K.** The 7B runner uses SamsungLabs' implementation. The CPU experiment is an independent, scaled-down reproduction with working packed checkpoints. Only the latter has been run here.

## Run Qwen3-8B locally on Steam Deck

Measure peak memory, time to first token, and decoding speed for the pretrained **Qwen3-8B LittleBit-2 (0.55 BPW)** checkpoint. The benchmark runs on the CPU in Desktop Mode without training, CUDA, ROCm, or SSH. The weights take about 3 GB on disk. Allow 8 GB of free disk space for the download, Python, and dependencies. The benchmark reports actual peak memory use.

**1. Set up the environment.** Switch to Desktop Mode, open Konsole, and run `bash` before pasting the commands below. They install Python and the packages in your home directory without sudo. If you already cloned the repository, replace the first two lines with a change to your existing checkout and `git pull --ff-only`.

```bash
git clone https://github.com/ILayZ/littlebit-steamdeck.git
cd littlebit-steamdeck
df -h .
free -h
command -v uv >/dev/null || curl --proto '=https' -fsSL https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 .venv-deck
source .venv-deck/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
uv pip install 'transformers==4.51.3' 'accelerate==1.6.0' 'numpy==2.2.4' 'huggingface-hub==0.30.2' 'safetensors==0.5.3'
```

**2. Download the model.** This command caches a pinned public revision on your Deck. You can run it without a Hugging Face account and rerun it if the download is interrupted.

```bash
python deck_qwen.py --download-only
```

**3. Run a short generation test.** Close any running game and use the same Konsole session. In a new session, start Bash, change to the repository directory, and activate `.venv-deck` as shown above.

```bash
set -o pipefail
python -u deck_qwen.py --offline --threads 4 --tokens 16 2>&1 | tee deck-generation.log
```

The test asks for the capital of Spain and prints the model’s answer. The log reports model load time (`load_s`), time from the start of prompt processing to the first token (`first_token_s`), decoding speed for subsequent tokens (`decode_tokens_s`), and peak memory for the whole process, including loading (`peak_rss_mib`). If the first token ends the response, `decode_tokens_s` is `null`. To compare four and eight threads, rerun with `--threads 8` and a different log filename. Keep the same power mode and power limit for both runs.

**4. Optionally measure WikiText-2 perplexity.** The first run needs an internet connection to download the test split. Start with one 2,048-token block to estimate runtime. The `--offline` flag applies to the model weights.

```bash
uv pip install 'datasets==3.5.0'
python -u deck_qwen.py --offline --ppl-blocks 1 2>&1 | tee deck-ppl-one.log
```

Run the full test to compare with the checkpoint publisher’s reported WikiText-2 perplexity of **24.48**. The one-block run scores only a sample. The CPU loader and BF16 arithmetic may yield a different result.

```bash
python -u deck_qwen.py --offline --ppl-blocks 0 2>&1 | tee deck-ppl-full.log
```

**Troubleshooting.** If the download fails, check free disk space with `df -h ~/.cache`. If the process exits with `Killed`, check the kernel log with `journalctl -k -b | grep -Ei 'out of memory|oom|killed process' | tail`. When reporting a problem, include `deck-generation.log` or the last lines of the error, the output of `free -h`, and the perplexity log if you ran that test.

The loader in `deck_qwen.py` builds the Qwen3 model structure without allocating the original 8 billion dense weights. Factor signs stay packed between layers and are unpacked for each matrix multiplication. Bit decoding, row padding, and the command-line interface have been checked. A physical Steam Deck OLED completed the CPU generation test with four threads. The observed prompt took 9.91 seconds to produce the first token, followed by 0.218 tokens/s over 15 decode steps. Loading took 3.90 seconds and peak process RSS was 2,140 MiB. The 16-token output repeated the question and began a citation. This single prompt does not measure perplexity.

[Model card and metrics](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw) · [SamsungLabs binary factor format](https://github.com/SamsungLabs/LittleBit/blob/0ceadcfa842b2c26e14a6e00d98d47aed8e53dde/quantization/utils/binary_packer.py)

## Experimental Vulkan offload on Steam Deck OLED

Use the same Bash session, model cache, CPU PyTorch wheel and virtual environment described above. From the repository root:

```bash
git pull --ff-only
source .venv-deck/bin/activate
uv pip install 'wgpu==0.31.1'
set -o pipefail
python -u deck_qwen.py --offline --backend vulkan --threads 4 --tokens 16 2>&1 | tee deck-vulkan.log
```

This hybrid backend sends both packed matrix multiplications in each LittleBit branch (including the residual branch) to a hardware Vulkan adapter. It caches packed factors in GPU buffers. Activations and results cross the CPU/GPU boundary for each multiplication. Attention, scales, embeddings and the output head run on CPU. Dispatch and readback costs can outweigh the savings. No ROCm installation is needed.

At startup the runner prints the selected adapter after comparing shader output against a NumPy reference on known random inputs. A missing hardware adapter or a failed check stops the run. GPU upload is lazy, so the first-token time includes weight uploads; load time includes adapter setup. Process RSS excludes any driver allocations outside the process accounting. Compare the output text, token counts and decode rate at the same power limit as the CPU run. The shader accumulates in FP32 and rounds the result to the activation dtype, so greedy token choices can diverge. Physical GPU throughput and end-to-end numerical agreement are pending.

## Measured CPU results

Eight hand-written English story openings, 537 next-token predictions. Each line starts at BOS and is scored independently without EOS. The original FP32 model scored 6.9403921 perplexity. As a cross-check, Karpathy's C tokenizer and forward pass scored the same 537 targets at 6.9403928.

| Method | Actual projection bits/weight | Relative Frobenius error before training | PPL before training | PPL after 256 QAT steps |
| --- | ---: | ---: | ---: | ---: |
| Dual-SVID | 0.632 | 0.8420 | 666,465,849 | — |
| Dual-SVID + Joint-ITQ | 0.632 | 0.8226 | 509,708,126 | — |
| Dual-SVID | 0.899 | 0.7926 | 19,239,280 | 102.72 |
| Dual-SVID + Joint-ITQ | 0.899 | 0.7459 | 851,783 | 85.53 |

The two trained archives occupy 136,082 bytes each, including the NPZ container. Their parameter payload is 92,404 bytes for 260,032 original parameters (2.84 bits/weight across the entire model). A fresh x86-64 Linux process scoring the trained ITQ archive peaked at 45.19 MiB RSS. The Python path expands the projections to FP32 on load. This measurement includes the interpreter, libraries, weights and working arrays for these short prompts. Steam Deck device speed and sustained generation were not tested.

The tiny model uses 5 transformer blocks, 35 compressed attention/MLP matrices, a tied embedding and head, and FP16 scales, embeddings and norms. We use exact truncated SVD, rank-one magnitude factors and optional 50-iteration Joint-ITQ before binarization. Rank rounds down to multiples of eight with a floor of eight. The requested 0.5 target exceeds its budget on these small shapes. The mini experiment uses one factor pair per projection, clipped straight-through gradients, Adam and teacher-logit distillation on 128 teacher-sampled prefixes of up to 32 tokens, cycled twice. Both arms see the same samples and ordering. This differs from the papers' residual branch, SmoothSign, long-context training data and training schedule. One seed and a small custom evaluation set limit the comparison.

## Run the checkpoints

Python 3.12 and a Linux CPU are sufficient. All saved models and the original 512-token vocabulary are included.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
python score.py runs/itq-qat.npz tok512.bin
python score.py runs/svid-qat.npz tok512.bin
```

To repeat compression and training from the pretrained model, run the following in the same environment. The loader verifies SHA-256 for the original weights and tokenizer. Results can vary slightly with BLAS and SVD implementations.

```bash
mkdir -p inputs
curl -fL https://huggingface.co/karpathy/tinyllamas/resolve/cbffbc90324256436c13dc6b4592c82c1e716354/stories260K/stories260K.bin -o inputs/stories.bin
python experiment.py inputs/stories.bin tok512.bin --out reproduced
python train.py inputs/stories.bin tok512.bin --out reproduced
```

## Paper-scale experiment

The papers' Llama-2-7B comparison at nominal 1 BPW reports WikiText-2 perplexities of 9.08 and 8.27, respectively, with a total weight footprint of 1.3 GB for either method. For batch one at 2,048 tokens, the standard FP16 KV cache is another 1 GiB (2 x 32 layers x 2,048 x 4,096 x 2 bytes). A runtime's unpacking strategy, temporary buffers and allocator determine actual RAM use. This project contains no measured 7B peak RSS on SteamOS.

On a CUDA training host, obtain authorized access to meta-llama/Llama-2-7b-hf and install the upstream dependencies including a CUDA-compatible PyTorch, Transformers 4.51.x and tensorboard. Run `bash paper-run.sh` from this directory. It pins SamsungLabs commit 0ceadcfa842b2c26e14a6e00d98d47aed8e53dde, trains separate baseline/ITQ checkpoints and scores WikiText-2 at 2,048 tokens plus the five Table 1 zero-shot tasks. See `outputs/*/eval.log`. This runner has only been syntax-checked here. The tiny CPU reader accepts the TinyStories layout only.

## Sources

- [LittleBit, NeurIPS 2025](https://arxiv.org/abs/2506.13771)
- [LittleBit-2, ICML 2026, Table 1](https://arxiv.org/html/2603.00042v2)
- [SamsungLabs implementation and environment instructions](https://github.com/SamsungLabs/LittleBit/tree/0ceadcfa842b2c26e14a6e00d98d47aed8e53dde)
- [Andrej Karpathy's pretrained TinyStories model and tokenizer](https://huggingface.co/karpathy/tinyllamas/tree/cbffbc90324256436c13dc6b4592c82c1e716354/stories260K)
- [llama2.c reference implementation](https://github.com/karpathy/llama2.c)
