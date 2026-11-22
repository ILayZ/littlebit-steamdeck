"""CPU and hybrid Vulkan runner for the pinned LittleBitLLM Qwen3-8B checkpoint."""
import argparse
import json
import math
import resource
import time
from pathlib import Path

MODEL = "LittleBitLLM/littlebit-qwen3-8b-0.55bpw"
REV = "6b2c47aa9707e624e2e737d16d64a66be8090ae9"


def signs(words, shape):
    import numpy as np

    rows, cols = shape
    words = np.asarray(words, dtype="<i4", order="C")
    if words.shape != (rows, (cols + 31) // 32):
        raise ValueError(f"Invalid packed shape {words.shape} for {shape}")
    bits = np.unpackbits(words.view(np.uint8), axis=1, bitorder="little")[:, :cols]
    return 1 - 2 * bits.astype(np.int8)


def load(path, gpu=None):
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    class PackedLinear(torch.nn.Module):
        def __init__(self, parts, original):
            super().__init__()
            self.parts = []
            for suffix in ("", "_R"):
                def get(name):
                    return parts.pop(name + suffix)
                def packed(name):
                    key = name + suffix
                    return parts.pop(key + "_packed"), tuple(parts.pop(key + "_shape").tolist())
                v, vs = packed("V")
                u, us = packed("U")
                if (vs[1], us[0], vs[0]) != (original.in_features, original.out_features, us[1]):
                    raise ValueError("Incompatible projection dimensions")
                self.parts.append((v, vs, u, us, get("v2"), get("v1") * get("u2"), get("u1")))
            self.bias = parts.pop("bias", None)
            unexpected = [key for key in parts if not key.startswith("_")]
            if unexpected:
                raise ValueError(f"Unknown projection tensors: {unexpected}")

        def forward(self, x):
            result = None
            for v, vs, u, us, col, latent, row in self.parts:
                def linear(value, words, shape):
                    if gpu is not None:
                        return gpu.linear(value, words, shape)
                    weight = torch.from_numpy(signs(words.numpy(), shape)).to(value.dtype)
                    return torch.nn.functional.linear(value, weight)
                y = linear(x * col, v, vs)
                y = linear(y * latent, u, us) * row
                result = y if result is None else result + y
            return result if self.bias is None else result + self.bias

    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError("Expected Qwen3")
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa")
    with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
        state = {key: f.get_tensor(key) for key in f.keys()}
    names = [k.removesuffix(".U_packed") for k in state if k.endswith(".U_packed")]
    if len(names) != 7 * config.num_hidden_layers:
        raise ValueError(f"Unexpected number of packed projections: {len(names)}")
    for name in names:
        parent, attr = name.rsplit(".", 1)
        prefix = name + "."
        parts = {k[len(prefix):]: state.pop(k) for k in list(state) if k.startswith(prefix)}
        module = model.get_submodule(parent)
        setattr(module, attr, PackedLinear(parts, getattr(module, attr)))
    model.load_state_dict(state, assign=True, strict=True)
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Checkpoint has unresolved tensors")
    return model.eval()


def benchmark_prompt(tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of Spain?"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")


def benchmark(model, tokenizer, count):
    prompt = benchmark_prompt(tokenizer)
    start = time.perf_counter()
    out = model(prompt, use_cache=True, logits_to_keep=1)
    token = out.logits[:, -1].argmax(-1, keepdim=True)
    ids = [token.item()]
    first = time.perf_counter() - start
    start = time.perf_counter()
    while len(ids) < count and ids[-1] != tokenizer.eos_token_id:
        out = model(token, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        ids.append(token.item())
    elapsed = time.perf_counter() - start
    return dict(prompt_tokens=prompt.shape[1], output_tokens=len(ids), first_token_s=first,
                decode_tokens_s=(len(ids)-1)/elapsed if len(ids)>1 else None,
                text=tokenizer.decode(ids, skip_special_tokens=True))


def logit_trace(model, prompt, limit, eos_id, forced=None):
    """Record next-token scores, feeding either CPU choices or their replay."""
    import numpy as np

    current, cache = prompt, None
    scores, chosen = [], []
    for i in range(limit):
        output = model(current, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        row = output.logits[0, -1].float().numpy().copy()
        if not np.isfinite(row).all():
            raise ValueError(f"Non-finite logits at output position {i + 1}")
        scores.append(row)
        token = int(row.argmax()) if forced is None else forced[i]
        chosen.append(token)
        if token == eos_id:
            break
        current = prompt.new_tensor([[token]])
    return scores, chosen


def compare_position(cpu, vulkan, tokenizer, position):
    import numpy as np

    def best(row):
        token = int(row.argmax())
        second = np.partition(row, -2)[-2]
        return {"id": token, "text": tokenizer.decode([token]),
                "top2_gap": float(row[token] - second)}

    delta = vulkan - cpu
    return {"position": position, "cpu": best(cpu), "vulkan": best(vulkan),
            "max_abs_logit_delta": float(np.max(np.abs(delta))),
            "rms_logit_delta": float(np.sqrt(np.mean(delta.astype(np.float64)**2)))}


def compare_backends(path, tokenizer, count):
    import gc
    from deck_vulkan import VulkanFactors

    prompt = benchmark_prompt(tokenizer)
    print("Recording CPU greedy trajectory...", flush=True)
    cpu_model = load(path)
    cpu, tokens = logit_trace(cpu_model, prompt, count, tokenizer.eos_token_id)
    del cpu_model
    gc.collect()
    gpu = VulkanFactors()
    print(f"Replaying CPU token IDs on Vulkan: {gpu.info}", flush=True)
    vulkan, _ = logit_trace(load(path, gpu), prompt, len(tokens), tokenizer.eos_token_id, tokens)
    rows = [compare_position(a, b, tokenizer, pos)
            for pos, (a, b) in enumerate(zip(cpu, vulkan, strict=True), 1)]
    mismatches = [r["position"] for r in rows if r["cpu"]["id"] != r["vulkan"]["id"]]
    return {"adapter": gpu.info, "prompt_tokens": prompt.shape[1], "positions": rows,
            "top1_matches": len(rows) - len(mismatches), "compared_tokens": len(rows),
            "first_top1_mismatch": mismatches[0] if mismatches else None}


def perplexity(model, tokenizer, limit):
    import torch
    from datasets import load_dataset
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    blocks = ids.shape[1] // 2048
    n = min(limit, blocks) if limit else blocks
    nll = 0.
    for i in range(n):
        block = ids[:, 2048*i:2048*(i+1)]
        hidden = model.model(block, use_cache=False).last_hidden_state
        for start in range(0, 2047, 64):
            end = min(start+64, 2047)
            logits = model.lm_head(hidden[:, start:end]).float()
            nll += torch.nn.functional.cross_entropy(logits[0], block[0,start+1:end+1],
                                                     reduction="sum").item()
        print(f"WikiText-2 {i+1}/{n}", flush=True)
    return dict(wikitext2_ppl=math.exp(nll/(2047*n)), blocks=n, full_test=n == blocks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cpu", "vulkan", "compare"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--ppl-blocks", type=int, help="2048-token blocks, 0 for all")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--offline", action="store_true", help="Use cached model files")
    args = p.parse_args()
    if min(args.threads,args.tokens)<1 or (args.ppl_blocks is not None and args.ppl_blocks<0):
        p.error("threads/tokens must be positive and ppl-blocks nonnegative")
    if args.backend == "compare" and args.ppl_blocks is not None:
        p.error("use --tokens for paired next-token comparison")
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(MODEL, revision=REV, local_files_only=args.offline,
                                  allow_patterns=["*.json","*.safetensors","merges.txt","vocab.txt"]))
    if args.download_only:
        print(path)
        return
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if args.backend == "compare":
        with torch.inference_mode():
            report = compare_backends(path, tokenizer, args.tokens)
        report.update(model=MODEL, revision=REV, torch=torch.__version__, threads=args.threads)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    start = time.perf_counter()
    gpu = None
    if args.backend == "vulkan":
        from deck_vulkan import VulkanFactors
        gpu = VulkanFactors()
        print(f"Vulkan factor self-test passed: {gpu.info}", flush=True)
    model = load(path, gpu)
    result = dict(model=MODEL, revision=REV, torch=torch.__version__, threads=args.threads,
                  load_s=time.perf_counter()-start)
    result.update(backend=args.backend, adapter=gpu.info if gpu else None)
    with torch.inference_mode():
        if args.ppl_blocks is None:
            result.update(benchmark(model, tokenizer, args.tokens))
        else:
            result.update(perplexity(model, tokenizer, args.ppl_blocks))
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
