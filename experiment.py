"""CPU comparison for the legacy llama2.c stories260K checkpoint."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, expit, softmax

HASH = "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696"
TOK = "037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312"
PROJECTIONS = ("q", "k", "v", "o", "gate", "down", "up")


def checked(path, digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected file checksum: {path}")
    return data


def load_base(path):
    data = checked(path, HASH)
    dim, hidden, layers, heads, kvheads, vocab, ctx = struct.unpack_from("<7i", data)
    kv = dim * kvheads // heads
    layout = dict(emb=(vocab, dim), att=(layers, dim), q=(layers, dim, dim),
                  k=(layers, kv, dim), v=(layers, kv, dim), o=(layers, dim, dim),
                  ffn=(layers, dim), gate=(layers, hidden, dim),
                  down=(layers, dim, hidden), up=(layers, hidden, dim), final=(dim,))
    weights = {}
    offset = 28
    for name, shape in layout.items():
        arr = np.frombuffer(data, "<f4", int(np.prod(shape)), offset).reshape(shape)
        offset += arr.nbytes
        weights[name] = arr
    # The legacy format ends with two RoPE tables. Inference computes these analytically.
    assert len(data) - offset == ctx * (dim // heads) * 4
    return weights


def tokenize(path, sentences):
    raw = checked(path, TOK)
    vocab, scores = [], []
    offset = 4
    while offset < len(raw):
        score, length = struct.unpack_from("<fi", raw, offset)
        offset += 8
        vocab.append(raw[offset:offset + length])
        scores.append(score)
        offset += length
    ids = {piece: i for i, piece in enumerate(vocab)}
    for line in sentences:
        pieces = [ids[bytes([c])] for c in b" " + line.encode("ascii")]
        while len(pieces) > 1:
            choices = [(ids.get(vocab[a] + vocab[b]), i)
                       for i, (a, b) in enumerate(zip(pieces, pieces[1:]))]
            pairs = [(scores[j], -i, j) for j, i in choices if j is not None]
            if not pairs:
                break
            _, neg, merged = max(pairs)
            pieces[-neg:-neg + 2] = [merged]
        yield [1, *pieces]


def latent_factors(weight, rank, itq, rng):
    u, s, v = np.linalg.svd(weight, full_matrices=False)
    a = u[:, :rank] * np.sqrt(s[:rank])
    b = v[:rank].T * np.sqrt(s[:rank])
    if itq:
        x = np.vstack([a, b])
        rotation, _ = np.linalg.qr(rng.standard_normal((rank, rank)))
        for _ in range(50):
            left, _, right = np.linalg.svd(x.T @ np.sign(x @ rotation))
            rotation = left @ right
        a, b = a @ rotation, b @ rotation

    def scales(f):
        left, singular, right = np.linalg.svd(np.abs(f), full_matrices=False)
        return np.abs(left[:, 0]), singular[0] * np.abs(right[0])

    row, la = scales(a)
    col, lb = scales(b)
    return (a / np.maximum(row[:, None] * la, 1e-10),
            b / np.maximum(col[:, None] * lb, 1e-10), row, col, la * lb)


def pack(a, b, row, col, latent):
    return (np.packbits(a >= 0), np.packbits(b >= 0),
            row.astype("f2"), col.astype("f2"), latent.astype("f2"))


def unpack(parts):
    aa, bb, row, col, latent = parts
    def signs(bits, n):
        return (2 * np.unpackbits(bits, count=n * latent.size).astype("f4") - 1).reshape(n, -1)
    return ((signs(aa, row.size) * latent * row[:, None]) @ signs(bb, col.size).T) * col


def compress(base, method, budget, seed, path):
    rng = np.random.default_rng(seed)
    arrays = {k: v.astype("f2") for k, v in base.items() if k not in PROJECTIONS}
    size = count = 0
    err = energy = 0.
    for name in PROJECTIONS:
        for i, w in enumerate(base[name]):
            m, n = w.shape
            rank = max(8, 8 * int((budget*m*n - 16*(m+n)) // (8*(m+n+16))))
            rank = min(rank, m, n)
            parts = pack(*latent_factors(w, rank, method == "itq", rng))
            for j, data in enumerate(parts):
                arrays[f"{name}_{i}_{j}"] = data
                size += data.nbytes
            count += w.size
            err += np.square(w - unpack(parts)).sum(dtype="f8")
            energy += np.square(w).sum(dtype="f8")
    np.savez(path, **arrays)
    return dict(projection_bpw=8*size/count, relative_frobenius=(err/energy)**0.5,
                model_payload_bytes=sum(v.nbytes for v in arrays.values()),
                archive_bytes=path.stat().st_size)


def reload_model(path):
    with np.load(path, allow_pickle=False) as f:
        w = {k: f[k].astype("f4") for k in ("emb", "att", "ffn", "final")}
        for k in PROJECTIONS:
            w[k] = np.stack([unpack([f[f"{k}_{i}_{j}"] for j in range(5)])
                             for i in range(len(w["att"]))])
    return w


def rms(x, scale):
    return x * (np.mean(x*x, axis=-1, keepdims=True) + 1e-5)**-0.5 * scale


def logits(w, ids):
    t = len(ids)
    x = w["emb"][ids]
    theta = np.arange(t)[:, None, None] * 10000.**(-np.arange(0, 8, 2)/8)
    cos, sin = np.cos(theta).astype("f4"), np.sin(theta).astype("f4")
    def rope(y):
        z = y.reshape(t, -1, 8)
        a, b = z[..., ::2], z[..., 1::2]
        return np.stack((a*cos-b*sin, a*sin+b*cos), axis=-1).reshape(z.shape)
    mask = np.triu(np.ones((t,t), dtype=bool), 1)
    for i, norm in enumerate(w["att"]):
        h = rms(x, norm)
        q = rope(h @ w["q"][i].T)
        k = rope(h @ w["k"][i].T).repeat(2, axis=1)
        v = (h @ w["v"][i].T).reshape(t,4,8).repeat(2, axis=1)
        attention = np.einsum("thd,shd->hts", q, k) / np.sqrt(np.float32(8))
        attention[:, mask] = -np.inf
        y = np.einsum("hts,shd->thd", softmax(attention, axis=-1), v)
        x = x + y.reshape(t,64) @ w["o"][i].T
        h = rms(x, w["ffn"][i])
        gate = h @ w["gate"][i].T
        x = x + ((gate * expit(gate)) * (h @ w["up"][i].T)) @ w["down"][i].T
    return rms(x, w["final"]) @ w["emb"].T


def evaluate(w, texts):
    total, count = 0., 0
    for ids in texts:
        pred = logits(w, ids[:-1])
        total += (logsumexp(pred, axis=1) - pred[np.arange(len(pred)), ids[1:]]).sum(dtype="f8")
        count += len(pred)
    return dict(tokens=count, perplexity=float(np.exp(total/count)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("weights", type=Path)
    p.add_argument("tokenizer", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs"))
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    base = load_base(args.weights)
    texts = list(tokenize(args.tokenizer, Path(__file__).with_name("fixture.txt").read_text().splitlines()))
    report = {"fp32": evaluate(base, texts)}
    for budget in (0.5, 1.0):
        for method in ("svid", "itq"):
            name = f"{method}-{budget}"
            path = args.out / (name + ".npz")
            report[name] = compress(base, method, budget, 42, path)
            report[name].update(evaluate(reload_model(path), texts))
    print(json.dumps(report, indent=2))
