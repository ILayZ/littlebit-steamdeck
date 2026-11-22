"""Small end-to-end QAT experiment with distillation from the original teacher."""
import argparse
import json
from pathlib import Path
import numpy as np
import autograd.numpy as an
from autograd import value_and_grad
from autograd.extend import primitive, defvjp
from autograd.scipy.special import logsumexp
from experiment import (PROJECTIONS, evaluate, latent_factors, load_base, logits,
                        pack, reload_model, tokenize)


@primitive
def hard_sign(x):
    return np.where(x >= 0, 1., -1.)


defvjp(hard_sign, lambda ans, x: lambda g: g * (np.abs(x) <= 2))


def forward(params, base, tokens):
    def rms(x, scale):
        return x * (an.mean(x*x, axis=-1, keepdims=True) + 1e-5)**-0.5 * scale

    t = len(tokens)
    theta = np.arange(t)[:, None, None] * 10000.**(-np.arange(0, 8, 2)/8)
    def rope(h):
        z = an.reshape(h, (t,-1,8))
        even, odd = z[...,::2], z[...,1::2]
        return an.reshape(an.stack((even*np.cos(theta)-odd*np.sin(theta),
                                    even*np.sin(theta)+odd*np.cos(theta)), axis=-1), z.shape)

    x = base['emb'][tokens]
    for i in range(5):
        def linear(x, name):
            a, b, row, col, latent = params[(i,name)]
            return (((x * col) @ hard_sign(b)) * latent) @ hard_sign(a).T * row
        h = rms(x, base['att'][i])
        q = rope(linear(h,'q'))
        # Reshape GQA queries to avoid differentiating through repeated keys/values.
        q = an.reshape(q, (t,4,2,8))
        k = rope(linear(h,'k'))
        scores = an.einsum('thgd,shd->hgts', q, k) / np.sqrt(8)
        scores = scores + np.triu(np.full((t,t), -1e9), 1)
        prob = an.exp(scores - logsumexp(scores, axis=-1, keepdims=True))
        v = an.reshape(linear(h,'v'), (t,4,8))
        y = an.reshape(an.einsum('hgts,shd->thgd',prob,v), (t,64))
        x = x + linear(y,'o')
        h = rms(x, base['ffn'][i])
        g = linear(h,'gate')
        x = x + linear(g/(1+an.exp(-an.clip(g,-60,60))) * linear(h,'up'), 'down')
    return rms(x, base['final']) @ base['emb'].T


def sample_teacher(base, count, length):
    rng = np.random.default_rng(123)
    data = []
    for _ in range(count):
        ids = [1]
        while len(ids) < length:
            scores = logits(base, ids)[-1].astype(float)
            prob = np.exp(scores - logsumexp(scores))
            token = rng.choice(len(prob), p=prob)
            if token == 2:
                break
            ids.append(token)
        teacher = logits(base, ids)
        data.append((ids, np.exp(teacher-logsumexp(teacher, axis=1, keepdims=True))))
    return data


def train(base, data, itq, steps, out):
    rng = np.random.default_rng(42)
    params = {}
    for name in PROJECTIONS:
        for i, w in enumerate(base[name]):
            m,n = w.shape
            rank = max(8, 8 * int((m*n - 16*(m+n)) // (8*(m+n+16))))
            params[i,name] = list(latent_factors(w,rank,itq,rng))
    fixed = {k:v.astype('f2').astype(float) for k,v in base.items() if k not in PROJECTIONS}
    moments = {k:[np.zeros_like(a) for a in group] for k,group in params.items()}
    squares = {k:[np.zeros_like(a) for a in group] for k,group in params.items()}
    def loss(p, ids, teacher):
        z = forward(p,fixed,ids)
        return an.mean(logsumexp(z,axis=1)-an.sum(teacher*z,axis=1))
    gradient = value_and_grad(loss)
    for step in range(1,steps+1):
        ids, teacher = data[(step-1) % len(data)]
        _, grads = gradient(params,ids,teacher)
        for k, group in params.items():
            for j, a in enumerate(group):
                g = grads[k][j]
                m = moments[k][j] = .9*moments[k][j] + .1*g
                v = squares[k][j] = .999*squares[k][j] + .001*g*g
                lr = (.008, .008, .0008, .0008, .004)[j]
                group[j] = a - lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    arrays = {k:v.astype('f2') for k,v in fixed.items()}
    for (i,k), group in params.items():
        for j,a in enumerate(pack(*group)):
            arrays[f'{k}_{i}_{j}'] = a
    np.savez(out, **arrays)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('weights')
    p.add_argument('tokenizer')
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--out', type=Path, default=Path('runs'))
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    base = load_base(args.weights)
    texts = list(tokenize(args.tokenizer, Path(__file__).with_name('fixture.txt').read_text().splitlines()))
    data = sample_teacher(base, min(args.steps,128), 32)
    results = {}
    for mode in ('svid','itq'):
        path = args.out / f'{mode}-qat.npz'
        train(base,data,mode=='itq',args.steps,path)
        results[mode] = evaluate(reload_model(path),texts)
    print(json.dumps(dict(steps=args.steps, training_sequences=len(data), results=results), indent=2))
