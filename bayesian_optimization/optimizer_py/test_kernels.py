"""
test_kernels.py -- compare three CoilSetKernel implementations side by side.

Expected next to this script (plus your input.py, which provides K):
    Kernel.py        original version           -> "v1 original"   (reference)
    Kernels_v2.py    your GEMM-distance version -> "v2 yours"
    Kernel_v3.py     fused-exp version          -> "v3 exact"  and  "v3 RFF-1024" (approximate mode)

Usage:
    python test_kernels.py              # correctness + properties + speed
    python test_kernels.py --quick      # smaller speed benchmarks
    python test_kernels.py --no-speed   # skip timing
    python test_kernels.py --fit-n 1000 --skip-old
                                        # time at YOUR real training-set size (v1/v2 need several GB at N>=800,
                                        # so --skip-old times only v3; speedups are then relative to v3 exact)

Three questions are answered, one table each:
    1. CORRECTNESS : does it return the same numbers (and gradients) as v1?
    2. PROPERTIES  : permutation-invariant, PSD, unit diagonal, right shapes?
    3. SPEED       : how long does it take, relative to v1?
"""
import argparse
import gc
import importlib
import time
import warnings

import torch

from input import K

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)
torch.manual_seed(0)

ELL, KAPPA = 0.7, 1.7      # fixed hyperparameters so all variants are evaluated at the same point
TOL = 1e-9                 # exact variants must agree with v1 to this tolerance


# ----------------------------------------------------------------------------------------
# loading the variants
# ----------------------------------------------------------------------------------------
_said = set()
_cache = {}
_present = set()   # module names whose file exists (even if its content is wrong)


def say(msg):
    """Print each diagnostic message only once."""
    if msg not in _said:
        _said.add(msg)
        print(msg)


def load(module_names):
    """Return CoilSetKernel from the first module in `module_names` that provides it, else None."""
    for name in module_names:
        if name not in _cache:
            cls = None
            try:
                mod = importlib.import_module(name)
                _present.add(name)
                cls = getattr(mod, "CoilSetKernel", None)
                if cls is None:
                    say(f"[skipped] {name}.py was found ({getattr(mod, '__file__', '?')}) but defines no "
                        f"CoilSetKernel. It most likely contains the wrong code (e.g. a test script instead "
                        f"of the kernel). Note: importing it also ran whatever code it contains.")
            except ModuleNotFoundError as e:
                if e.name != name:   # the file exists, but one of ITS imports is missing
                    say(f"[skipped] {name}.py: its import of '{e.name}' failed ({e})")
            except Exception as e:
                say(f"[skipped] {name}.py: {type(e).__name__}: {e}")
            _cache[name] = cls
        if _cache[name] is not None:
            return _cache[name]
    if not any(n in _present for n in module_names):
        say(f"[skipped] no file named {' or '.join(n + '.py' for n in module_names)} found next to this script")
    return None


CANDIDATES = [
    # label,          candidate module names (first that works),  kwargs,                approximate-by-design?
    ("v1 original",  ("Kernels",),                                 {},                     False),
    ("v2 yours",     ("Kernels_v2",),                             {},                     False),
    ("v3 exact",     ("Kernels_v3",),                 {},                     False),
    ("v3 RFF-1024",  ("Kernels_v3",),                 dict(n_features=1024),  True),
]
VARIANTS = []
for label, mods, kw, approx in CANDIDATES:
    cls = load(mods)
    if cls is not None:
        VARIANTS.append((label, cls, kw, approx))
assert VARIANTS and VARIANTS[0][0] == "v1 original", "Kernel.py (the reference) must be importable"
if not any(v[0].startswith("v3") for v in VARIANTS):
    print("WARNING: v3 was not loaded, so only v1 and v2 are compared below.\n")


def build(cls, kw, **extra):
    k = cls(n_coils=K, **kw, **extra)
    k.lengthscale = ELL
    k.kappa = KAPPA
    return k


def make(N):
    """N random configurations, flattened as [p(3), n(3)] x K per row."""
    p = torch.rand(N, K, 3) * 2.0
    n = torch.nn.functional.normalize(torch.randn(N, K, 3), dim=-1)
    return torch.cat([p, n], dim=-1).reshape(N, K * 6)


def shuffle_coils(x):
    out = x.reshape(-1, K, 6).clone()
    for i in range(out.shape[0]):
        out[i] = out[i][torch.randperm(K)]
    return out.reshape(-1, K * 6)


def row(cells, widths):
    return " | ".join(str(c).rjust(w) for c, w in zip(cells, widths))


# ----------------------------------------------------------------------------------------
# 1. CORRECTNESS vs v1
# ----------------------------------------------------------------------------------------
def correctness():
    print("\n" + "=" * 100)
    print("1. CORRECTNESS -- numerical agreement with v1 (reference). chunk_size=5 so chunking is exercised.")
    print("=" * 100)
    x1, x2 = make(23), make(17)

    def evaluate(cls, kw):
        res = {}
        for norm in (True, False):
            k = build(cls, kw, normalize=norm, chunk_size=5)
            res[("full", norm)] = k(x1, x2).to_dense().detach()
            res[("sym", norm)] = k(x1, x1).to_dense().detach()
        k = build(cls, kw, normalize=False, chunk_size=5)
        res["diag"] = k.forward(x1, x1, diag=True).detach()
        k = build(cls, kw, normalize=True, chunk_size=5)
        k.zero_grad()
        k(x1, x2).to_dense().pow(2).sum().backward()
        res["grad"] = torch.stack([k.raw_lengthscale.grad.flatten()[0], k.raw_kappa.grad.flatten()[0]])
        return res

    ref = evaluate(VARIANTS[0][1], VARIANTS[0][2])
    widths = [13, 16, 16, 14, 14, 10]
    print(row(["variant", "max|dK| normed", "max|dK| unnormed", "max|d diag|", "grad rel.err", "verdict"], widths))
    print("-" * 100)
    for label, cls, kw, approx in VARIANTS:
        r = evaluate(cls, kw)
        d_norm = max((r[("full", True)] - ref[("full", True)]).abs().max().item(),
                     (r[("sym", True)] - ref[("sym", True)]).abs().max().item())
        d_unn = max((r[("full", False)] - ref[("full", False)]).abs().max().item(),
                    (r[("sym", False)] - ref[("sym", False)]).abs().max().item())
        d_diag = (r["diag"] - ref["diag"]).abs().max().item()
        g_rel = ((r["grad"] - ref["grad"]).abs() / ref["grad"].abs()).max().item()
        if label == VARIANTS[0][0]:
            verdict = "reference"
        elif approx:
            verdict = "APPROX"          # random features: close, but not equal, by design
        else:
            verdict = "PASS" if max(d_norm, d_unn, d_diag) < TOL and g_rel < 1e-7 else "FAIL"
        print(row([label, f"{d_norm:.1e}", f"{d_unn:.1e}", f"{d_diag:.1e}", f"{g_rel:.1e}", verdict], widths))
    print(f"(exact variants must agree with v1 to < {TOL:.0e}; APPROX rows are expected to differ -- see accuracy note)")

    # extra accuracy figure for approximate variants: error of the NORMALIZED kernel on a larger set
    X = make(150)
    Kref = build(VARIANTS[0][1], VARIANTS[0][2])(X, X).to_dense().detach()
    for label, cls, kw, approx in VARIANTS:
        if approx:
            e = (build(cls, kw)(X, X).to_dense().detach() - Kref).abs()
            print(f"accuracy of {label}: mean |error| = {e.mean().item():.4f}, max |error| = {e.max().item():.4f} "
                  f"(kernel entries lie in [{Kref.min().item():.2f}, 1])")


# ----------------------------------------------------------------------------------------
# 2. PROPERTIES
# ----------------------------------------------------------------------------------------
def properties():
    print("\n" + "=" * 100)
    print("2. PROPERTIES -- invariance under coil relabeling, PSD, unit diagonal, shapes (normalized kernel, N=150)")
    print("=" * 100)
    X, Y = make(150), make(40)
    a, b = torch.rand(3, 7, K * 6), torch.rand(3, 4, K * 6)
    widths = [13, 14, 14, 14, 12, 8, 9]
    print(row(["variant", "perm. error", "min eigenvalue", "max|diag-1|", "asymmetry", "shapes", "verdict"], widths))
    print("-" * 100)
    for label, cls, kw, approx in VARIANTS:
        k = build(cls, kw)
        perm = (k(X, Y).to_dense() - k(shuffle_coils(X), shuffle_coils(Y)).to_dense()).abs().max().item()
        Kxx = k(X, X).to_dense()
        mineig = torch.linalg.eigvalsh(Kxx).min().item()
        dgerr = (Kxx.diag() - 1.0).abs().max().item()
        asym = (Kxx - Kxx.T).abs().max().item()
        shapes_ok = (tuple(k.forward(a, b).shape) == (3, 7, 4)) and (tuple(k.forward(a, a, diag=True).shape) == (3, 7))
        ok = perm < 1e-10 and mineig > -1e-8 and dgerr < 1e-10 and asym < 1e-10 and shapes_ok
        print(row([label, f"{perm:.1e}", f"{mineig:+.1e}", f"{dgerr:.1e}", f"{asym:.1e}",
                   "ok" if shapes_ok else "BAD", "PASS" if ok else "FAIL"], widths))


# ----------------------------------------------------------------------------------------
# 3. SPEED
# ----------------------------------------------------------------------------------------
def speed(quick, fit_n=None, skip_old=False):
    n_fit = fit_n or (200 if quick else 400)
    n_post = 600 if quick else 1500
    n_ts = 500 if quick else 1000
    scenarios = [(f"fit step  K(X,X) fwd+bwd, N={n_fit}", (n_fit, n_fit), True),
                 (f"posterior K({n_post} cand, {n_fit} train)", (n_post, n_fit), False),
                 (f"TS joint  K(C,C), {n_ts}x{n_ts}", (n_ts, n_ts), False)]

    timed = [v for v in VARIANTS if not (skip_old and v[0] in ("v1 original", "v2 yours"))]
    kernels = [(label, build(cls, kw)) for label, cls, kw, _ in timed]
    base_label = kernels[0][0]

    print("\n" + "=" * 100)
    print(f"3. SPEED -- seconds (speedup vs {base_label} in brackets). torch threads = {torch.get_num_threads()}. Lower is better.")
    print("=" * 100)
    if n_fit >= 800 and not skip_old:
        print(f"note: at N={n_fit}, v1/v2 can need several GB of RAM; use --skip-old if memory is limited.")

    warm = make(30)
    for _, k in kernels:                       # warm-up (allocator, BLAS threads)
        with torch.no_grad():
            k(warm, warm).to_dense()

    widths = [36] + [18] * len(kernels)
    print(row(["scenario"] + [label for label, _ in kernels], widths))
    print("-" * 100)
    for name, (n1, n2), with_grad in scenarios:
        A, B = make(n1), make(n2)
        times = []
        for _, k in kernels:
            best = float("inf")
            for _ in range(1 if n1 >= 1000 else 2):
                gc.collect()
                t0 = time.perf_counter()
                if with_grad:
                    k.zero_grad()
                    k(A, B).to_dense().sum().backward()
                else:
                    with torch.no_grad():
                        k(A, B).to_dense()
                best = min(best, time.perf_counter() - t0)
            times.append(best)
        cells = [f"{t:7.3f} ({times[0] / t:4.1f}x)" for t in times]
        print(row([name] + cells, widths))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller speed benchmarks")
    ap.add_argument("--no-speed", action="store_true", help="skip the timing section")
    ap.add_argument("--fit-n", type=int, default=None, help="training-set size for the speed section (default 400, 200 with --quick)")
    ap.add_argument("--skip-old", action="store_true", help="time only v3 (v1/v2 use a lot of RAM at large N)")
    args = ap.parse_args()

    correctness()
    properties()
    if not args.no_speed:
        speed(args.quick, args.fit_n, args.skip_old)
    print()