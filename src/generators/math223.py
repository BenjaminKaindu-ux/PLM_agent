"""MATH 223 FieldSense item generator — sympy is the ground-truth oracle.

Every answer key is computed symbolically. For the 'zero' class we construct fields
that are exactly divergence-free (from a stream function) or curl-free (from a
potential), so the key is provably correct, not approximately correct.
"""

import random

import numpy as np
import sympy as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

X, Y = sp.symbols("x y")

CATEGORIES = {
    "divergence_sign": {
        "prompt": "At the marked point, the divergence of this field is:",
        "choices": ["positive", "negative", "zero"],
        "rt_threshold_s": 8.0,
    },
    "curl_sign": {
        "prompt": "At the marked point, the (z-)curl of this field is:",
        "choices": ["positive (counterclockwise)", "negative (clockwise)", "zero"],
        "rt_threshold_s": 8.0,
    },
}

_FEEDBACK = {
    ("divergence_sign", "positive"): "Arrows spread apart / grow outward near the point — net outflow, div > 0.",
    ("divergence_sign", "negative"): "Arrows converge / shrink inward near the point — net inflow, div < 0.",
    ("divergence_sign", "zero"): "Outflow balances inflow around the point — an incompressible pattern, div = 0.",
    ("curl_sign", "positive"): "The local swirl is counterclockwise — a paddle wheel here spins CCW, curl > 0.",
    ("curl_sign", "negative"): "The local swirl is clockwise — a paddle wheel here spins CW, curl < 0.",
    ("curl_sign", "zero"): "No net rotation around the point — locally irrotational, curl = 0.",
}


def _random_poly(rng: random.Random, max_terms: int = 4):
    terms = [sp.Integer(1), X, Y, X * Y, X**2, Y**2]
    picked = rng.sample(terms, k=rng.randint(2, max_terms))
    return sum(rng.choice([-2, -1, 1, 2]) * t for t in picked)


def _field_for_target(category: str, target: str, rng: random.Random):
    """Return (P, Q, px, py) whose sympy-computed value at (px, py) matches target."""
    for _ in range(200):
        if target == "zero":
            if category == "divergence_sign":
                psi = _random_poly(rng)          # stream function -> exactly div-free
                P, Q = sp.diff(psi, Y), -sp.diff(psi, X)
            else:
                phi = _random_poly(rng)          # potential -> exactly curl-free
                P, Q = sp.diff(phi, X), sp.diff(phi, Y)
        else:
            P, Q = _random_poly(rng), _random_poly(rng)
        if P == 0 and Q == 0:
            continue
        px, py = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2)
        div = sp.diff(P, X) + sp.diff(Q, Y)
        curl = sp.diff(Q, X) - sp.diff(P, Y)
        expr = div if category == "divergence_sign" else curl
        val = float(expr.subs({X: px, Y: py}))
        if target == "zero" and sp.simplify(expr) == 0:
            return P, Q, px, py
        if target == "positive" and val > 0.5:
            return P, Q, px, py
        if target == "negative" and val < -0.5:
            return P, Q, px, py
    raise RuntimeError("could not synthesize field for target")


def _render(P, Q, px, py) -> Image.Image:
    f_p = sp.lambdify((X, Y), P, "numpy")
    f_q = sp.lambdify((X, Y), Q, "numpy")
    g = np.linspace(-2, 2, 16)
    XX, YY = np.meshgrid(g, g)
    U = np.broadcast_to(np.asarray(f_p(XX, YY), dtype=float), XX.shape).copy()
    V = np.broadcast_to(np.asarray(f_q(XX, YY), dtype=float), XX.shape).copy()
    mag = np.hypot(U, V)
    cap = np.percentile(mag[mag > 0], 90) if (mag > 0).any() else 1.0
    scale = np.minimum(1.0, cap / np.maximum(mag, 1e-9))
    U, V = U * scale, V * scale

    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=110)
    ax.quiver(XX, YY, U, V, mag, cmap="viridis", width=0.004)
    ax.plot([px], [py], "o", color="red", markersize=13, markerfacecolor="none", markeredgewidth=3)
    ax.plot([px], [py], ".", color="red", markersize=5)
    ax.set_xlim(-2, 2), ax.set_ylim(-2, 2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img


def make_item(category: str, rng: random.Random | None = None, difficulty: int = 1) -> dict:
    """Generate one schema-compliant FieldSense item with a sympy-verified key."""
    rng = rng or random.Random()
    seed = rng.randint(0, 10**6)
    rng = random.Random(seed)
    spec = CATEGORIES[category]
    target = rng.choice(["positive", "negative", "zero"])
    P, Q, px, py = _field_for_target(category, target, rng)
    correct_idx = next(i for i, ch in enumerate(spec["choices"]) if ch.startswith(target))
    return {
        "id": f"math223.fieldsense.{seed:06d}",
        "course": "MATH223",
        "category": "FieldSense",
        "subcategory": category,
        "stimulus": {"type": "pil_image", "image": _render(P, Q, px, py)},
        "prompt": spec["prompt"],
        "choices": spec["choices"],
        "correct": correct_idx,
        "feedback": _FEEDBACK[(category, target)],
        "ground_truth_method": f"sympy: F=({P}, {Q}); evaluated at ({px:.2f},{py:.2f})",
        "difficulty": difficulty,
        "transfer": False,
        "provenance": {"generator": "template_fieldsense_v1", "seed": seed},
    }
