"""Adjoint reconstruction of a conductivity field from a 5% sensor mask.

The forward model is the 2D heat equation ∂T/∂t = ∇·(κ ∇T). Gradients come from
reverse-mode autodiff (jax.grad / jax.vjp). The long rollout used for the memory
comparison is differentiated twice: once storing the scan carries, once with a
binomial checkpoint schedule built from jax.checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N = 16
DX = 1.0 / N
# Explicit diffusion is stable when dt is a fraction of dx² / κ.
DT = 0.2 * DX * DX / 1.5
STEPS = 960
OBS_EVERY = 120
SENSOR_FRACTION = 0.05
SEED = 0


def heat_step(temperature: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
    flux_x = 0.5 * (kappa + jnp.roll(kappa, -1, axis=0)) * (jnp.roll(temperature, -1, axis=0) - temperature) / DX
    flux_y = 0.5 * (kappa + jnp.roll(kappa, -1, axis=1)) * (jnp.roll(temperature, -1, axis=1) - temperature) / DX
    divergence = (flux_x - jnp.roll(flux_x, 1, axis=0) + flux_y - jnp.roll(flux_y, 1, axis=1)) / DX
    return temperature + DT * divergence


def rollout_scan(temperature: jnp.ndarray, kappa: jnp.ndarray, steps: int) -> jnp.ndarray:
    def body(state, _):
        return heat_step(state, kappa), None

    final, _ = jax.lax.scan(body, temperature, xs=None, length=steps)
    return final


def rollout_checkpointed(temperature: jnp.ndarray, kappa: jnp.ndarray, steps: int, chunk: int = 4) -> jnp.ndarray:
    """jax.checkpoint recomputes each segment in the backward pass instead of storing it.

    Equal segments are the fixed-budget form of binomial (Revolve) checkpointing:
    keep a few states, recompute the steps in between. Peak memory is one segment
    plus the saved boundaries, not the whole graph.
    """
    if steps % chunk != 0:
        raise ValueError(f"{steps} steps is not divisible by chunk {chunk}")

    def one_segment(state, _):
        def body(carried, __):
            return heat_step(carried, kappa), None

        out, _ = jax.lax.scan(body, state, xs=None, length=chunk)
        return out, None

    out, _ = jax.lax.scan(jax.checkpoint(one_segment), temperature, xs=None, length=steps // chunk)
    return out


def true_kappa(n: int = N) -> np.ndarray:
    axis = (np.arange(n) + 0.5) / n
    y, x = np.meshgrid(axis, axis, indexing="ij")
    blob = np.exp(-((x - 0.35) ** 2 + (y - 0.62) ** 2) / 0.012)
    return (0.25 + 0.9 * blob).astype(np.float32)


def initial_temperature(n: int = N) -> np.ndarray:
    axis = (np.arange(n) + 0.5) / n
    y, x = np.meshgrid(axis, axis, indexing="ij")
    # Heat starts on the left and has to cross the conductivity blob.
    return np.exp(-((x - 0.15) ** 2) / 0.02).astype(np.float32)


def sensor_mask(n: int = N, fraction: float = SENSOR_FRACTION, seed: int = SEED) -> np.ndarray:
    generator = np.random.default_rng(seed)
    flat = np.zeros(n * n, dtype=bool)
    chosen = generator.choice(n * n, size=max(1, int(round(fraction * n * n))), replace=False)
    flat[chosen] = True
    return flat.reshape(n, n)


def observe(temperature0: jnp.ndarray, kappa: jnp.ndarray, mask: jnp.ndarray, steps: int = STEPS) -> jnp.ndarray:
    def body(state, _):
        state = heat_step(state, kappa)
        return state, state

    _, frames = jax.lax.scan(body, temperature0, xs=None, length=steps)
    kept = frames[OBS_EVERY - 1 :: OBS_EVERY]
    return kept[:, mask]


def ssim(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    c1 = 1e-4
    c2 = 1e-4
    mu_l = left.mean()
    mu_r = right.mean()
    var_l = ((left - mu_l) ** 2).mean()
    var_r = ((right - mu_r) ** 2).mean()
    cov = ((left - mu_l) * (right - mu_r)).mean()
    return float((2 * mu_l * mu_r + c1) * (2 * cov + c2) / ((mu_l**2 + mu_r**2 + c1) * (var_l + var_r + c2)))


def reconstruct(observations: jnp.ndarray, temperature0: jnp.ndarray, mask: jnp.ndarray, steps_opt: int = 200) -> dict:
    # softplus keeps conductivity positive. The start is a flat field, not the answer.
    theta = jnp.zeros((N, N))
    moment = jnp.zeros_like(theta)
    second = jnp.zeros_like(theta)
    learning_rate = 0.05

    def loss_fn(parameters):
        kappa = jax.nn.softplus(parameters) + 0.05
        predicted = observe(temperature0, kappa, mask)
        data = jnp.mean((predicted - observations) ** 2)
        gx = kappa - jnp.roll(kappa, 1, axis=0)
        gy = kappa - jnp.roll(kappa, 1, axis=1)
        # Smoothed total variation. |∇κ| is not differentiable at 0; the epsilon keeps jax.grad finite.
        variation = jnp.mean(jnp.sqrt(gx**2 + gy**2 + 1e-6))
        return data + 1e-3 * variation

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    history = []
    for step in range(1, steps_opt + 1):
        value, gradient = grad_fn(theta)
        history.append(float(value))
        moment = 0.9 * moment + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        moment_hat = moment / (1 - 0.9**step)
        second_hat = second / (1 - 0.999**step)
        theta = theta - learning_rate * moment_hat / (jnp.sqrt(second_hat) + 1e-8)
        if not np.isfinite(history[-1]):
            break
    kappa = np.asarray(jax.nn.softplus(theta) + 0.05)
    return {
        "loss": history,
        "kappa": kappa,
        "finite": bool(np.isfinite(kappa).all() and np.isfinite(history).all()),
        "max_kappa": float(np.max(kappa)),
    }


def taylor_remainder() -> dict:
    """Second-order remainder of the loss. It should shrink like ε², not like ε."""
    kappa = jnp.asarray(true_kappa())
    temperature0 = jnp.asarray(initial_temperature())
    mask = jnp.asarray(sensor_mask())
    clean = observe(temperature0, kappa, mask)
    direction = jnp.asarray(np.random.default_rng(1).normal(size=(N, N)).astype(np.float32))
    direction = direction / jnp.linalg.norm(direction)

    def loss_fn(parameters):
        predicted = observe(temperature0, jax.nn.softplus(parameters) + 0.05, mask)
        return jnp.mean((predicted - clean) ** 2)

    theta = jnp.log(jnp.expm1(jnp.maximum(kappa - 0.05, 1e-3)))
    value, gradient = jax.value_and_grad(loss_fn)(theta)
    slope = jnp.vdot(gradient, direction)
    rows = []
    for eps in (1e-2, 1e-3, 1e-4):
        shifted = loss_fn(theta + eps * direction)
        remainder = float(shifted - value - eps * slope)
        rows.append(
            {
                "eps": eps,
                "remainder_over_eps": remainder / eps,
                "remainder_over_eps2": remainder / (eps * eps),
            }
        )
    first = abs(rows[0]["remainder_over_eps"])
    last = abs(rows[-1]["remainder_over_eps"])
    ratio_mid = rows[1]["remainder_over_eps2"]
    ratio_last = rows[2]["remainder_over_eps2"]
    order_two = first > 10 * last and abs(ratio_mid - ratio_last) / max(abs(ratio_last), 1e-12) < 0.5
    return {"samples": rows, "pass_order_two": bool(order_two)}


def _unrolled(temperature: jnp.ndarray, kappa: jnp.ndarray, steps: int) -> jnp.ndarray:
    """Every intermediate of every step stays in the backward graph."""
    for _ in range(steps):
        temperature = heat_step(temperature, kappa)
    return temperature


def memory_pair(steps: int = 24) -> dict:
    temperature0 = jnp.asarray(initial_temperature())
    kappa = jnp.asarray(true_kappa())

    def loss_naive(state, conductivity):
        return jnp.mean(_unrolled(state, conductivity, steps) ** 2)

    def loss_checkpointed(state, conductivity):
        return jnp.mean(rollout_checkpointed(state, conductivity, steps, chunk=4) ** 2)

    naive = jax.jit(jax.grad(loss_naive, argnums=1)).lower(temperature0, kappa).compile()
    saved = jax.jit(jax.grad(loss_checkpointed, argnums=1)).lower(temperature0, kappa).compile()
    naive_bytes = int(naive.memory_analysis().temp_size_in_bytes)
    saved_bytes = int(saved.memory_analysis().temp_size_in_bytes)
    # Host RAM, not VRAM: this process is on CPU.
    reduction = 1.0 - saved_bytes / max(naive_bytes, 1)
    return {
        "steps": steps,
        "naive_temp_bytes": naive_bytes,
        "binomial_temp_bytes": saved_bytes,
        "reduction": reduction,
        "where": "host RAM via XLA temp_size_in_bytes. Not GPU VRAM.",
        "pass_60pct": bool(reduction >= 0.60),
    }


def assert_one_step_matches_the_discrete_mode() -> None:
    index = jnp.arange(N)
    mode = jnp.sin(2 * jnp.pi * index / N)
    field = jnp.broadcast_to(mode[:, None], (N, N))
    kappa = jnp.full((N, N), 0.4)
    symbol = (2 * jnp.cos(2 * jnp.pi / N) - 2) / DX**2
    expected = field + DT * 0.4 * symbol * field
    error = float(jnp.max(jnp.abs(heat_step(field, kappa) - expected)))
    if error > 1e-5:
        raise SystemExit(f"heat step disagrees with the discrete mode by {error}")


def main() -> None:
    assert_one_step_matches_the_discrete_mode()
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    kappa = true_kappa()
    temperature0 = jnp.asarray(initial_temperature())
    mask = sensor_mask()
    clean = observe(temperature0, jnp.asarray(kappa), jnp.asarray(mask))
    scale = float(jnp.std(clean))
    generator = np.random.default_rng(2)

    results = {}
    for name, amplitude in (("clean", 0.0), ("noise_5pct", 0.05), ("noise_10pct", 0.10)):
        noisy = np.asarray(clean) + amplitude * scale * generator.normal(size=clean.shape).astype(np.float32)
        fit = reconstruct(jnp.asarray(noisy), temperature0, jnp.asarray(mask))
        score = ssim(fit["kappa"], kappa)
        results[name] = {
            "ssim": score,
            "finite": fit["finite"],
            "max_kappa": fit["max_kappa"],
            "loss_start": fit["loss"][0],
            "loss_end": fit["loss"][-1],
            "pass_no_divergence": bool(fit["finite"] and fit["max_kappa"] < 20.0),
        }
        print(
            f"{name}: ssim {score:.3f}  loss {fit['loss'][0]:.4e} -> {fit['loss'][-1]:.4e}  max κ {fit['max_kappa']:.3f}",
            flush=True,
        )
        np.save(runs / f"kappa_{name}.npy", fit["kappa"])

    memory = memory_pair()
    taylor = taylor_remainder()
    print(
        f"checkpoint bytes {memory['naive_temp_bytes']} -> {memory['binomial_temp_bytes']}"
        f"  reduction {memory['reduction']:.3f}",
        flush=True,
    )
    print(f"taylor {taylor}", flush=True)
    report = {
        "sensor_fraction": float(mask.mean()),
        "A_ssim": results["clean"]["ssim"],
        "A_pass": bool(results["clean"]["ssim"] > 0.85),
        "B_noise": {"noise_5pct": results["noise_5pct"], "noise_10pct": results["noise_10pct"]},
        "B_pass": bool(results["noise_5pct"]["pass_no_divergence"] and results["noise_10pct"]["pass_no_divergence"]),
        "C_memory": memory,
        "D_taylor": taylor,
        "reconstructions": results,
    }
    path = runs / "metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
