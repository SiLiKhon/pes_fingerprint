import numpy as np
import pandas as pd


def _calc_intersection_sizes(
    x_target: np.ndarray,
    x_prediction: np.ndarray,
    k_max: int | None = None,
) -> np.ndarray:
    if x_target.ndim < 1:
        raise ValueError(f"Expected an array, got {x_target.ndim=} instead")
    if x_target.size <= 1:
        raise ValueError(f"Not enough elements in array: {x_target.size=}")

    idx_target = x_target.argsort(-1)
    idx_prediction = np.take_along_axis(x_prediction, idx_target, -1).argsort(-1)

    if k_max is None:
        k_max = idx_prediction.shape[-1]
    if k_max <= 0 or k_max > idx_prediction.shape[-1]:
        raise ValueError(f"Expected `k_max` in [1, {idx_prediction.shape[-1]}], fot {k_max=} instead")

    # Can (and should) this be vectorized?
    return np.concatenate(
        [
            (idx_prediction[..., :i] < i).sum(axis=-1, keepdims=True)
            for i in range(1, k_max + 1)
        ],
        axis=-1
    )


def average_precision_at_k(
    target: pd.Series,
    rankers: pd.DataFrame,
    *,
    k_quantile: float = 0.1,
    target_errors: pd.Series | None = None,
    random_seed: int = 42,
    n_resample: int = 100,
    n_random_guess: int | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(random_seed)
    resample_rng = np.random.default_rng(rng.integers(2**31))
    guess_rng = np.random.default_rng(rng.integers(2**31))

    if len(target) != len(rankers):
        raise ValueError(f"Expected len(target) == len(rankers), got {len(target)=}, {len(rankers)=}")
    if target_errors is not None and len(target_errors) != len(target):
        raise ValueError(f"Expected len(target_errors) == len(target), got {len(target_errors)=}, {len(target)=}")

    k = int(np.ceil(len(target) * k_quantile))

    target_array = target.to_numpy()[None, None, :]
    if target_errors is not None:
        target_array = (
            target_array
            + resample_rng.normal(
                size=(1, n_resample, len(target))
            ) * target_errors.to_numpy()[None, None, :]
        )

    rankers_array = rankers.to_numpy().T[:, None, :]
    if n_random_guess is not None:
        guesses = guess_rng.uniform(size=(n_random_guess, 1, len(target)))
        rankers_array = np.concatenate([rankers_array, guesses], axis=0)

    int_sizes = _calc_intersection_sizes(target_array, rankers_array, k)
    ap_values = (int_sizes / np.arange(1, k + 1)[None, None, :]).mean(axis=(1, 2))

    if n_random_guess is not None:
        ap_values, ap_values_guess = ap_values[:-n_random_guess], ap_values[-n_random_guess:]

    result = {
        col: float(ap_val)
        for col, ap_val in zip(rankers.columns, ap_values, strict=True)
    }

    if n_random_guess is not None:
        for q in [0.158655, 0.5, 0.841345]:
            q_int = int(np.round(q * 100))
            key = f"RANDOM_GUESS_Q{q_int:02d}"
            assert key not in result
            result[key] = np.quantile(ap_values_guess, q)

    return result
