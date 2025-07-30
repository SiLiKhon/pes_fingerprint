import numpy as np
import pandas as pd


ONE_SIGMA_QUANTILES = (0.158655, 0.5, 0.841345)


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
) -> pd.DataFrame:
    """
    Calculating AP@K ranking metric.

    Parameters
    ----------
    target: pd.Series
        ground truth relevance values (lower value means higher relevance)
    rankers: pd.DataFrame
        ranking variables to evaluate (lower value means higher relevance). This has to be indexed
        identically to `target`.
    k_quantile: float
        precision values will be averaged over top-ranked lists of lengths from `1` to `k`, where
        `k = ceil(len(target) * k_quantile)`
    target_errors: pd.Series | None
        per-element error values for `target`. If provided, target values will be resampled `n_resample`
        times assuming gaussian errors. This has to be indexed identically to `target`.
    random_seed: int
        randomization seed for reproducibility in resampling
    n_resample: int
        only used if `target_errors` provided; see `target_errors`
    n_random_guess: int | None
        if provided, AP@K will also be calculated for this many random guess rankers

    Returns
    -------
    pd.DataFrame
        Data frame with `mean`, `err` and quantile columns with one row for each input ranker variable.
        Note that `err` values correspond to the standard error of mean from the target resampling
        procedure. This is only meant to be used for checking that `n_resample` is large enough. Use
        bootstrapping to estimate statistical error related to the limited sample size.

    """

    rng = np.random.default_rng(random_seed)
    resample_rng = np.random.default_rng(rng.integers(2**31))
    guess_rng = np.random.default_rng(rng.integers(2**31))

    if len(target) != len(rankers):
        raise ValueError(f"Expected len(target) == len(rankers), got {len(target)=}, {len(rankers)=}")
    if target_errors is not None and len(target_errors) != len(target):
        raise ValueError(f"Expected len(target_errors) == len(target), got {len(target_errors)=}, {len(target)=}")
    if (target.index != rankers.index).any():
        raise ValueError("target and rankers indexes not aligned")

    target = target.copy()
    rankers = rankers.copy()
    if target_errors is not None:
        target_errors = target_errors.copy()
        if (target.index != target_errors.index).any():
            raise ValueError("target and target_errors indexes not aligned")

    rankers["__IDEAL_PREDICTION__"] = target
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
    ap_values_samples = (int_sizes / np.arange(1, k + 1)[None, None, :]).mean(axis=2)

    aggregations = {
        "mean": ap_values_samples.mean(axis=1),
        "err": ap_values_samples.std(axis=1) / (
            np.sqrt(ap_values_samples.shape[1] - 1) if ap_values_samples.shape[1] > 1
            else np.nan
        ),
    }
    (
        aggregations["q16"],
        aggregations["q50"],
        aggregations["q84"],
    ) = np.quantile(ap_values_samples, ONE_SIGMA_QUANTILES, axis=1)

    aggregations = pd.DataFrame(aggregations)

    if n_random_guess is not None:
        aggregations, aggregations_guess = aggregations.iloc[:-n_random_guess], aggregations.iloc[-n_random_guess:]

    aggregations.index = rankers.columns

    if n_random_guess is not None:
        guess_entry_v1 = {
            "mean": aggregations_guess["mean"].mean(),
            "err": np.nan,
        }
        (
            guess_entry_v1["q16"],
            guess_entry_v1["q50"],
            guess_entry_v1["q84"],
        ) = np.quantile(aggregations_guess["mean"], ONE_SIGMA_QUANTILES)
        aggregations.loc["__RANDOM_GUESS_V1__"] = pd.Series(guess_entry_v1)

        guess_entry_v2 = {
            "mean": ap_values_samples[-n_random_guess:].mean(),
            "err": np.nan,
        }
        (
            guess_entry_v2["q16"],
            guess_entry_v2["q50"],
            guess_entry_v2["q84"],
        ) = np.quantile(ap_values_samples[-n_random_guess:], ONE_SIGMA_QUANTILES)
        aggregations.loc["__RANDOM_GUESS_V2__"] = pd.Series(guess_entry_v2)

    return aggregations
