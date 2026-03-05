from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mne.filter import filter_data, notch_filter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import EEGConfig, MentalCommandModelConfig


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_block(block: np.ndarray, eeg_cfg: EEGConfig, sfreq: float) -> np.ndarray:
    """Wide bandpass + optional notch for a continuous segment.

    Uses IIR (4th-order Butterworth, zero-phase) so that edge effects are
    negligible even on segments as short as ~2-3 s.

    block shape: (n_ch, n_samples).
    """
    Xf = np.asarray(block, dtype=np.float64, order="C")
    if eeg_cfg.notch is not None:
        Xf = notch_filter(Xf, Fs=sfreq, freqs=[float(eeg_cfg.notch)], verbose="ERROR")
    Xf = filter_data(
        Xf,
        sfreq=float(sfreq),
        l_freq=float(eeg_cfg.l_freq),
        h_freq=float(eeg_cfg.h_freq),
        method="iir",
        iir_params=dict(order=4, ftype="butter", output="sos"),
        verbose="ERROR",
    )
    return Xf.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def split_windows(
    block: np.ndarray,
    sfreq: float,
    window_s: float,
    step_s: float,
) -> np.ndarray:
    """Split one continuous block into fixed windows.

    block shape: (n_ch, n_samples).
    returns shape: (n_windows, n_ch, n_window_samples).
    """
    n_ch, n_samples = block.shape
    w = int(round(float(window_s) * float(sfreq)))
    h = int(round(float(step_s) * float(sfreq)))
    if w <= 0 or h <= 0:
        raise ValueError("window_s and step_s must produce at least one sample")
    if n_samples < w:
        return np.empty((0, n_ch, w), dtype=np.float32)
    starts = np.arange(0, n_samples - w + 1, h, dtype=int)
    out = np.empty((len(starts), n_ch, w), dtype=np.float32)
    for i, s in enumerate(starts):
        out[i] = block[:, s : s + w]
    return out


# ---------------------------------------------------------------------------
# Sub-band helper (shared by both feature extractors)
# ---------------------------------------------------------------------------

def _iir_bandpass(X: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase IIR bandpass on an (n_windows, n_ch, n_samples) array.

    Reshapes to 2-D so that ``mne.filter.filter_data`` always receives a
    simple (rows, time) matrix — guaranteed safe for the IIR path.
    """
    X64 = np.asarray(X, dtype=np.float64)
    orig_shape = X64.shape
    if X64.ndim == 3:
        X64 = X64.reshape(-1, orig_shape[-1])
    filtered = filter_data(
        X64,
        sfreq=float(sfreq),
        l_freq=float(lo),
        h_freq=float(hi),
        method="iir",
        iir_params=dict(order=4, ftype="butter", output="sos"),
        verbose="ERROR",
    )
    return filtered.reshape(orig_shape).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """Per-band Riemannian covariance → tangent-space projection, concatenated."""

    def __init__(self, bands=((4, 8), (8, 13), (13, 30), (30, 45)),
                 sfreq: float = 300.0, cov_estimator: str = "oas"):
        self.bands = bands
        self.sfreq = sfreq
        self.cov_estimator = cov_estimator

    def fit(self, X, y=None):
        self.cov_estimators_: list[Covariances] = []
        self.ts_estimators_: list[TangentSpace] = []
        for lo, hi in self.bands:
            X_band = _iir_bandpass(X, self.sfreq, lo, hi)
            cov_est = Covariances(estimator=self.cov_estimator)
            covs = cov_est.fit_transform(X_band)
            ts = TangentSpace(metric="riemann")
            ts.fit(covs)
            self.cov_estimators_.append(cov_est)
            self.ts_estimators_.append(ts)
        return self

    def transform(self, X):
        parts = []
        for i, (lo, hi) in enumerate(self.bands):
            X_band = _iir_bandpass(X, self.sfreq, lo, hi)
            covs = self.cov_estimators_[i].transform(X_band)
            parts.append(self.ts_estimators_[i].transform(covs))
        return np.concatenate(parts, axis=1)


class LogBandPowerFeatures(BaseEstimator, TransformerMixin):
    """Log-variance (≈ log band power) per channel per sub-band."""

    def __init__(self, bands=((4, 8), (8, 13), (13, 30), (30, 45)),
                 sfreq: float = 300.0):
        self.bands = bands
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        parts = []
        for lo, hi in self.bands:
            X_band = _iir_bandpass(X, self.sfreq, lo, hi)
            log_bp = np.log(np.var(X_band, axis=-1) + 1e-10)
            parts.append(log_bp)
        return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Pipeline constructors
# ---------------------------------------------------------------------------

def _make_lr(model_cfg: MentalCommandModelConfig) -> LogisticRegression:
    return LogisticRegression(
        C=float(model_cfg.C),
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=int(model_cfg.max_iter),
        class_weight=model_cfg.class_weight,
        random_state=42,
    )


def make_fb_riemannian_classifier(
    model_cfg: MentalCommandModelConfig, sfreq: float,
) -> Pipeline:
    return Pipeline([
        ("fb_ts", FilterBankTangentSpace(
            bands=model_cfg.filter_bank_bands,
            sfreq=sfreq,
        )),
        ("scaler", StandardScaler()),
        ("clf", _make_lr(model_cfg)),
    ])


def make_bandpower_classifier(
    model_cfg: MentalCommandModelConfig, sfreq: float,
) -> Pipeline:
    return Pipeline([
        ("bp", LogBandPowerFeatures(
            bands=model_cfg.filter_bank_bands,
            sfreq=sfreq,
        )),
        ("scaler", StandardScaler()),
        ("clf", _make_lr(model_cfg)),
    ])


# ---------------------------------------------------------------------------
# Cross-validation quality estimate
# ---------------------------------------------------------------------------

@dataclass
class MCQuality:
    balanced_accuracy: float
    macro_f1: float
    n_samples: int
    n_per_class: dict[str, int]
    per_class_accuracy: dict[str, float]


def evaluate_cv_quality(
    X: np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    pipeline: Pipeline,
    cv_splits_max: int,
) -> MCQuality:
    """Block-grouped cross-validation (one registration block held out per class per fold)."""
    y = np.asarray(y, dtype=int)
    block_ids = np.asarray(block_ids, dtype=int)
    if y.shape[0] != X.shape[0] or block_ids.shape[0] != X.shape[0]:
        raise ValueError("X, y, and block_ids must have the same first dimension")

    classes, counts = np.unique(y, return_counts=True)
    class_block_ids: dict[int, np.ndarray] = {}
    for c in classes:
        class_mask = y == int(c)
        class_block_ids[int(c)] = np.sort(np.unique(block_ids[class_mask]))

    min_block_count = min(len(v) for v in class_block_ids.values())
    n_splits = min(int(cv_splits_max), int(min_block_count))
    if n_splits < 2:
        block_counts = {int(c): len(class_block_ids[int(c)]) for c in classes}
        raise ValueError(
            f"Not enough registration blocks for grouped CV: per-class blocks={block_counts}"
        )

    y_pred = np.empty_like(y)
    for k in range(n_splits):
        test_mask = np.zeros(y.shape[0], dtype=bool)
        for c in classes:
            held_out_block = class_block_ids[int(c)][k]
            test_mask |= (y == int(c)) & (block_ids == int(held_out_block))
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise ValueError(f"Invalid CV split at fold {k}")

        fold_clf = clone(pipeline)
        fold_clf.fit(X[train_mask], y[train_mask])
        y_pred[test_mask] = fold_clf.predict(X[test_mask])

    cm = confusion_matrix(y, y_pred, labels=classes)
    denom = np.sum(cm, axis=1)
    per_class_acc = {}
    for i, c in enumerate(classes):
        per_class_acc[str(int(c))] = float(cm[i, i] / denom[i]) if denom[i] > 0 else 0.0

    return MCQuality(
        balanced_accuracy=float(balanced_accuracy_score(y, y_pred)),
        macro_f1=float(f1_score(y, y_pred, average="macro")),
        n_samples=int(len(y)),
        n_per_class={str(int(c)): int(n) for c, n in zip(classes, counts)},
        per_class_accuracy=per_class_acc,
    )


# ---------------------------------------------------------------------------
# Live-mode smoother
# ---------------------------------------------------------------------------

class EMAProbSmoother:
    def __init__(self, alpha: float, n_classes: int):
        self.alpha = float(alpha)
        self.n_classes = int(n_classes)
        self._state: np.ndarray | None = None

    def update(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (self.n_classes,):
            raise ValueError(f"Expected probs shape ({self.n_classes},), got {p.shape}")
        if self._state is None:
            self._state = p.copy()
        else:
            self._state = (1.0 - self.alpha) * self._state + self.alpha * p
        s = float(np.sum(self._state))
        if s > 0:
            self._state = self._state / s
        return self._state.copy()
