# src/measures/focus_measure_library.py

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Mapping, Tuple

import cv2 as cv
import numpy as np

try:
    from scipy.fftpack import dct as scipy_dct
except Exception:  # pragma: no cover
    scipy_dct = None

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------
MeasureFunc = Callable[[np.ndarray], float]
MeasureEntry = Dict[str, Any]
MeasureRegistry = Dict[str, MeasureEntry]

EPS: float = 1e-12
MEASURE_ALIASES: Dict[str, str] = {
    "Curvelet Transform Sharpness Index": "Wavelet Detail Energy (db1)",
}


# -----------------------------------------------------------------------------
# Core image helpers
# -----------------------------------------------------------------------------
def _as_gray_float(image: np.ndarray) -> np.ndarray:
    """
    Convert input to a 2D float64 grayscale array.

    Accepts:
    - 2D grayscale images
    - 3D images with trailing channel dimension (RGB-like)

    This library assumes images are already grayscale in the corrected pipeline,
    but keeps this safeguard to avoid brittle failures.
    """
    arr = np.asarray(image)

    if arr.ndim == 3:
        # simple luminance fallback
        if arr.shape[-1] == 3:
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        else:
            arr = np.mean(arr, axis=-1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {arr.shape}")

    arr = arr.astype(np.float64, copy=False)
    return arr


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def _safe_var(x: np.ndarray) -> float:
    return float(np.var(x)) if x.size else 0.0


def _as_gray_float32(image: np.ndarray) -> np.ndarray:
    return _as_gray_float(image).astype(np.float32, copy=False)


def _pad_reflect(img: np.ndarray, pad: int = 1) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")


def _conv2_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Lightweight 2D convolution using reflect padding and explicit loops over kernel.
    Avoids requiring scipy/cv2.
    """
    img = _as_gray_float(img)
    kernel = np.asarray(kernel, dtype=np.float64)

    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel size must be odd in both dimensions")

    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]

    return out


def _sobel_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kx = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float64,
    )
    ky = np.array(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=np.float64,
    )
    gx = _conv2_same(img, kx)
    gy = _conv2_same(img, ky)
    return gx, gy


def _roberts_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)

    # 2x2 conv "same" via padding
    img = _as_gray_float(img)
    padded = np.pad(img, ((0, 1), (0, 1)), mode="reflect")
    gx = (
        kx[0, 0] * padded[:-1, :-1]
        + kx[0, 1] * padded[:-1, 1:]
        + kx[1, 0] * padded[1:, :-1]
        + kx[1, 1] * padded[1:, 1:]
    )
    gy = (
        ky[0, 0] * padded[:-1, :-1]
        + ky[0, 1] * padded[:-1, 1:]
        + ky[1, 0] * padded[1:, :-1]
        + ky[1, 1] * padded[1:, 1:]
    )
    return gx, gy


def _laplacian(img: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]],
        dtype=np.float64,
    )
    return _conv2_same(img, kernel)


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx, gy = _sobel_gradients(img)
    return np.sqrt(gx ** 2 + gy ** 2)


def _fft_magnitude(img: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(_as_gray_float(img))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    return mag


def _radial_mask(shape: Tuple[int, int], inner_ratio: float = 0.0, outer_ratio: float = 1.0) -> np.ndarray:
    """
    Build a circular annulus mask in normalized radius units [0, 1].
    """
    h, w = shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = np.sqrt(cy ** 2 + cx ** 2) + EPS
    rrn = rr / rmax
    return (rrn >= inner_ratio) & (rrn <= outer_ratio)


def _quantize_image(img: np.ndarray, levels: int = 32) -> np.ndarray:
    arr = _as_gray_float(img)
    arr = arr - arr.min()
    arr = arr / (arr.max() + EPS)
    q = np.floor(arr * (levels - 1)).astype(np.int32)
    return q


def _glcm_horizontal(img: np.ndarray, levels: int = 32) -> np.ndarray:
    q = _quantize_image(img, levels=levels)
    glcm = np.zeros((levels, levels), dtype=np.float64)

    left = q[:, :-1].ravel()
    right = q[:, 1:].ravel()

    for i, j in zip(left, right):
        glcm[i, j] += 1.0

    # symmetrize
    glcm = glcm + glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def _haar_wavelet_details(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple one-level 2D Haar detail coefficients (LH, HL, HH).
    Assumes/forces even dimensions via truncation of last row/col if needed.
    """
    x = _as_gray_float(img)
    h, w = x.shape
    if h < 2 or w < 2:
        raise ValueError("Image too small for Haar wavelet details")

    if h % 2 == 1:
        x = x[:-1, :]
    if w % 2 == 1:
        x = x[:, :-1]

    # row transform
    low_r = (x[:, 0::2] + x[:, 1::2]) / 2.0
    high_r = (x[:, 0::2] - x[:, 1::2]) / 2.0

    # column transform
    ll = (low_r[0::2, :] + low_r[1::2, :]) / 2.0
    lh = (low_r[0::2, :] - low_r[1::2, :]) / 2.0
    hl = (high_r[0::2, :] + high_r[1::2, :]) / 2.0
    hh = (high_r[0::2, :] - high_r[1::2, :]) / 2.0

    return lh, hl, hh


@lru_cache(maxsize=32)
def _dct_basis(n: int) -> np.ndarray:
    """
    DCT-II orthonormal basis matrix.
    """
    C = np.zeros((n, n), dtype=np.float64)
    factor = np.pi / (2.0 * n)
    scale0 = np.sqrt(1.0 / n)
    scale = np.sqrt(2.0 / n)

    for k in range(n):
        alpha = scale0 if k == 0 else scale
        for i in range(n):
            C[k, i] = alpha * np.cos((2 * i + 1) * k * factor)

    return C


def _dct2(img: np.ndarray) -> np.ndarray:
    x = _as_gray_float(img)
    h, w = x.shape
    Ch = _dct_basis(h)
    Cw = _dct_basis(w)
    return Ch @ x @ Cw.T


# -----------------------------------------------------------------------------
# Focus measures used in labels + benchmark-critical set
# -----------------------------------------------------------------------------
def tenengrad(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
    return float(np.sum(gx ** 2 + gy ** 2))


def brenner_gradient(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    diff = img[:-2, :] - img[2:, :]
    return float(np.sum(diff ** 2))


def variance_of_laplacian(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    lap = cv.Laplacian(img, cv.CV_32F)
    return float(np.var(lap))


def sum_modified_laplacian(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    if min(img.shape) < 3:
        return 0.0
    lx = np.abs(2 * img[1:-1, 1:-1] - img[2:, 1:-1] - img[:-2, 1:-1])
    ly = np.abs(2 * img[1:-1, 1:-1] - img[1:-1, 2:] - img[1:-1, :-2])
    return float(np.sum((lx + ly) ** 2))


def normalized_variance(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    mu = float(np.mean(img))
    if mu < 1e-6:
        return 0.0
    return float(np.var(img) / mu)


def energy_of_gradient(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
    return float(np.sum(gx ** 2 + gy ** 2))


def histogram_entropy(image: np.ndarray, bins: int = 256) -> float:
    img = _as_gray_float32(image)
    hist, _ = np.histogram(img.flatten(), bins=bins, range=(0, 256))
    p = hist.astype(np.float64)
    p /= (p.sum() + EPS)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def glcm_contrast(image: np.ndarray, levels: int = 32) -> float:
    glcm = _glcm_horizontal(image, levels=levels)
    i, j = np.indices(glcm.shape)
    contrast = np.sum(((i - j) ** 2) * glcm)
    return float(contrast)


def variance_of_gradient(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.var(grad))


def fourier_transform_sharpness_index(image: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(_as_gray_float32(image)))
    return float(np.mean(np.abs(f)))


def fourier_high_frequency_energy_ratio(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    h, w = img.shape
    crow, ccol = h // 2, w // 2
    radius = int(min(crow, ccol) * 0.2)
    rr, cc = np.ogrid[:h, :w]
    low_mask = (rr - crow) ** 2 + (cc - ccol) ** 2 <= radius ** 2
    low = np.sum(mag[low_mask])
    total = np.sum(mag)
    return float((total - low) / (total + EPS))


def roberts_focus_measure(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    gx = img[1:, 1:] - img[:-1, :-1]
    gy = img[1:, :-1] - img[:-1, 1:]
    return float(np.sum(gx ** 2 + gy ** 2))


def wavelet_w(image: np.ndarray, level: int) -> float:
    try:
        import pywt
    except Exception as exc:  # pragma: no cover
        raise ImportError("pywt is required for Wavelet W measures") from exc

    img = _as_gray_float32(image)
    if min(img.shape) < 2 ** level:
        return float("nan")
    coeffs = pywt.wavedec2(img, "haar", level=level)
    detail = coeffs[-1]
    return float(np.sum(np.abs(detail[0])) + np.sum(np.abs(detail[1])) + np.sum(np.abs(detail[2])))


def wavelet_w1(image: np.ndarray) -> float:
    return wavelet_w(image, 1)


def wavelet_w2(image: np.ndarray) -> float:
    return wavelet_w(image, 2)


def wavelet_w3(image: np.ndarray) -> float:
    return wavelet_w(image, 3)


def wavelet_detail_energy_db1(image: np.ndarray) -> float:
    try:
        import pywt
    except Exception as exc:  # pragma: no cover
        raise ImportError("pywt is required for Wavelet Detail Energy (db1)") from exc

    coeffs = pywt.wavedec2(_as_gray_float32(image), "db1", level=1)
    return float(sum(np.sum(np.abs(c)) for c in coeffs[1:]))


def curvelet_transform_sharpness_index(image: np.ndarray) -> float:
    return wavelet_detail_energy_db1(image)


def dct_focus_measure(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    if scipy_dct is not None:
        coeff = scipy_dct(scipy_dct(img.T, norm="ortho").T, norm="ortho")
    else:  # pragma: no cover
        coeff = _dct2(img)
    u = np.arange(coeff.shape[0]).reshape(-1, 1)
    v = np.arange(coeff.shape[1]).reshape(1, -1)
    weight = u + v
    return float(np.sum(weight * (coeff ** 2)))


def intensity_skewness_index(image: np.ndarray) -> float:
    img = _as_gray_float32(image).reshape(-1)
    mu = np.mean(img)
    sigma = np.std(img) + EPS
    skew = np.mean(((img - mu) / sigma) ** 3)
    return float(skew)


def squared_gradient(image: np.ndarray) -> float:
    img = _as_gray_float32(image)
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
    return float(np.sum(gx ** 2 + gy ** 2))


def gradient_squared_energy(image: np.ndarray) -> float:
    return squared_gradient(image)


# -----------------------------------------------------------------------------
# Additional simple handcrafted measures (helpful interim expansion)
# -----------------------------------------------------------------------------
def mean_intensity(image: np.ndarray) -> float:
    return float(np.mean(_as_gray_float(image)))


def intensity_variance(image: np.ndarray) -> float:
    return float(np.var(_as_gray_float(image)))


def intensity_range_index(image: np.ndarray) -> float:
    img = _as_gray_float(image)
    return float(np.max(img) - np.min(img))


def maximal_intensity(image: np.ndarray) -> float:
    return float(np.max(_as_gray_float(image)))


def laplacian_energy(image: np.ndarray) -> float:
    lap = _laplacian(image)
    return float(np.sum(lap ** 2))


def mean_laplacian_abs(image: np.ndarray) -> float:
    lap = _laplacian(image)
    return float(np.mean(np.abs(lap)))


def spatial_frequency(image: np.ndarray) -> float:
    img = _as_gray_float(image)
    rf = np.diff(img, axis=0)
    cf = np.diff(img, axis=1)
    rf2 = np.mean(rf ** 2) if rf.size else 0.0
    cf2 = np.mean(cf ** 2) if cf.size else 0.0
    return float(np.sqrt(rf2 + cf2))


def modified_tenengrad(image: np.ndarray) -> float:
    gx, gy = _sobel_gradients(image)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    thr = np.mean(mag)
    return float(np.sum((mag[mag > thr]) ** 2))


def absolute_central_moment_2(image: np.ndarray) -> float:
    img = _as_gray_float(image)
    mu = np.mean(img)
    return float(np.mean(np.abs(img - mu) ** 2))


def absolute_central_moment_4(image: np.ndarray) -> float:
    img = _as_gray_float(image)
    mu = np.mean(img)
    return float(np.mean(np.abs(img - mu) ** 4))


def image_power(image: np.ndarray) -> float:
    img = _as_gray_float(image)
    return float(np.sum(img ** 2))


# -----------------------------------------------------------------------------
# Registry builder
# -----------------------------------------------------------------------------
def build_focus_measure_registry() -> MeasureRegistry:
    """
    Practical interim registry.

    This registry is intentionally designed to:
    1. fully support surrogate-label voting and LOO voting now,
    2. cover benchmark-critical operators already referenced in settings/scripts,
    3. be easy for Codex to extend/replace using exact legacy implementations later.

    All measures here follow the convention:
        larger value => sharper / better focus
    """
    registry: MeasureRegistry = {
        # Surrogate voters (required now)
        "Tenengrad": {
            "func": tenengrad,
            "maximize": True,
            "family": "gradient",
            "notes": "Required surrogate voter",
        },
        "Brenner Gradient": {
            "func": brenner_gradient,
            "maximize": True,
            "family": "gradient",
            "notes": "Required surrogate voter",
        },
        "Variance of Laplacian": {
            "func": variance_of_laplacian,
            "maximize": True,
            "family": "laplacian",
            "notes": "Required surrogate voter",
        },
        "Sum Modified Laplacian": {
            "func": sum_modified_laplacian,
            "maximize": True,
            "family": "laplacian",
            "notes": "Required surrogate voter",
        },
        "Normalized Variance": {
            "func": normalized_variance,
            "maximize": True,
            "family": "statistical",
            "notes": "Required surrogate voter",
        },
        "Energy of Gradient": {
            "func": energy_of_gradient,
            "maximize": True,
            "family": "gradient",
            "notes": "Required surrogate voter",
        },
        "Histogram Entropy": {
            "func": histogram_entropy,
            "maximize": True,
            "family": "entropy",
            "notes": "Required surrogate voter",
        },
        "GLCM Contrast": {
            "func": glcm_contrast,
            "maximize": True,
            "family": "texture",
            "notes": "Required surrogate voter",
        },
        "Variance of Gradient": {
            "func": variance_of_gradient,
            "maximize": True,
            "family": "gradient",
            "notes": "Required surrogate voter",
        },
        "Fourier Transform Sharpness Index": {
            "func": fourier_transform_sharpness_index,
            "maximize": True,
            "family": "frequency",
            "notes": "Required surrogate voter",
        },

        # Additional benchmark-critical operators
        "Fourier High Frequency Energy Ratio": {
            "func": fourier_high_frequency_energy_ratio,
            "maximize": True,
            "family": "frequency",
        },
        "Roberts Focus Measure": {
            "func": roberts_focus_measure,
            "maximize": True,
            "family": "gradient",
        },
        "Wavelet W1": {
            "func": wavelet_w1,
            "maximize": True,
            "family": "wavelet",
        },
        "Wavelet W2": {
            "func": wavelet_w2,
            "maximize": True,
            "family": "wavelet",
        },
        "Wavelet W3": {
            "func": wavelet_w3,
            "maximize": True,
            "family": "wavelet",
        },
        "Wavelet Detail Energy (db1)": {
            "func": wavelet_detail_energy_db1,
            "maximize": True,
            "family": "wavelet",
            "notes": "Legacy alias target used in place of deprecated curvelet label",
        },
        "Curvelet Transform Sharpness Index": {
            "func": curvelet_transform_sharpness_index,
            "maximize": True,
            "family": "wavelet_alias",
            "notes": "Legacy alias mapped to Wavelet Detail Energy (db1)",
        },
        "DCT Focus Measure": {
            "func": dct_focus_measure,
            "maximize": True,
            "family": "frequency",
        },
        "Intensity Skewness Index": {
            "func": intensity_skewness_index,
            "maximize": True,
            "family": "statistical",
        },
        "Squared Gradient": {
            "func": squared_gradient,
            "maximize": True,
            "family": "gradient",
        },
        "Gradient Squared Energy": {
            "func": gradient_squared_energy,
            "maximize": True,
            "family": "gradient",
        },

        # Extra interim measures
        "Mean Intensity": {
            "func": mean_intensity,
            "maximize": True,
            "family": "statistical",
        },
        "Intensity Variance": {
            "func": intensity_variance,
            "maximize": True,
            "family": "statistical",
        },
        "Intensity Range Index": {
            "func": intensity_range_index,
            "maximize": True,
            "family": "statistical",
        },
        "Maximal Intensity": {
            "func": maximal_intensity,
            "maximize": True,
            "family": "statistical",
        },
        "Laplacian Energy": {
            "func": laplacian_energy,
            "maximize": True,
            "family": "laplacian",
        },
        "Mean Absolute Laplacian": {
            "func": mean_laplacian_abs,
            "maximize": True,
            "family": "laplacian",
        },
        "Spatial Frequency": {
            "func": spatial_frequency,
            "maximize": True,
            "family": "gradient",
        },
        "Modified Tenengrad": {
            "func": modified_tenengrad,
            "maximize": True,
            "family": "gradient",
        },
        "Absolute Central Moment 2": {
            "func": absolute_central_moment_2,
            "maximize": True,
            "family": "statistical",
        },
        "Absolute Central Moment 4": {
            "func": absolute_central_moment_4,
            "maximize": True,
            "family": "statistical",
        },
        "Image Power": {
            "func": image_power,
            "maximize": True,
            "family": "statistical",
        },
    }

    return registry


def get_focus_measure_registry() -> MeasureRegistry:
    return build_focus_measure_registry()


FOCUS_MEASURE_REGISTRY: MeasureRegistry = build_focus_measure_registry()


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------
def list_focus_measure_names() -> Tuple[str, ...]:
    return tuple(FOCUS_MEASURE_REGISTRY.keys())


def canonicalize_measure_name(name: str) -> str:
    return MEASURE_ALIASES.get(name, name)


def get_focus_measure(name: str) -> MeasureFunc:
    name = canonicalize_measure_name(name)
    if name not in FOCUS_MEASURE_REGISTRY:
        raise KeyError(f"Unknown focus measure: {name}")
    return FOCUS_MEASURE_REGISTRY[name]["func"]


def get_measure_entry(name: str) -> MeasureEntry:
    name = canonicalize_measure_name(name)
    if name not in FOCUS_MEASURE_REGISTRY:
        raise KeyError(f"Unknown focus measure: {name}")
    return FOCUS_MEASURE_REGISTRY[name]


def validate_registry(required_names: Tuple[str, ...] | None = None) -> None:
    if required_names is None:
        required_names = tuple()

    missing = [
        name
        for name in required_names
        if canonicalize_measure_name(name) not in FOCUS_MEASURE_REGISTRY
    ]
    if missing:
        raise ValueError(f"Missing required focus measures in registry: {missing}")


__all__ = [
    "MeasureFunc",
    "MeasureEntry",
    "MeasureRegistry",
    "build_focus_measure_registry",
    "get_focus_measure_registry",
    "FOCUS_MEASURE_REGISTRY",
    "list_focus_measure_names",
    "canonicalize_measure_name",
    "get_focus_measure",
    "get_measure_entry",
    "validate_registry",
]
