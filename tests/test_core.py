"""Run:  python -m pytest -q   (or  python tests/test_core.py)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import color_hist, face_detect as fd, spatial_filter as sf  # noqa: E402


def expect_error(text):
    try:
        sf.parse_kernel(text)
    except sf.KernelError:
        return True
    return False


def test_parse_ok():
    assert sf.parse_kernel("0,0,0;0 1 0\n0 0 0".replace(";", "\n")).shape == (3, 3)


def test_parse_bad_inputs():
    assert expect_error("")
    assert expect_error("1 2\n3 4")          # even size
    assert expect_error("1 2 3\n4 5 6")      # not square
    assert expect_error("1 2 3\n4 5\n6 7 8")  # ragged
    assert expect_error("1 a 3\n4 5 6\n7 8 9")  # non numeric
    assert expect_error("nan 0 0\n0 1 0\n0 0 0")
    assert expect_error("\n".join([" ".join(["1"] * 33)] * 33))  # too large


def test_normalize():
    k, w = sf.normalize_kernel(np.ones((3, 3)), "sum")
    assert abs(k.sum() - 1) < 1e-9 and w is None
    _, w = sf.normalize_kernel(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0.0]]), "sum")
    assert w is not None


def test_psnr_known_value():
    a = np.full((8, 8, 3), 100, np.uint8)
    b = np.full((8, 8, 3), 110, np.uint8)  # MSE = 100
    assert abs(sf.psnr(a, b) - 10 * math.log10(255 ** 2 / 100)) < 1e-9
    assert math.isinf(sf.psnr(a, a))


def test_identity_kernel_is_lossless():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    out = sf.apply_convolution(img, sf.parse_kernel(sf.PRESETS["Identity 3x3"]))
    assert math.isinf(sf.psnr(img, out))


def test_extreme_kernel_saturates_without_crash():
    img = np.full((10, 10, 3), 200, np.uint8)
    out = sf.apply_convolution(img, np.full((3, 3), 1e6))
    assert out.dtype == np.uint8 and out.max() == 255


def test_yuv_white():
    white = np.full((4, 4, 3), 255, np.uint8)
    yuv = color_hist.rgb_to_yuv(white)
    assert yuv["Y"][0, 0] == 255 and abs(int(yuv["U"][0, 0]) - 128) <= 1


def test_histogram_sum():
    ch = np.random.default_rng(1).integers(0, 256, (30, 30), dtype=np.uint8)
    h = color_hist.compute_histogram(ch)
    assert h.shape == (256,) and h.sum() == 900


def test_nms_removes_overlap_keeps_separate():
    a = fd.Detection(100, 100, 100, 100, 0.9)
    b = fd.Detection(120, 110, 100, 100, 0.5)   # heavy overlap with a
    c = fd.Detection(400, 100, 100, 100, 0.7)   # separate face
    kept = fd.non_max_suppression([b, c, a], 0.3)
    assert kept == [a, c]
    assert len(fd.non_max_suppression([a, b, c], 1.0)) == 3  # disabled
    assert fd.iou(a, c) == 0.0


def test_tile_origins():
    assert fd.tile_origins(100, 200, 120) == [0]
    xs = fd.tile_origins(1310, 740, 444)
    assert xs[0] == 0 and xs[-1] == 1310 - 740 and all(b > a for a, b in zip(xs, xs[1:]))


def _naive_convolve(img, k):
    """Reference implementation: explicit double loop, true convolution (kernel flipped),
    reflect-101 border (same as cv2.BORDER_REFLECT_101 / np.pad 'reflect')."""
    r = k.shape[0] // 2
    pad = np.pad(img.astype(np.float64), ((r, r), (r, r), (0, 0)), mode="reflect")
    out = np.zeros(img.shape, np.float64)
    H, W = img.shape[:2]
    for y in range(H):
        for x in range(W):
            acc = np.zeros(img.shape[2])
            for i in range(-r, r + 1):
                for j in range(-r, r + 1):
                    acc += k[i + r, j + r] * pad[y + r - i, x + r - j]
            out[y, x] = acc
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def test_convolution_matches_naive_reference():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (12, 15, 3), dtype=np.uint8)
    k = np.array([[1, 2, 0], [0, -1, 3], [2, 0, 1]], float)  # asymmetric: detects missing flip
    got = sf.apply_convolution(img, k)
    ref = _naive_convolve(img, k)
    assert np.abs(got.astype(int) - ref.astype(int)).max() <= 1


def test_psnr_matches_opencv():
    import cv2
    rng = np.random.default_rng(4)
    a = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-20, 21, a.shape), 0, 255).astype(np.uint8)
    assert abs(sf.psnr(a, b) - cv2.PSNR(a, b)) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)