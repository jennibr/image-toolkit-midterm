"""Main Streamlit entry point.  Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st

# Show the real cause on the page if OpenCV cannot load (e.g. missing system library)
try:
    import cv2  # noqa: F401
except Exception as exc:  # pragma: no cover
    st.error(f"OpenCV failed to import: {type(exc).__name__}: {exc}")
    st.info("If this mentions libGL.so.1 or libgthread, make sure packages.txt (libgl1, "
            "libglib2.0-0) is at the repo root, then delete and re-create the app.")
    st.stop()

from modules import color_hist, face_detect, image_io, spatial_filter

import matplotlib.pyplot as plt

st.set_page_config(page_title="AI-Assisted Image Processing Toolkit", layout="wide")
st.title("AI-Assisted Image Processing & Face Analysis Toolkit")

# ---------------------------------------------------------------- sidebar / load
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload an image", type=image_io.SUPPORTED_EXTENSIONS)
    st.caption("Supported: PNG, JPG/JPEG, BMP")

if uploaded is None:
    st.info("Upload an image in the sidebar to begin.")
    st.stop()

try:
    rgb, info = image_io.load_image(uploaded.getvalue(), uploaded.name)
except image_io.ImageLoadError as exc:
    st.error(str(exc))
    st.stop()


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Image Info", "2. Color Spaces & Histogram", "3. Kernel Filtering & PSNR", "4. Face Detection"])

# ---------------------------------------------------------------- feature 1
with tab1:
    c1, c2 = st.columns([2, 1])
    c1.image(rgb, caption=info.filename, use_container_width=True)
    with c2:
        st.subheader("Metadata")
        st.table(pd.DataFrame({
            "Property": ["File name", "Format", "Resolution (W x H)", "Megapixels",
                         "Original mode", "Channels", "Bit depth / channel", "File size"],
            "Value": [info.filename, info.fmt, f"{info.width} x {info.height}",
                      f"{info.megapixels:.2f}", info.original_mode, info.channels,
                      info.bit_depth, info.file_size_human],
        }).astype(str).set_index("Property"))
        if info.resized_from:
            st.warning(f"Image downscaled from {info.resized_from[0]}x{info.resized_from[1]} "
                       f"to {info.width}x{info.height} for performance.")

# ---------------------------------------------------------------- feature 2
with tab2:
    mode = st.radio("Display mode", ["Original", "RGB channels", "YUV channels"], horizontal=True)

    if mode == "Original":
        left, right = st.columns(2)
        left.image(rgb, caption="Original", use_container_width=True)
        with right:
            chans = color_hist.split_rgb(rgb)
            series = [(n, color_hist.compute_histogram(c), color_hist.RGB_COLORS[n])
                      for n, c in chans.items()]
            show_fig(color_hist.plot_histograms(series, "Original: R/G/B histogram"))
            gray = color_hist.to_gray(rgb)
            show_fig(color_hist.plot_histograms(
                [("Gray", color_hist.compute_histogram(gray), "#555555")],
                "Original: grayscale histogram"))

    elif mode == "RGB channels":
        tinted = st.checkbox("Show channels in colour (otherwise grayscale)", value=True)
        cols = st.columns(3)
        for col, (name, ch) in zip(cols, color_hist.split_rgb(rgb).items()):
            with col:
                col.image(color_hist.tint_channel(ch, name) if tinted else ch,
                          caption=f"{name} channel", use_container_width=True)
                show_fig(color_hist.plot_histograms(
                    [(name, color_hist.compute_histogram(ch), color_hist.RGB_COLORS[name])],
                    f"{name} histogram", figsize=(4, 2.8)))

    else:
        cols = st.columns(3)
        desc = {"Y": "Y (luma)", "U": "U (Cb chroma)", "V": "V (Cr chroma)"}
        for col, (name, ch) in zip(cols, color_hist.rgb_to_yuv(rgb).items()):
            with col:
                col.image(ch, caption=desc[name], use_container_width=True)
                show_fig(color_hist.plot_histograms(
                    [(name, color_hist.compute_histogram(ch), color_hist.YUV_COLORS[name])],
                    f"{name} histogram", figsize=(4, 2.8)))
        st.caption("U and V are offset by +128, so 'no colour' sits near 128.")

# ---------------------------------------------------------------- feature 3
with tab3:
    st.subheader("Custom convolution kernel")
    cfg, _ = st.columns([1, 2])
    with cfg:
        preset = st.selectbox("Preset", list(spatial_filter.PRESETS))
        text = st.text_area(
            "Kernel (one row per line; separate values by space or comma)",
            value=spatial_filter.PRESETS[preset], height=180, key=f"kernel_{preset}")
        norm_options = ["None", "Divide by sum of weights", "Divide by custom value"]
        default_norm = 1 if spatial_filter.PRESET_NORMALIZATION.get(preset) == "sum" else 0
        norm_label = st.radio("Normalization", norm_options, index=default_norm,
                              key=f"norm_{preset}")
        divisor = 1.0
        if norm_label == "Divide by custom value":
            divisor = st.number_input("Divisor", value=9.0)
        bias = st.slider("Bias / offset added after convolution", -255, 255, 0)
        border = st.selectbox("Border handling", list(spatial_filter.BORDER_MODES))
        gray_first = st.checkbox("Convert to grayscale first", value=False)

    norm_mode = {"None": "none", "Divide by sum of weights": "sum",
                 "Divide by custom value": "divisor"}[norm_label]
    try:
        kernel = spatial_filter.parse_kernel(text)
        kernel, warn = spatial_filter.normalize_kernel(kernel, norm_mode, divisor)
        if warn:
            st.warning(warn)
        if norm_mode == "none" and kernel.sum() > 1.5:
            st.warning(f"Kernel weights sum to {kernel.sum():.1f}, so the output will be about "
                       f"{kernel.sum():.0f}x brighter and will likely saturate. "
                       "Consider 'Divide by sum of weights'.")
        filtered = spatial_filter.apply_convolution(rgb, kernel, border, bias, gray_first)
    except spatial_filter.KernelError as exc:
        st.error(f"Invalid kernel: {exc}")
        st.stop()

    # reference for PSNR must match colour handling
    reference = (np.repeat(color_hist.to_gray(rgb)[..., None], 3, axis=2)
                 if gray_first else rgb)
    m = spatial_filter.mse(reference, filtered)
    p = spatial_filter.psnr(reference, filtered)

    a, b = st.columns(2)
    a.image(rgb, caption="Before (original)", use_container_width=True)
    b.image(filtered, caption="After (filtered)", use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("PSNR", "∞ dB (identical)" if np.isinf(p) else f"{p:.2f} dB")
    m2.metric("MSE", f"{m:.2f}")
    m3.metric("Kernel size / sum", f"{kernel.shape[0]}x{kernel.shape[1]} / {kernel.sum():.3f}")

    if np.isinf(p):
        st.caption("Filtered image is pixel-identical to the original.")
    elif p >= 40:
        st.caption("Very small change (PSNR ≥ 40 dB).")
    elif p >= 30:
        st.caption("Moderate change (30–40 dB).")
    else:
        st.caption("Large change (< 30 dB), typical for edge/emboss filters.")
    st.caption("Note: here PSNR measures how far the result deviates from the original, "
               "not whether the result is 'better'.")

    with st.expander("PSNR formula"):
        st.latex(r"MSE=\frac{1}{HWC}\sum_{x=1}^{H}\sum_{y=1}^{W}\sum_{c=1}^{C}"
                 r"\big(I(x,y,c)-K(x,y,c)\big)^2")
        st.latex(r"PSNR=10\log_{10}\!\left(\frac{MAX_I^{2}}{MSE}\right)"
                 r"=20\log_{10}(MAX_I)-10\log_{10}(MSE),\quad MAX_I=255")
        st.write("I = original, K = filtered image. If MSE = 0 the PSNR is infinite.")

# ---------------------------------------------------------------- feature 4
with tab4:
    st.subheader("Face detection")
    mp_ok = face_detect.mediapipe_available()
    engines = (["MediaPipe Face Detection"] if mp_ok else []) + ["Haar Cascade (OpenCV)"]
    engine = st.selectbox("Detector", engines)
    if not mp_ok:
        st.caption("MediaPipe not installed; only Haar available.")

    try:
        if engine.startswith("Haar"):
            c1, c2, c3, c4 = st.columns(4)
            sf = c1.slider("scaleFactor", 1.05, 1.5, 1.1, 0.05)
            mn = c2.slider("minNeighbors", 1, 15, 5)
            ms = c3.slider("minSize (px)", 20, 200, 30)
            mc = c4.slider("Min confidence", 0.0, 0.99, 0.0, 0.05)
            haar_all = face_detect.detect_haar(rgb, sf, mn, ms)
            dets = [d for d in haar_all if d.confidence >= mc]
            if len(dets) != len(haar_all):
                st.caption(f"Haar found {len(haar_all)} candidate(s); {len(haar_all) - len(dets)} "
                           f"hidden by the min-confidence filter ({mc:.2f}).")
        else:
            st.caption("First run downloads a small model file (~200 KB) into models/.")
            mc = st.slider("Min detection confidence", 0.1, 0.99, 0.5, 0.05)
            tiled = st.checkbox("Scan image in square tiles (better for small faces, slower)",
                                value=True)
            dets = face_detect.detect_mediapipe(rgb, mc, tiled)
    except Exception as exc:  # keep UI alive on model errors
        st.error(f"Detection failed: {exc}")
        st.stop()

    raw_count = len(dets)
    iou_thr = st.slider("Overlap threshold (IoU) for removing duplicate boxes "
                        "(1.0 = off)", 0.05, 1.0, 0.3, 0.05)
    dets = face_detect.non_max_suppression(dets, iou_thr)

    st.metric("Faces detected", len(dets))
    if raw_count != len(dets):
        st.caption(f"NMS merged {raw_count - len(dets)} overlapping duplicate box(es) "
                   f"({raw_count} raw -> {len(dets)}).")
    st.image(face_detect.draw_detections(rgb, dets), use_container_width=True)
    if dets:
        df = pd.DataFrame([d.as_row() for d in dets])
        df.index = range(1, len(df) + 1)
        df.index.name = "face #"
        st.dataframe(df, use_container_width=True)
        if engine.startswith("Haar"):
            st.caption("Haar confidence is a heuristic: 1 − exp(−w/2) of OpenCV's level weight.")
    else:
        st.info("No faces found. Try lowering minNeighbors / minSize or use another image.")