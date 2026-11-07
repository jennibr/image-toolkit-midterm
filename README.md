# AI-Assisted Image Processing & Face Analysis Toolkit

Python 3.11 · Streamlit

## Setup (VS Code terminal)
```bash
py -3.11 -m venv .venv            # Windows;  macOS/Linux: python3.11 -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q               # optional unit tests
```
In VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> choose `.venv`.

## Layout
| File | Feature | Report chapter |
|---|---|---|
| requirements.txt, README.md | environment | Ch.1 |
| (GitHub / Streamlit Cloud links) | deployment | Ch.2 |
| app.py + modules/image_io.py | F1 loading & metadata | Ch.3 |
| modules/color_hist.py | F2 RGB/YUV + histograms | Ch.3 |
| modules/spatial_filter.py | F3 kernel filtering + PSNR | Ch.3 |
| modules/face_detect.py | F4 face detection | Ch.3 |
| tests/test_core.py | boundary-case tests | Ch.3 |
| docs/AI_AUDIT_TRAIL.md | prompt log | Ch.4 |