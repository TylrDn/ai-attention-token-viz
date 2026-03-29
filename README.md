# ai-attention-token-viz

Interactive visualization tool for token-to-token attention in language models.

![Demo GIF placeholder](assets/demo.gif)
<!-- Replace the path above with your recorded demo GIF once available -->

---

## Features

- 🔥 **Plotly heatmaps** — interactive per-layer & per-head attention grids
- 🧠 **BertViz integration** — clickable head-view widget inside the Streamlit app
- 🤗 **Any HF encoder model** — swap in `roberta-base`, `distilbert-base-uncased`, etc.
- ☁️ **Runs in Colab** — expose the Streamlit app via `pyngrok` with one cell

---

## Repository structure

```
.
├── notebooks/
│   └── 01_demo_attention.ipynb   # Plotly heatmap demos
├── src/
│   ├── extract_attention.py      # Attention extraction helpers
│   └── viz.py                    # Streamlit app
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Quick start

### Install

```bash
pip install -r requirements.txt
```

### Run the Streamlit app (locally)

```bash
streamlit run src/viz.py
```

Open <http://localhost:8501>, enter a sentence and click **Extract & Visualize**.

### Run in Google Colab

```python
!pip install -q streamlit pyngrok transformers bertviz plotly torch
!git clone https://github.com/TylrDn/ai-attention-token-viz.git

from pyngrok import ngrok
import subprocess, threading, time

def run():
    subprocess.run([
        "streamlit", "run", "ai-attention-token-viz/src/viz.py",
        "--server.port", "8501",
        "--server.headless", "true",
    ])

threading.Thread(target=run, daemon=True).start()
time.sleep(6)
public_url = ngrok.connect(8501)
print("Streamlit URL:", public_url)
```

### Explore the notebook

```bash
jupyter lab notebooks/01_demo_attention.ipynb
```

---

## CI

GitHub Actions runs on every push / PR:
- `flake8` lint of `src/`
- `nbformat` validation of notebooks
- Import smoke-test for `extract_attention`

[![CI](https://github.com/TylrDn/ai-attention-token-viz/actions/workflows/ci.yml/badge.svg)](https://github.com/TylrDn/ai-attention-token-viz/actions/workflows/ci.yml)

