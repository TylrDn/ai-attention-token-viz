"""Streamlit app for interactive token-token attention visualization.

Run locally:
    streamlit run src/viz.py

Run in Google Colab:
    !pip install -q streamlit pyngrok transformers bertviz plotly torch
    from pyngrok import ngrok
    import subprocess, threading, time

    def run():
        subprocess.run(["streamlit", "run", "src/viz.py",
                        "--server.port", "8501",
                        "--server.headless", "true"])

    threading.Thread(target=run, daemon=True).start()
    time.sleep(5)
    public_url = ngrok.connect(8501)
    print("Streamlit URL:", public_url)
"""

from __future__ import annotations

import torch
import plotly.graph_objects as go
import streamlit as st

from extract_attention import get_attention, get_mean_attention_per_layer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attention_heatmap(
    tokens: list[str],
    matrix,
    title: str = "Attention",
) -> go.Figure:
    """Return a Plotly heatmap for a 2-D attention matrix."""
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=tokens,
            y=tokens,
            colorscale="Blues",
            zmin=0.0,
            zmax=float(matrix.max()),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Key token",
        yaxis_title="Query token",
        yaxis_autorange="reversed",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Attention Token Viz", layout="wide")
    st.title("🔍 AI Attention Token Visualizer")
    st.markdown(
        "Interactive token-to-token attention maps for Hugging Face transformer models."
    )

    with st.sidebar:
        st.header("Settings")
        model_name = st.text_input(
            "HF model name",
            value="bert-base-uncased",
            help="Any encoder model from huggingface.co/models",
        )
        sentence = st.text_area(
            "Input sentence",
            value="The cat sat on the mat.",
        )
        run_btn = st.button("Extract & Visualize", type="primary")

    if not run_btn:
        st.info("Enter a sentence in the sidebar and click **Extract & Visualize**.")
        return

    with st.spinner(f"Loading *{model_name}* and computing attention…"):
        try:
            tokens, attentions = get_attention(sentence, model_name=model_name)
        except Exception as exc:
            st.error(f"Failed to load model or extract attention:\n\n`{exc}`")
            return

    num_layers, num_heads, _, _ = attentions.shape

    st.success(
        f"Model: **{model_name}** — {num_layers} layers × {num_heads} heads — "
        f"{len(tokens)} tokens"
    )

    # --- Layer / head selector ---
    tab_avg, tab_layer = st.tabs(["Average across layers", "Single layer / head"])

    with tab_avg:
        mean_attn = attentions.mean(dim=(0, 1)).numpy()
        fig = _attention_heatmap(tokens, mean_attn, "Mean attention (all layers & heads)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_layer:
        col1, col2 = st.columns(2)
        with col1:
            layer_idx = st.slider("Layer", 0, num_layers - 1, 0)
        with col2:
            head_idx = st.slider("Head", 0, num_heads - 1, 0)

        per_layer = get_mean_attention_per_layer(attentions)
        layer_mean = per_layer[layer_idx].numpy()
        head_attn = attentions[layer_idx, head_idx].numpy()

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _attention_heatmap(
                    tokens,
                    layer_mean,
                    f"Layer {layer_idx} — mean over heads",
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _attention_heatmap(
                    tokens,
                    head_attn,
                    f"Layer {layer_idx}, Head {head_idx}",
                ),
                use_container_width=True,
            )

    # --- BertViz head-view (optional) ---
    st.divider()
    with st.expander("🧠 BertViz head-view (interactive)"):
        try:
            from bertviz import head_view
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, output_attentions=True)
            model.eval()
            inputs = tokenizer(sentence, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            html_obj = head_view(
                outputs.attentions,
                tokens,
                html_action="return",
            )
            st.components.v1.html(html_obj.data, height=600, scrolling=True)
        except Exception as exc:
            st.warning(f"BertViz head-view unavailable: {exc}")


if __name__ == "__main__":
    main()
