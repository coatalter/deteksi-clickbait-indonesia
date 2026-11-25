import os
import re
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf
from tensorflow import keras


# ============================================================
# TOKENIZER
# ============================================================

class SequentialHeadlineTokenizer:

    def __init__(self, max_vocab=12000, max_len=25):

        self.max_vocab = max_vocab
        self.max_len = max_len

        self.word_index = {
            "<PAD>": 0,
            "<OOV>": 1
        }

        self.index_word = {
            0: "<PAD>",
            1: "<OOV>"
        }

    def texts_to_sequences(self, texts):

        oov_idx = self.word_index.get("<OOV>", 1)

        seqs = []

        for text in texts:

            tokens = str(text).split()

            seq = [
                self.word_index.get(token, oov_idx)
                for token in tokens[:self.max_len]
            ]

            if len(seq) < self.max_len:

                seq += [
                    0
                ] * (self.max_len - len(seq))

            seqs.append(seq)

        return np.asarray(
            seqs,
            dtype=np.int32
        )


# ============================================================
# PREPROCESSING
# ============================================================

def clean_text(text, keep_punct=True):

    text = str(text).lower()

    if keep_punct:

        text = re.sub(
            r"[^a-z0-9\s!?]",
            " ",
            text
        )

        text = re.sub(
            r"([!?])",
            r" \1 ",
            text
        )

    else:

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# MEMUAT MODEL DAN ARTEFAK
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)

seq_tokenizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "seq_tokenizer.pkl"
    )
)

mlp_model = keras.models.load_model(
    os.path.join(
        MODEL_DIR,
        "mlp_model_uts.keras"
    )
)

lstm_model = keras.models.load_model(
    os.path.join(
        MODEL_DIR,
        "lstm_fasttext_best.keras"
    )
)


# ============================================================
# TOKEN PERTURBATION
# ============================================================

def token_perturbation(model, text, tokenizer):

    clean = clean_text(
        text,
        keep_punct=True
    )

    tokens = clean.split()[:tokenizer.max_len]

    if not tokens:
        raise ValueError(
            "Judul tidak memiliki token setelah preprocessing."
        )

    sequence = np.asarray(
        tokenizer.texts_to_sequences([
            clean
        ])[0],
        dtype=np.int32
    )

    valid_length = min(
        len(tokens),
        int(np.count_nonzero(sequence))
    )

    tokens = tokens[:valid_length]

    if valid_length == 0:
        raise ValueError(
            "Tidak ada token valid setelah tokenisasi."
        )

    valid_ids = sequence[:valid_length].copy()

    base_prob = float(
        model.predict(
            sequence.reshape(1, -1),
            verbose=0
        ).ravel()[0]
    )

    perturbed_batch = []

    for idx in range(valid_length):

        shortened = np.concatenate([
            valid_ids[:idx],
            valid_ids[idx + 1:]
        ])

        modified = np.zeros_like(
            sequence,
            dtype=np.int32
        )

        modified[:len(shortened)] = shortened

        perturbed_batch.append(modified)

    perturbed_batch = np.asarray(
        perturbed_batch,
        dtype=np.int32
    )

    perturbed_probs = model.predict(
        perturbed_batch,
        batch_size=32,
        verbose=0
    ).ravel()

    deltas = base_prob - perturbed_probs

    return tokens, deltas, base_prob


# ============================================================
# FUNGSI PREDIKSI
# ============================================================

def predict_clickbait(judul):

    if not judul or not str(judul).strip():

        return (
            "Masukkan judul berita terlebih dahulu.",
            {},
            [],
            None
        )

    clean = clean_text(
        judul,
        keep_punct=True
    )

    if not clean:

        return (
            "Judul tidak memiliki token setelah preprocessing.",
            {},
            [],
            None
        )

    # MLP UTS
    X_tfidf = tfidf.transform([
        clean
    ])

    prob_uts = float(
        mlp_model.predict(
            X_tfidf,
            verbose=0
        ).ravel()[0]
    )

    # LSTM FastText UAS
    tokens, deltas, prob_uas = token_perturbation(
        lstm_model,
        judul,
        seq_tokenizer
    )

    label_uas = (
        "CLICKBAIT"
        if prob_uas >= 0.5
        else "NON-CLICKBAIT"
    )

    label_uts = (
        "CLICKBAIT"
        if prob_uts >= 0.5
        else "NON-CLICKBAIT"
    )

    summary = (
        f"### Hasil Model Utama UAS: **{label_uas}**\n"
        f"- **LSTM + FastText (UAS):** "
        f"{prob_uas*100:.2f}% Clickbait ({label_uas})\n"
        f"- **MLP + TF-IDF (UTS):** "
        f"{prob_uts*100:.2f}% Clickbait ({label_uts})"
    )

    probabilities = {
        "Clickbait": prob_uas,
        "Non-Clickbait": 1.0 - prob_uas
    }

    highlighted = []

    for token, delta in zip(tokens, deltas):

        if delta > 0.005:

            category = "Mendukung Clickbait"

        elif delta < -0.005:

            category = "Menekan Clickbait"

        else:

            category = "Pengaruh Kecil"

        highlighted.append(
            (token, category)
        )

    colors = [
        "#238b45" if d >= 0 else "#d73027"
        for d in deltas
    ]

    fig, ax = plt.subplots(
        figsize=(9, 3.5)
    )

    ax.bar(
        range(len(tokens)),
        deltas * 100,
        color=colors,
        edgecolor="#333"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.set_xticks(
        range(len(tokens))
    )

    ax.set_xticklabels(
        tokens,
        rotation=25,
        ha="right"
    )

    ax.set_ylabel(
        "Perubahan Probabilitas (pp)"
    )

    ax.set_title(
        "Pengaruh Token terhadap Probabilitas Clickbait"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    return (
        summary,
        probabilities,
        highlighted,
        fig
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Deteksi Clickbait Indonesia - LSTM FastText"
) as demo:

    gr.Markdown(
        "# Sistem Deteksi Judul Berita Clickbait Indonesia\n"
        "**Model UAS: LSTM + FastText**\n\n"
        "Perbandingan dengan baseline MLP + TF-IDF (UTS)."
    )

    with gr.Row():

        with gr.Column(scale=3):

            input_title = gr.Textbox(
                label="Masukkan Judul Berita",
                lines=2
            )

            button = gr.Button(
                "Analisis Judul Berita",
                variant="primary"
            )

        with gr.Column(scale=2):

            output_probability = gr.Label(
                num_top_classes=2,
                label="Probabilitas Kelas LSTM + FastText"
            )

    output_summary = gr.Markdown()

    output_tokens = gr.HighlightedText(
        label="Pengaruh Token",
        combine_adjacent=False,
        show_legend=True
    )

    output_plot = gr.Plot(
        label="Grafik Token Perturbation"
    )

    button.click(
        fn=predict_clickbait,
        inputs=input_title,
        outputs=[
            output_summary,
            output_probability,
            output_tokens,
            output_plot
        ]
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
