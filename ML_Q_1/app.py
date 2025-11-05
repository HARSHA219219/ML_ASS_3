import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import random
import json
import os

# ---------------------------------------
# 🔧 Model Definition
# ---------------------------------------
class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_length, hidden1, hidden2=None,
                 activation='relu', dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        activation_layer = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU()
        }.get(activation.lower(), nn.ReLU())

        layers = [
            nn.Flatten(),
            nn.Linear(context_length * emb_dim, hidden1),
            activation_layer,
            nn.Dropout(dropout)
        ]

        if hidden2:
            layers += [
                nn.Linear(hidden1, hidden2),
                activation_layer,
                nn.Dropout(dropout),
                nn.Linear(hidden2, vocab_size)
            ]
        else:
            layers += [nn.Linear(hidden1, vocab_size)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.emb(x)
        return self.mlp(x)


# ---------------------------------------
# 🧠 Text Generation Utility
# ---------------------------------------
def generate_next_k_words(model, seed_text, stoi, itos, device, k, context_length, temperature=1.0):
    model.eval()
    words_in = seed_text.lower().split()
    context = (
        [0] * max(0, context_length - len(words_in)) +
        [stoi.get(w, 1) for w in words_in]
    )[-context_length:]
    generated = words_in.copy()

    for _ in range(k):
        x = torch.tensor([context], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)[0]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1).cpu()

            if probs.dim() != 1:
                probs = probs.flatten()

            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_word = itos.get(next_idx, "<UNK>")
            generated.append(next_word)
            context = context[1:] + [next_idx]

    return ' '.join(generated)


# ---------------------------------------
# ⚙️ Streamlit Interface
# ---------------------------------------
st.title("🧠 Next Word Generator using MLP (Holmes / Linux Dataset)")

st.markdown("#### Select your configuration below:")

# Dataset selection
dataset = st.selectbox("Select Dataset", ["Holmes", "Linux"])

# Activation selection
activation = st.selectbox("Select Activation Function", ["ReLU", "Tanh"])

# Embedding (fixed)
embedding = st.selectbox("Select Embedding Size", [64])

# Compose model key dynamically
key = f"{dataset.lower()}_mlp{embedding}_{activation.lower()}"

# Model file mapping
model_files = {
    "holmes_mlp64_relu": {
        "state": "holmes_mlp64_relu_state.pt",
        "vocab": "holmes_mlp64_relu_vocab.json",
        "config": "holmes_mlp64_relu_config.json"
    },
    "holmes_mlp64_tanh": {
        "state": "holmes_mlp64_tanh_state.pt",
        "vocab": "holmes_mlp64_tanh_vocab.json",
        "config": "holmes_mlp64_tanh_config.json"
    },
    "linux_mlp64_relu": {
        "state": "linux_mlp64_relu_state.pt",
        "vocab": "linux_mlp64_relu_vocab.json",
        "config": "linux_mlp64_relu_config.json"
    },
    "linux_mlp64_tanh": {
        "state": "linux_mlp64-tanh_state.pt",
        "vocab": "linux_mlp64-tanh_vocab.json",
        "config": "linux_mlp64-tanh_config.json"
    }
}

# Load correct file paths
selected = model_files[key]

# Sidebar controls
st.sidebar.header("⚙️ Generation Settings")
k = st.sidebar.slider("Number of words to generate", 1, 100, 30)
temperature = st.sidebar.slider("Sampling Temperature", 0.3, 2.0, 1.0, 0.1)
random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Text input
text = st.text_area("Enter your input text:", "the adventure of", height=100)

# Generate button
if st.button("✨ Generate Text"):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    with st.spinner(f"Loading {dataset} - {activation} model..."):
        # Load vocab and config
        with open(selected["vocab"], "r", encoding="utf-8") as f:
            stoi = json.load(f)
        itos = {i: w for w, i in stoi.items()}

        with open(selected["config"], "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load model
        model = NextWordMLP(
            vocab_size=config["vocab_size"],
            emb_dim=config["emb_dim"],
            context_length=config["context_length"],
            hidden1=config["hidden1"],
            hidden2=config.get("hidden2", None),
            activation=activation.lower(),
            dropout=config["dropout"]
        ).to(device)

        ckpt = torch.load(selected["state"], map_location=device)
        model.load_state_dict(ckpt["model_state"])

        # Generate words
        generated = generate_next_k_words(
            model=model,
            seed_text=text,
            stoi=stoi,
            itos=itos,
            device=device,
            k=k,
            context_length=config["context_length"],
            temperature=temperature
        )

        st.subheader("📝 Generated Text:")
        st.write(generated)

st.markdown("---")
st.markdown("**Tip:** Unknown words in input are replaced with `<UNK>`.")
