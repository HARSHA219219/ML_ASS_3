import streamlit as st
import torch, json, random, numpy as np, os
import torch.nn as nn

# -----------------------------
# Model Definition
class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_length, hidden1, hidden2=None,
                 activation='relu', dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        act = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU()
        }.get(activation.lower(), nn.ReLU())
        layers = [nn.Flatten(), nn.Linear(context_length * emb_dim, hidden1), act, nn.Dropout(dropout)]
        if hidden2:
            layers += [nn.Linear(hidden1, hidden2), act, nn.Dropout(dropout), nn.Linear(hidden2, vocab_size)]
        else:
            layers += [nn.Linear(hidden1, vocab_size)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.emb(x)
        return self.mlp(x)

# -----------------------------
# Text Generation
def sample_from_logits(logits, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)

def generate_next_k_words(model, seed_text, stoi, itos, device, k, context_length, temperature=1.0):
    model.eval()
    words = seed_text.lower().split()
    context = ([stoi['<PAD>']] * max(0, context_length - len(words))
               + [stoi.get(w, stoi['<UNK>']) for w in words])[-context_length:]
    generated = words.copy()
    for _ in range(k):
        x = torch.tensor([context], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x).squeeze(0)
        next_idx = sample_from_logits(logits, temperature)
        next_word = itos.get(next_idx, '<UNK>')
        generated.append(next_word)
        context = context[1:] + [next_idx]
    return ' '.join(generated)

# -----------------------------
# Streamlit Interface
st.title("🧠 Next Word Generator (MLP-based)")

st.markdown("""
This Streamlit app uses your trained **MLP-based next-word prediction model**
on *The Adventures of Sherlock Holmes* dataset.
Use the controls below to experiment with text generation.
""")

# --- Sidebar (fixed architecture)
st.sidebar.header("⚙️ Controls")

temperature = st.sidebar.slider("Temperature (controls randomness)", 0.5, 2.0, 1.0, 0.1)
seed = st.sidebar.number_input("Random Seed", 0, value=42)

model_name = "holmes_mlp64_relu"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    with open(f"{model_name}_vocab.json", 'r') as f:
        stoi = json.load(f)
    with open(f"{model_name}_config.json", 'r') as f:
        config = json.load(f)
    ckpt = torch.load(f"{model_name}_state.pt", map_location=device)

    embedding_dim = config["emb_dim"]
    context_length = config["context_length"]
    activation = config["activation"]

    st.sidebar.write(f"Embedding Dimension: **{embedding_dim}** (fixed)")
    st.sidebar.write(f"Context Length: **{context_length}** (fixed)")
    st.sidebar.write(f"Activation: **{activation}** (fixed)")

    model = NextWordMLP(
        vocab_size=config["vocab_size"],
        emb_dim=embedding_dim,
        context_length=context_length,
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        activation=activation,
        dropout=config["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    itos = {int(v): k for k, v in stoi.items()}
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# --- Text input and generation
st.subheader("📝 Input your starting text")
seed_text = st.text_input("Seed text", "the adventure of")
k = st.slider("Number of words to generate", 1, 100, 20)

if st.button("✨ Generate Text"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not seed_text.strip():
        st.warning("Please enter some input text!")
    else:
        out = generate_next_k_words(model, seed_text, stoi, itos, device, k, context_length, temperature)
        st.subheader("Generated Text:")
        st.write(out)

st.markdown("---")
st.caption("Created by Vadithya Harsha Vardhan Nayak (23110349) - NLP Assignment 1.4")
