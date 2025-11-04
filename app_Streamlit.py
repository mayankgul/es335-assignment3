import os
import requests
import zipfile
import streamlit as st

MODEL_FILES = [
    "model_shakespeare_state.pt",
    "model_linux_state.pt",
    "stoi_shake(1).pkl", "itos_shake(1).pkl",
    "stoi_code(1).pkl", "itos_code(1).pkl"
]

if not all(os.path.exists(p) for p in MODEL_FILES):
    st.info("Downloading model weights from GitHub release… please wait (~250 MB).")
    url = "https://github.com/mayankgul/es335-assignment3/releases/download/v1.0/Models.zip"
    zip_path = "Models.zip"

    # Download and extract
    with open(zip_path, "wb") as f:
        f.write(requests.get(url).content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    st.success("Model weights downloaded successfully.")

import torch
import torch.nn as nn
import pickle
import random

#  Define Model Class

class MLPTextGenerator(nn.Module):
    def __init__(self, vocab_size, pad_idx, emb_dim, context_len, hidden_sizes, activation='relu', dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        layers = []
        in_dim = emb_dim * context_len
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        e = self.emb(x).view(x.size(0), -1)
        return self.net(e)

#  Helper Functions
@st.cache_resource(show_spinner=False)
def load_vocab(stoi_path, itos_path):
    with open(stoi_path, "rb") as f:
        stoi = pickle.load(f)
    with open(itos_path, "rb") as f:
        itos = pickle.load(f)
    pad_idx = len(stoi)
    return stoi, itos, pad_idx

def load_model(model_path, stoi_path, itos_path, emb_dim, context_len, hidden_sizes, activation):
    stoi, itos, pad_idx = load_vocab(stoi_path, itos_path)
    model = MLPTextGenerator(
        vocab_size=len(stoi),
        pad_idx=pad_idx,
        emb_dim=emb_dim,
        context_len=context_len,
        hidden_sizes=hidden_sizes,
        activation=activation
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, stoi, itos, pad_idx

def encode_context(words, stoi, context_len, pad_idx):
    ids = [stoi.get(w, pad_idx) for w in words]
    if len(ids) < context_len:
        ids = [pad_idx] * (context_len - len(ids)) + ids
    else:
        ids = ids[-context_len:]
    return torch.tensor([ids], dtype=torch.long)

def sample_next(model, x, temperature=1.0):
    with torch.no_grad():
        logits = model(x)[0] / temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

def generate(model, stoi, itos, seed_text, k, context_len, pad_idx, temperature):
    words = seed_text.strip().lower().split()
    gen = words[:]
    for _ in range(k):
        x = encode_context(gen, stoi, context_len, pad_idx)
        nxt = sample_next(model, x, temperature)
        gen.append(itos.get(nxt, "<UNK>"))
    return " ".join(gen)

# 3. Streamlit UI
st.title("Next-Word Prediction using MLP")

#  User Controls 
dataset = st.selectbox("Choose Dataset", ["Shakespeare", "Linux Code"])
temperature = st.slider("Temperature (randomness control)", 0.3, 2.0, 1.0, 0.1)
k = st.slider("Number of words to predict", 1, 30, 10)
context_len = st.number_input("Context length", value=5, min_value=1, max_value=20)
emb_dim = st.number_input("Embedding dimension", value=64)
activation = st.selectbox("Activation Function", ["relu", "tanh"])
seed_text = st.text_input("Enter seed text", "to be or not to be")
random_seed = st.number_input("Random seed", value=42)

#  File Mapping (no renaming) 
if dataset == "Shakespeare":
    model_path = "model_shakespeare_state.pt"
    stoi_path = "stoi_shake (1).pkl"
    itos_path = "itos_shake (1).pkl"
else:
    model_path = "model_linux_state.pt"
    stoi_path = "stoi_code (1).pkl"
    itos_path = "itos_code (1).pkl"

# Generate Button 
if st.button("Generate"):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    model, stoi, itos, PAD_IDX = load_model(model_path, stoi_path, itos_path,
                                            emb_dim, context_len, [1024], activation)
    output = generate(model, stoi, itos, seed_text, k, context_len, PAD_IDX, temperature)
    st.subheader(" Generated Output:")

    st.write(output)
