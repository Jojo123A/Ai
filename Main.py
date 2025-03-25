import math
import random
import os
import time  # For timing and measuring inference speed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, concatenate_datasets
from flask import Flask, render_template, request, jsonify

# Additional ML and Deep Learning libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.models as tv_models
import tensorflow as tf

# For video tutorial creation (MoviePy)
from moviepy.editor import TextClip, CompositeVideoClip

#############################################
# Additional Reasoning & Debugging Modules
#############################################

# 1. Math Problem Solver using sympy
from sympy import symbols, Eq, solve
def solve_math_problem(equation_str):
    """
    Solve a math equation given as a string.
    Example: equation_str = "2*x+3-7" (interpreted as 2*x+3-7=0)
    """
    try:
        # Assume equation in variable x
        x = symbols('x')
        eq = Eq(eval(equation_str), 0)
        solution = solve(eq, x)
        return solution
    except Exception as e:
        return f"Error solving equation: {e}"

# 2. World Problem Reasoning using chain-of-thought
def solve_world_problem(problem_statement):
    """
    Solve a world problem by retrieving external context and generating a chain-of-thought.
    """
    reasoning_text = chain_of_thought_generation(model, tokenizer, problem_statement, max_length=50)
    return reasoning_text

# 3. Explain decision with SHAP and XGBoost
import shap
import xgboost as xgb

# Train a simple XGBoost classifier for demonstration
X_train_shap = np.random.rand(100, 5)
y_train_shap = np.random.randint(0, 2, 100)
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_shap, y_train_shap)

def explain_decision(input_data):
    """
    Generate a SHAP waterfall plot for the provided input_data.
    """
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(input_data)
    shap.plots.waterfall(shap_values[0])

# 4. Federated Learning Example using TensorFlow Federated
import tensorflow_federated as tff

def model_fn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

federated_model = tff.learning.from_keras_model(
    model_fn(), input_spec=(tf.TensorSpec([None, 5]), tf.TensorSpec([None, 1])),
    loss=tf.keras.losses.BinaryCrossentropy()
)

# 5. Code Analysis Function (for debugging)
def analyze_code(code_snippet):
    errors = []
    try:
        exec(code_snippet)
    except Exception as e:
        errors.append(str(e))
    return errors if errors else "No errors found."

print(analyze_code("print(1/0)"))  # Will detect a division by zero error

#############################################
# 6. Load and Combine Multiple Datasets
#############################################

import requests
from bs4 import BeautifulSoup
from readability import Document
import random
import nltk
import json
from googlesearch import search
from rank_bm25 import BM25Okapi
from serpapi import GoogleSearch

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# === CONFIGURATION ===
QUERY = "latest cybersecurity news"  # Change search term as needed
NUM_PAGES = 100  # Number of URLs to scrape
SERPAPI_KEY = "your_serpapi_api_key"  # Get API Key from https://serpapi.com/

# === FUNCTION TO GET URLs ===
def get_search_results(query, num_results=100):
    """ Fetches search results dynamically using SerpAPI. """
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERPAPI_KEY
    }
    search_results = GoogleSearch(params).get_dict()
    
    urls = []
    for result in search_results.get("organic_results", []):
        if "link" in result:
            urls.append(result["link"])
    
    return urls[:num_results]

# === FUNCTION TO SCRAPE CONTENT ===
def fetch_text(url):
    """ Fetches and extracts meaningful text from a webpage. """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyScraper/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            doc = Document(response.text)  # Extracts only the main readable part
            soup = BeautifulSoup(doc.summary(), 'html.parser')

            # Remove unwanted elements
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.decompose()

            text = soup.get_text(separator=" ", strip=True)
            return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    
    return ""

# === FETCH & PROCESS DATA ===
print("\n[+] Fetching search results...")
urls = get_search_results(QUERY, NUM_PAGES)

print(f"[+] Found {len(urls)} URLs. Scraping content now...\n")

all_sentences = []
for url in urls:
    print(f"Scraping: {url}")
    text = fetch_text(url)
    if text:
        sentences = sent_tokenize(text)
        all_sentences.extend(sentences)

print(f"\n[+] Scraped {len(all_sentences)} sentences from {len(urls)} pages.")

# === SAVE DATABASE TO JSON ===
with open("scraped_data.json", "w", encoding="utf-8") as f:
    json.dump(all_sentences, f, ensure_ascii=False, indent=4)

# === BM25 INDEXING FOR ADVANCED SEARCH ===
print("\n[+] Building BM25 Index for intelligent search...")
tokenized_corpus = [nltk.word_tokenize(sent.lower()) for sent in all_sentences]
bm25 = BM25Okapi(tokenized_corpus)

# === QUERY FUNCTION ===
def answer_query(query, bm25, sentences):
    """ Finds the most relevant sentences using BM25 ranking. """
    tokenized_query = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    
    return [sentences[i] for i in top_indexes]

# === EXAMPLE USAGE ===
while True:
    user_query = input("\nAsk a question (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break

    print("\n[+]thinking ...\n")
    results = answer_query(user_query, bm25, all_sentences)

    for i, ans in enumerate(results):
        print(f"{i+1}. {ans}\n")




#############################################
# 7. Build FAISS Index for Retrieval-Augmented Generation
#############################################
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
corpus = [ex["text"] for ex in combined_dataset]
tfidf_matrix = tfidf.fit_transform(corpus)
tfidf_dense = tfidf_matrix.toarray().astype('float32')
d = tfidf_dense.shape[1]
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(tfidf_dense)

#############################################
# 8. Improved Tokenization: Subword Tokenizer (with fallback)
#############################################
all_texts = [ex["text"] for ex in combined_dataset]

# Try to use SentencePiece tokenizer (if installed)
try:
    import sentencepiece as spm
    if not os.path.exists("spm.model"):
        print("Training SentencePiece tokenizer...")
        with open("spm_input.txt", "w", encoding="utf-8") as f:
            for text in all_texts:
                f.write(text + "\n")
        spm.SentencePieceTrainer.train(input="spm_input.txt", model_prefix="spm", vocab_size=5000, model_type='unigram', user_defined_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
    sp = spm.SentencePieceProcessor()
    sp.load("spm.model")
    class SentencePieceTokenizer:
        def encode(self, text):
            return [sp.bos_id()] + sp.encode_as_ids(text) + [sp.eos_id()]
        def decode(self, ids):
            return sp.decode_ids(ids)
        def vocab_size(self):
            return sp.get_piece_size()
    tokenizer = SentencePieceTokenizer()
    print("Using SentencePiece tokenizer with vocab size:", tokenizer.vocab_size())
except Exception as e:
    print("SentencePiece tokenizer not available, falling back to ByteLevelBPETokenizer...")
    try:
        from tokenizers import ByteLevelBPETokenizer
        if not os.path.exists("custom_tokenizer-vocab.json"):
            print("Training subword tokenizer...")
            tokenizer_model = ByteLevelBPETokenizer()
            tokenizer_model.train_from_iterator(
                all_texts,
                vocab_size=5000,
                min_frequency=2,
                special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
            )
            tokenizer_model.save_model(".", "custom_tokenizer")
        tokenizer_model = ByteLevelBPETokenizer("custom_tokenizer-vocab.json", "custom_tokenizer-merges.txt")
        class SubwordTokenizer:
            def __init__(self, tokenizer_model):
                self.model = tokenizer_model
            def encode(self, text):
                encoding = self.model.encode("<BOS> " + text + " <EOS>")
                return encoding.ids
            def decode(self, ids):
                return self.model.decode(ids)
            def vocab_size(self):
                return self.model.get_vocab_size()
        tokenizer = SubwordTokenizer(tokenizer_model)
        print("Using subword tokenizer with vocab size:", tokenizer.vocab_size())
    except Exception as e:
        print("Subword tokenizer not available, falling back to SimpleTokenizer:", e)
        class SimpleTokenizer:
            def __init__(self, texts, min_freq=2):
                self.word2idx = {}
                self.idx2word = {}
                self.build_vocab(texts, min_freq)
            def build_vocab(self, texts, min_freq):
                from collections import Counter
                counter = Counter()
                for text in texts:
                    tokens = text.split()
                    counter.update(tokens)
                vocab = [word for word, freq in counter.items() if freq >= min_freq]
                self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
                for word in vocab:
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
                self.idx2word = {idx: word for word, idx in self.word2idx.items()}
            def encode(self, text):
                tokens = text.split()
                tokens = ["<BOS>"] + tokens + ["<EOS>"]
                return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
            def decode(self, indices):
                tokens = [self.idx2word.get(idx, "<UNK>") for idx in indices]
                return " ".join(tokens)
            def vocab_size(self):
                return len(self.word2idx)
        tokenizer = SimpleTokenizer(all_texts, min_freq=2)
        print("Using simple tokenizer with vocab size:", tokenizer.vocab_size())

#############################################
# 9. Create a PyTorch Dataset for Language Modeling
#############################################
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = [self.tokenizer.encode(text) for text in texts]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        tokens = self.data[idx]
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return torch.tensor(tokens, dtype=torch.long)

texts = [ex["text"] for ex in combined_dataset]
lm_dataset = TextDataset(texts, tokenizer, max_length=64)
dataloader = DataLoader(lm_dataset, batch_size=16, shuffle=True)

#############################################
# 10. Define an Enhanced Custom Transformer Language Model
#     (with optional Next-Sentence Prediction)
#############################################
class NSPHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 2)
    def forward(self, hidden_states):
        cls_state = hidden_states[:, 0, :]
        logits = self.linear(cls_state)
        return logits

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=6, max_seq_length=64, dropout=0.1, use_nsp=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.use_nsp = use_nsp
        if self.use_nsp:
            self.nsp_head = NSPHead(d_model)
    def forward(self, x, return_hidden=False):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(positions)
        x_emb = self.dropout(token_emb + pos_emb)
        x_emb = x_emb.transpose(0, 1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        out = self.transformer_decoder(x_emb, x_emb, tgt_mask=mask)
        out = out.transpose(0, 1)
        logits = self.fc_out(out)
        if return_hidden:
            if self.use_nsp:
                nsp_logits = self.nsp_head(out)
                return out, logits, nsp_logits
            return out, logits
        if self.use_nsp:
            nsp_logits = self.nsp_head(out)
            return logits, nsp_logits
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size()
model = TransformerLM(vocab_size=vocab_size, use_nsp=True).to(device)

class ValueHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        value = self.linear(pooled)
        return value.squeeze(-1)

#############################################
# 11. Optional: LoRA Integration for Efficient Fine-Tuning
#############################################
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.lora_A = nn.Parameter(torch.Tensor(r, in_features))
        self.lora_B = nn.Parameter(torch.Tensor(out_features, r))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original = nn.functional.linear(x, self.weight, self.bias)
        lora_out = nn.functional.linear(x, self.lora_A.T)
        lora_out = nn.functional.linear(lora_out, self.lora_B.T)
        return original + self.alpha * lora_out

def apply_lora(module, r=4, alpha=1.0):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child.in_features, child.out_features, r=r, alpha=alpha, bias=(child.bias is not None)))
        else:
            apply_lora(child, r, alpha)

# Uncomment the following line to apply LoRA to the entire model:
# apply_lora(model)

#############################################
# 12. Pretraining: Self-Supervised LM with AMP & TensorBoard Logging
#############################################
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter("runs/custom_transformer_pretrain")
print("Starting pretraining ...")
num_pretrain_epochs = 3

for epoch in range(num_pretrain_epochs):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(batch)
            if model.use_nsp:
                lm_logits, _ = outputs
            else:
                lm_logits = outputs
            logits = lm_logits[:, :-1, :].contiguous().view(-1, vocab_size)
            targets = batch[:, 1:].contiguous().view(-1)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            writer.add_scalar("Pretrain Loss", loss.item(), epoch * len(dataloader) + i)
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_pretrain_epochs} - Loss: {avg_loss:.4f}")

#############################################
# 13. (Optional) NSP Pretraining Step for Multi-Task Learning
#############################################
def nsp_pretraining_step(model, texts, tokenizer, criterion_nsp, device, max_len=64):
    model.train()
    batch_sent1, batch_sent2, labels = [], [], []
    for text in texts:
        words = text.split()
        if len(words) < 10:
            continue
        mid = len(words) // 2
        sent1 = " ".join(words[:mid])
        sent2 = " ".join(words[mid:])
        batch_sent1.append(sent1)
        batch_sent2.append(sent2)
        labels.append(1)
        neg_sent2 = random.choice(texts)
        batch_sent1.append(sent1)
        batch_sent2.append(neg_sent2)
        labels.append(0)
    def encode_and_pad(s):
        seq = tokenizer.encode(s)[:max_len]
        if len(seq) < max_len:
            seq = seq + [0]*(max_len - len(seq))
        return seq
    inputs1 = [encode_and_pad(s) for s in batch_sent1]
    inputs1 = torch.tensor(inputs1, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    _, _, nsp_logits = model(inputs1, return_hidden=True)
    loss_nsp = criterion_nsp(nsp_logits, labels)
    return loss_nsp
    
    #############################################
# 14. Functions for PPO RL Fine-Tuning (RLHF)
#############################################
def generate_text_with_log_probs(model, tokenizer, prompt, max_length=50):
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    log_probs = []
    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            outputs = model(tokens_tensor)
            if model.use_nsp:
                lm_logits, _ = outputs
            else:
                lm_logits = outputs
            next_token_logits = lm_logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample()
            log_prob = dist.log_prob(next_token)
            log_probs.append(log_prob)
            tokens_tensor = torch.cat([tokens_tensor, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.encode("<EOS>")[0]:
                break
    generated_text = tokenizer.decode(tokens_tensor[0].tolist())
    return generated_text, torch.stack(log_probs)

def quality_reward_fn(generated_text):
    tokens = generated_text.split()
    if len(tokens) == 0:
        return 0.0
    lexical_diversity = len(set(tokens)) / len(tokens)
    punctuation_bonus = 1.0 if generated_text.strip()[-1] in ".!?" else 0.5
    length_bonus = min(len(tokens), 50) / 50.0
    return lexical_diversity * punctuation_bonus * length_bonus

def ppo_update(model, value_head, tokens_tensor, old_log_prob, reward, clip_epsilon=0.2, value_loss_coef=0.5):
    hidden, logits = model(tokens_tensor, return_hidden=True)
    log_probs_all = []
    tokens = tokens_tensor[0]
    for i in range(1, tokens_tensor.size(1)):
        token_logits = logits[0, i-1, :]
        prob = torch.softmax(token_logits, dim=-1)
        m = torch.distributions.Categorical(prob)
        log_prob = m.log_prob(tokens[i])
        log_probs_all.append(log_prob)
    new_log_prob = torch.stack(log_probs_all).sum()
    value = value_head(hidden)
    advantage = reward - value
    ratio = torch.exp(new_log_prob - old_log_prob)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
    policy_loss = -torch.min(surr1, surr2)
    value_loss = (value - reward)**2
    loss = policy_loss + value_loss_coef * value_loss
    return loss, new_log_prob, value

#############################################
# 15. PPO RL Fine-Tuning Loop (Advanced RLHF)
#############################################
value_head = ValueHead(d_model=model.d_model).to(device)
ppo_optimizer = optim.Adam(list(model.parameters()) + list(value_head.parameters()), lr=1e-4)
ppo_epochs = 4  # Number of PPO updates per episode
num_ppo_steps = 20  # Total RL steps

print("\nStarting PPO RL fine-tuning ...")
for step in range(num_ppo_steps):
    model.eval()
    with torch.no_grad():
        generated_text, log_probs_tensor = generate_text_with_log_probs(model, tokenizer, prompt="Once upon a time", max_length=50)
        tokens_generated = tokenizer.encode(generated_text)
        tokens_tensor = torch.tensor(tokens_generated, dtype=torch.long).unsqueeze(0).to(device)
        old_log_prob = log_probs_tensor.sum().detach()
    reward = quality_reward_fn(generated_text)
    model.train()
    for _ in range(ppo_epochs):
        ppo_optimizer.zero_grad()
        loss, new_log_prob, value_est = ppo_update(model, value_head, tokens_tensor, old_log_prob, reward)
        loss.backward()
        ppo_optimizer.step()
    print(f"PPO Step {step+1}/{num_ppo_steps} - Reward: {reward:.4f}, Loss: {loss.item():.4f}, New Log Prob: {new_log_prob.item():.4f}, Value Estimate: {value_est.item():.4f}")
    print("Generated:", generated_text)
    print("-" * 40)


import torch
import torch.optim as optim

def quality_reward_fn(generated_text):
    tokens = generated_text.split()
    if len(tokens) == 0:
        return 0.0
    lexical_diversity = len(set(tokens)) / len(tokens)
    punctuation_bonus = 1.0 if generated_text.strip()[-1] in ".!?" else 0.5
    length_bonus = min(len(tokens), 50) / 50.0
    return lexical_diversity * punctuation_bonus * length_bonus

def ppo_update(model, value_head, tokens_tensor, old_log_prob, reward):
    hidden, logits = model(tokens_tensor, return_hidden=True)
    log_probs_all = []
    tokens = tokens_tensor[0]
    
    for i in range(1, tokens_tensor.size(1)):
        token_logits = logits[0, i-1, :]
        prob = torch.softmax(token_logits, dim=-1)
        m = torch.distributions.Categorical(prob)
        log_prob = m.log_prob(tokens[i])
        log_probs_all.append(log_prob)
    
    new_log_prob = torch.stack(log_probs_all).sum()
    value = value_head(hidden)
    advantage = reward - value
    ratio = torch.exp(new_log_prob - old_log_prob)
    
    policy_loss = -torch.min(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage)
    value_loss = (value - reward)**2
    loss = policy_loss + 0.5 * value_loss

    return loss, new_log_prob, value

# Example: Fine-tune with RLHF
ppo_optimizer = optim.Adam(model.parameters(), lr=1e-4)
prompt = "Tell me a joke"

# Generate text and reward
generated_text, log_probs_tensor = generate_text_with_log_probs(model, tokenizer, prompt, max_length=50)
reward = quality_reward_fn(generated_text)

# Perform PPO optimization
ppo_optimizer.zero_grad()
tokens_tensor = torch.tensor(tokenizer.encode(generated_text), dtype=torch.long).unsqueeze(0).to(device)
loss, new_log_prob, value_est = ppo_update(model, value_head, tokens_tensor, log_probs_tensor.sum(), reward)
loss.backward()
ppo_optimizer.step()

print(f"Updated response: {generated_text}")

#lu
import torch
import torch.nn as nn

class ImprovedTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=12, dropout=0.1):
        super(ImprovedTransformerLM, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        # Transformer encoder with multiple layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        seq_len = x.size(1)
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = self.dropout(token_emb + pos_emb)
        # Transformer expects input shape: [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        logits = self.fc_out(x)
        return logits

# Example usage:
# model = ImprovedTransformerLM(vocab_size=5000)

#rl

import torch
import torch.nn.functional as F

def ppo_update(model, optimizer, states, actions, old_log_probs, advantages, clip_epsilon=0.2):
    """
    PPO update step with gradient clipping and advantage normalization.
    
    Parameters:
      model: the policy network
      optimizer: optimizer for updating model parameters
      states: batch of state inputs
      actions: actions taken
      old_log_probs: log probabilities of actions from the previous policy
      advantages: computed advantages for each action
      clip_epsilon: clipping threshold for PPO
    """
    # Forward pass: get logits and new log probabilities
    logits = model(states)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    
    # Compute probability ratios
    ratios = torch.exp(new_log_probs - old_log_probs)
    
    # Compute surrogate loss components
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Backpropagation with gradient clipping
    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    return policy_loss.item()

# Example usage in your RL loop:
# loss = ppo_update(model, optimizer, states, actions, old_log_probs, advantages)

#rhlf
def compute_rlhf_reward(base_reward, human_rating, scaling_factor=0.1):
    """
    Adjust the base reward using human feedback.
    
    Parameters:
      base_reward: the original reward value from your RL signal.
      human_rating: a numerical rating from human evaluators (e.g., -1 for negative, +1 for positive).
      scaling_factor: determines the influence of human feedback on the final reward.
      
    Returns:
      adjusted_reward: the final reward after incorporating human feedback.
    """
    adjusted_reward = base_reward + scaling_factor * human_rating
    return adjusted_reward

# Example usage:
base_reward = 1.0
human_rating = 0.8  # Suppose human feedback is positive
final_reward = compute_rlhf_reward(base_reward, human_rating)
print("Final RLHF reward:", final_reward)



#############################################
# 16. Save the Model (after RL fine-tuning)
#############################################
torch.save(model.state_dict(), "custom_transformer_rlhf.pt")
print("Model saved as custom_transformer_rlhf.pt")

#############################################
# 17. Flask Web Application Integration
#############################################
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"response": "No prompt provided."})
    generated_text, _ = generate_text_with_log_probs(model, tokenizer, prompt, max_length=50)
    return jsonify({"response": generated_text})

#############################################
# 18. Additional ML and Deep Learning Modules
#############################################
def image_classification_example():
    resnet = tv_models.resnet18(pretrained=True)
    resnet.eval()
    dummy_image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = resnet(dummy_image)
    print("ResNet18 output shape:", output.shape)
    return output

def tensorflow_regression_example():
    model_tf = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    model_tf.compile(optimizer='adam', loss='mse')
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.rand(100, 1)
    model_tf.fit(X_dummy, y_dummy, epochs=5, verbose=0)
    loss = model_tf.evaluate(X_dummy, y_dummy, verbose=0)
    print("TensorFlow model MSE loss:", loss)
    return model_tf

# Uncomment to run these examples:
# image_classification_example()
# tensorflow_regression_example()

#############################################
# 19. Video Tutorial Creation
#############################################
def create_video_tutorial(text, output_filename="tutorial.mp4", duration=15):
    """
    Create a video tutorial using MoviePy.
    """
    clip = TextClip(text, fontsize=24, color='white', bg_color='black', size=(1280,720), method='caption')
    clip = clip.set_duration(duration)
    clip.write_videofile(output_filename, fps=24)
    print(f"Video tutorial saved as {output_filename}")

# Example usage:
# create_video_tutorial("Welcome to our advanced transformer tutorial. This video explains our PPO RL fine-tuning, retrieval-augmented generation, multi-task NSP, and interactive debugging.", "tutorial.mp4", 15)

#############################################
# 20. Advanced Features: Retrieval-Augmented Generation & Chain-of-Thought
#############################################
def retrieve_web_context(query, top_k=3):
    """
    Retrieve external web context using FAISS over TF-IDF vectors.
    """
    query_vec = tfidf.transform([query]).toarray().astype('float32')
    distances, indices = faiss_index.search(query_vec, top_k)
    retrieved_texts = [corpus[idx] for idx in indices[0]]
    return " ".join(retrieved_texts)

def chain_of_thought_generation(model, tokenizer, prompt, max_length=50):
    """
    Enhanced chain-of-thought generation that first retrieves external web context,
    then generates intermediate reasoning and a final answer.
    """
    external_context = retrieve_web_context(prompt)
    augmented_prompt = prompt + "\nWeb Context: " + external_context + "\nReasoning:"
    cot_text, _ = generate_text_with_log_probs(model, tokenizer, augmented_prompt, max_length=max_length//2)
    final_prompt = augmented_prompt + "\nFinal Answer:"
    final_text, _ = generate_text_with_log_probs(model, tokenizer, final_prompt, max_length=max_length)
    return f"Chain-of-Thought: {cot_text}\nFinal Answer: {final_text}"

#############################################
# 21. Interactive Debugging and Code Explanation (for coding tasks)
#############################################
@app.route("/debug", methods=["POST"])
def debug_code():
    data = request.get_json()
    code = data.get("code", "")
    if not code:
        return jsonify({"response": "No code provided."})
    explanation = f"This code attempts to: {code[:100]}... [explanation truncated]"
    return jsonify({"response": explanation})

#############################################
# 22. Flask Route for Advanced Generation with Chain-of-Thought
#############################################
@app.route("/advanced_generate", methods=["POST"])
def advanced_generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"response": "No prompt provided."})
    advanced_text = chain_of_thought_generation(model, tokenizer, prompt, max_length=50)
    return jsonify({"response": advanced_text})

#############################################
# 23. Performance Endpoint: Evaluate and Display Metrics
#############################################
def evaluate_model(model, dataset, batch_size=16):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            if model.use_nsp:
                lm_logits, _ = outputs
            else:
                lm_logits = outputs
            logits = lm_logits[:, :-1, :].contiguous().view(-1, vocab_size)
            targets = batch[:, 1:].contiguous().view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item() * (batch.size(0) * (batch.size(1) - 1))
            total_tokens += batch.size(0) * (batch.size(1) - 1)
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

val_texts = [ex["text"] for ex in combined_dataset.select(range(800, 1000))]
val_dataset = TextDataset(val_texts, tokenizer, max_length=64)

@app.route("/performance", methods=["GET"])
def performance():
    val_perplexity = evaluate_model(model, val_dataset)
    baselines = {"GPT-2": 35.0, "GPT-3": 25.0, "GPT-4": 18.0}
    comparisons = {name: {"baseline": baseline, "difference": val_perplexity - baseline} for name, baseline in baselines.items()}
    return jsonify({"validation_perplexity": val_perplexity, "comparisons": comparisons})

#############################################
# 24. Inference Speed Endpoint: Measure Accuracy & Speed
#############################################
def measure_inference_speed(model, tokenizer, prompt, max_length=50, num_trials=10):
    times = []
    total_tokens = 0
    for _ in range(num_trials):
        start_time = time.time()
        generated_text, _ = generate_text_with_log_probs(model, tokenizer, prompt, max_length=max_length)
        end_time = time.time()
        times.append(end_time - start_time)
        total_tokens += len(tokenizer.encode(generated_text))
    avg_time = sum(times) / len(times)
    tokens_per_sec = total_tokens / sum(times) if sum(times) > 0 else 0
    return avg_time, tokens_per_sec

@app.route("/inference_speed", methods=["GET"])
def inference_speed():
    test_prompt = "Once upon a time"
    avg_time, tokens_per_sec = measure_inference_speed(model, tokenizer, test_prompt, max_length=50, num_trials=10)
    return jsonify({"average_inference_time_seconds": avg_time, "tokens_per_second": tokens_per_sec})

#############################################
# 25. Uncertainty Estimation & Efficiency Enhancements
#############################################
def estimate_uncertainty(model, tokenizer, prompt, num_samples=10, max_length=50):
    model.train()  # Enable dropout during inference for uncertainty estimation
    outputs = []
    for _ in range(num_samples):
        generated_text, _ = generate_text_with_log_probs(model, tokenizer, prompt, max_length=max_length)
        outputs.append(generated_text)
    unique_count = len(set(outputs))
    uncertainty = 1 - (unique_count / num_samples)
    return uncertainty, outputs

@app.route("/uncertainty", methods=["POST"])
def uncertainty_endpoint():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"response": "No prompt provided."})
    uncertainty, samples = estimate_uncertainty(model, tokenizer, prompt, num_samples=10, max_length=50)
    return jsonify({"uncertainty": uncertainty, "samples": samples})

def quantize_model(model):
    model.cpu()
    model_int8 = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return model_int8

# Uncomment to quantize the model after training:
# model = quantize_model(model)

#############################################
# 26. Continual & Meta-Learning and Hyperparameter Tuning
#############################################
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    model_trial = TransformerLM(vocab_size=vocab_size, d_model=256, nhead=4, num_layers=6,
                                max_seq_length=64, dropout=dropout, use_nsp=True).to(device)
    optimizer_trial = optim.Adam(model_trial.parameters(), lr=lr)
    small_dataset = torch.utils.data.Subset(lm_dataset, list(range(100)))
    small_dataloader = DataLoader(small_dataset, batch_size=16, shuffle=True)
    model_trial.train()
    total_loss = 0.0
    for batch in small_dataloader:
        batch = batch.to(device)
        optimizer_trial.zero_grad()
        outputs = model_trial(batch)
        if model_trial.use_nsp:
            lm_logits, _ = outputs
        else:
            lm_logits = outputs
        logits = lm_logits[:, :-1, :].contiguous().view(-1, vocab_size)
        targets = batch[:, 1:].contiguous().view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer_trial.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(small_dataloader)
    perplexity = math.exp(avg_loss)
    return perplexity

def run_hyperparameter_tuning(n_trials=10):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

def continual_learning_update(new_texts, epochs=1, batch_size=16):
    new_dataset = TextDataset(new_texts, tokenizer, max_length=64)
    new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for batch in new_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            if model.use_nsp:
                lm_logits, _ = outputs
            else:
                lm_logits = outputs
            logits = lm_logits[:, :-1, :].contiguous().view(-1, vocab_size)
            targets = batch[:, 1:].contiguous().view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
    print("Continual learning update completed.")

import torch
import torch.nn as nn

class BestTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=16, dropout=0.1, max_seq_length=512):
        super(BestTransformerLM, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Use learnable positional embeddings for longer contexts
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        seq_len = x.size(1)
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = self.dropout(token_emb + pos_emb)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        logits = self.fc_out(x)
        return logits

# Example usage:
# best_model = BestTransformerLM(vocab_size=5000)


import torch
import torch.nn.functional as F

def advanced_ppo_update(model, optimizer, states, actions, old_log_probs, advantages, clip_epsilon=0.2, max_grad_norm=0.5):
    # Forward pass to obtain new logits and distributions
    logits = model(states)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    
    # Compute probability ratio and normalized advantages
    ratios = torch.exp(new_log_probs - old_log_probs)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Add entropy bonus for better exploration
    entropy_bonus = dist.entropy().mean()
    loss = policy_loss - 0.01 * entropy_bonus  # adjust entropy weight as needed

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    
    return loss.item()

# Example usage:
# loss = advanced_ppo_update(model, optimizer, states, actions, old_log_probs, advantages)

import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, features):
        x = F.relu(self.fc1(features))
        reward = self.fc2(x)
        return reward

# Assume you have a function to extract features from generated text:
def extract_features(generated_text):
    # This function could extract features like lexical diversity, sentence length,
    # sentiment scores, etc. For simplicity, we'll use a dummy vector.
    return torch.tensor([len(generated_text), generated_text.count(" "), 1.0])  # example features

# During RLHF, combine base reward with reward model output:
def compute_enhanced_reward(base_reward, generated_text, reward_model, scaling_factor=0.1):
    features = extract_features(generated_text).float().unsqueeze(0)  # shape [1, feature_dim]
    model_reward = reward_model(features)
    # Combine base reward with model-predicted reward adjusted by human feedback scaling
    adjusted_reward = base_reward + scaling_factor * model_reward.item()
    return adjusted_reward

# Example usage:
# Initialize reward model (feature_dim = 3 in our dummy example)
reward_model = RewardModel(input_dim=3)
base_reward = 1.0
generated_text = "This is an example generated response!"
enhanced_reward = compute_enhanced_reward(base_reward, generated_text, reward_model)
print("Enhanced RLHF reward:", enhanced_reward)

def enhanced_quality_reward(generated_text):
    # Compute basic quality metrics
    tokens = generated_text.split()
    if not tokens:
        return 0.0
    lexical_diversity = len(set(tokens)) / len(tokens)
    punctuation_bonus = 1.0 if generated_text.strip()[-1] in ".!?" else 0.5
    length_bonus = min(len(tokens), 50) / 50.0
    raw_reward = lexical_diversity * punctuation_bonus * length_bonus
    
    # Normalize the reward (example: scale between 0 and 1)
    normalized_reward = raw_reward / 1.0  # Adjust normalization factor as needed
    return normalized_reward

# Alternatively, use a reward model (as shown earlier) for more nuanced feedback:
class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, features):
        x = F.relu(self.fc1(features))
        return self.fc2(x)

def extract_features(generated_text):
    # Example feature extraction: [length, number of spaces, constant bias]
    return torch.tensor([len(generated_text), generated_text.count(" "), 1.0]).float()

def compute_enhanced_rlhf_reward(base_reward, generated_text, reward_model, scaling_factor=0.1):
    features = extract_features(generated_text).unsqueeze(0)  # Shape: [1, feature_dim]
    model_reward = reward_model(features)
    return base_reward + scaling_factor * model_reward.item()

# Within your training loop:
writer.add_scalar("Reward/Quality", reward, global_step)
writer.add_scalar("Loss/PPO", loss.item(), global_step)
writer.add_scalar("Advantage/Mean", advantages.mean().item(), global_step)

import torch.nn.functional as F

def advanced_ppo_update(model, optimizer, states, actions, old_log_probs, advantages, clip_epsilon=0.2, max_grad_norm=0.5):
    # Forward pass to get logits and distributions
    logits = model(states)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    
    # Compute ratios and normalize advantages
    ratios = torch.exp(new_log_probs - old_log_probs)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Entropy bonus to encourage exploration
    entropy_bonus = dist.entropy().mean()
    loss = policy_loss - 0.01 * entropy_bonus  # Adjust entropy coefficient as needed
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    
    return loss.item()

import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Create a model instance with current trial parameters
    model_trial = TransformerLM(
        vocab_size=vocab_size, 
        d_model=256, 
        nhead=4, 
        num_layers=6,
        max_seq_length=64,
        dropout=dropout,
        use_nsp=True
    ).to(device)
    
    optimizer_trial = optim.Adam(model_trial.parameters(), lr=lr)
    
    # Use a small subset for fast tuning
    small_dataset = torch.utils.data.Subset(lm_dataset, list(range(100)))
    small_dataloader = DataLoader(small_dataset, batch_size=16, shuffle=True)
    
    model_trial.train()
    total_loss = 0.0
    for batch in small_dataloader:
        batch = batch.to(device)
        optimizer_trial.zero_grad()
        outputs = model_trial(batch)
        lm_logits = outputs[0] if model_trial.use_nsp else outputs
        logits = lm_logits[:, :-1, :].contiguous().view(-1, vocab_size)
        targets = batch[:, 1:].contiguous().view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer_trial.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(small_dataloader)
    perplexity = math.exp(avg_loss)
    return perplexity

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("Best trial:", study.best_trial)

#############################################
# 27. Additional Reasoning Functions
#############################################
# Reasoning for math problems
def reasoning_math_problem(equation_str):
    result = solve_math_problem(equation_str)
    return f"Solution for the equation '{equation_str} = 0': {result}"

# Reasoning for world problems (using chain-of-thought generation)
def reasoning_world_problem(problem_statement):
    result = solve_world_problem(problem_statement)
    return f"Reasoning for the world problem:\n{result}"
    
    #dan mode def creative_response(prompt, dan_mode=False):
    base_response = chain_of_thought_generation(model, tokenizer, prompt, max_length=50)
    if dan_mode:
        # Append creative ideas (this can be generated by another module or prompt)
        creative_ideas = "Additional creative insight: Consider using a hybrid architecture that combines neural and symbolic reasoning."
        return base_response + "\n" + creative_ideas
    return base_response

#ocr
try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Install pytesseract and Pillow for OCR functionality.")

def perform_ocr(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

#esp32

]import serial, time

esp32 = serial.Serial('/dev/ttyUSB0', 115200)
def control_esp32(command):
    esp32.write(command.encode())
    time.sleep(0.5)
    return esp32.readline().decode().strip()

#os and terminal

def execute_os_command(command):
    import subprocess
    try:
        result = subprocess.run(command.split(), capture_output=True, text=True)
        return result.stdout if result.stdout else result.stderr
    except Exception as e:
        return str(e)

#test code 

import subprocess

def run_generated_code(code_str):
    try:
        process = subprocess.Popen(["python3", "-c", code_str],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        output, error = process.communicate(timeout=10)
        return output if output else error
    except subprocess.TimeoutExpired:
        return "Execution timed out."

chat_memory = []

def store_memory(user_input, ai_response):
    chat_memory.append((user_input, ai_response))
    if len(chat_memory) > 100:  # Prevent overflow
        chat_memory.pop(0)

def recall_last_conversation():
    return "\n".join([f"User: {q} | AI: {a}" for q, a in chat_memory[-100000:]])  # Show last 5 interactions

def solve_math_problem(problem):
    if "plus" in problem:
        numbers = [int(x) for x in problem.split() if x.isdigit()]
        return sum(numbers)
    elif "times" in problem:
        numbers = [int(x) for x in problem.split() if x.isdigit()]
        result = 1
        for num in numbers:
            result *= num
        return result
    return "I can't solve that yet."

print(solve_math_problem("What is 5 plus 3?"))  # Output: a

from sympy import symbols, Eq, solve

def solve_logic_problem(statement):
    x, y = symbols('x y')
    eq1 = Eq(2*x + y, 10)  # Example: 2x + y = 10
    eq2 = Eq(x - y, 4)  # Example: x - y = 4
    solution = solve((eq1, eq2), (x, y))
    return solution

print(solve_logic_problem("x and y relationship"))

feedback_db = {}

def store_feedback(user_input, ai_response, feedback):
    feedback_db[user_input] = {"response": ai_response, "feedback": feedback}

def improve_response(user_input):
    if user_input in feedback_db and feedback_db[user_input]["feedback"] == "bad":
        return "I will improve my answer!"  # Replace with actual re-training logic
    return feedback_db.get(user_input, {}).get("response", "I don't know.")

# Example Usage
store_feedback("What is AI?", "AI is a technology.", "bad")
print(improve_response("What is AI?"))  # AI will learn from mistakes

import torch
import torch.optim as optim

def quality_reward_fn(generated_text):
    tokens = generated_text.split()
    if len(tokens) == 0:
        return 0.0
    lexical_diversity = len(set(tokens)) / len(tokens)
    punctuation_bonus = 1.0 if generated_text.strip()[-1] in ".!?" else 0.5
    length_bonus = min(len(tokens), 50) / 50.0
    return lexical_diversity * punctuation_bonus * length_bonus

def ppo_update(model, value_head, tokens_tensor, old_log_prob, reward):
    hidden, logits = model(tokens_tensor, return_hidden=True)
    log_probs_all = []
    tokens = tokens_tensor[0]
    
    for i in range(1, tokens_tensor.size(1)):
        token_logits = logits[0, i-1, :]
        prob = torch.softmax(token_logits, dim=-1)
        m = torch.distributions.Categorical(prob)
        log_prob = m.log_prob(tokens[i])
        log_probs_all.append(log_prob)
    
    new_log_prob = torch.stack(log_probs_all).sum()
    value = value_head(hidden)
    advantage = reward - value
    ratio = torch.exp(new_log_prob - old_log_prob)
    
    policy_loss = -torch.min(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage)
    value_loss = (value - reward)**2
    loss = policy_loss + 0.5 * value_loss

    return loss, new_log_prob, value

# Example: Fine-tune with RLHF
ppo_optimizer = optim.Adam(model.parameters(), lr=1e-4)
prompt = "Tell me a joke"

# Generate text and reward
generated_text, log_probs_tensor = generate_text_with_log_probs(model, tokenizer, prompt, max_length=50)
reward = quality_reward_fn(generated_text)

# Perform PPO optimization
ppo_optimizer.zero_grad()
tokens_tensor = torch.tensor(tokenizer.encode(generated_text), dtype=torch.long).unsqueeze(0).to(device)
loss, new_log_prob, value_est = ppo_update(model, value_head, tokens_tensor, log_probs_tensor.sum(), reward)
loss.backward()
ppo_optimizer.step()

print(f"Updated response: {generated_text}")


#############################################
# 28. External Knowledge API Integration (Example Route)
#############################################
@app.route("/external_query", methods=["POST"])
def external_query():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"response": "No query provided."})
    context = retrieve_web_context(query)
    return jsonify({"response": context})

#############################################
# 29. Run Flask App
#############################################
if __name__ == "__main__":
    app.run(debug=True)
