# 🤖 Simple ChatGPT Implementation

A complete implementation of a GPT-style language model in Go, based on the **"Attention Is All You Need"** paper (Vaswani et al., 2017).

## 📚 Overview

This project implements the core Transformer architecture from scratch, including:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Layer Normalization
- Feed-Forward Networks
- Causal Masking for autoregressive generation

## 🏗️ Architecture

```
Input Text
    │
    ▼
┌─────────────┐
│  Tokenizer  │  (character-level)
└─────────────┘
    │
    ▼
┌─────────────┐
│  Embedding  │  (learned word vectors)
└─────────────┘
    │
    ▼
┌─────────────┐
│  Positional │  (sinusoidal encoding)
│  Encoding   │
└─────────────┘
    │
    ▼
┌─────────────────────────────────┐
│       Decoder Layer × N         │
│  ┌───────────────────────────┐  │
│  │     Layer Norm            │  │
│  │          ↓                │  │
│  │  Masked Multi-Head        │  │
│  │  Self-Attention           │←─┼── Residual Connection
│  │          ↓                │  │
│  │     Layer Norm            │  │
│  │          ↓                │  │
│  │  Feed-Forward Network     │←─┼── Residual Connection
│  │  (Linear → ReLU → Linear) │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────┐
│   Output    │  (project to vocabulary)
│  Projection │
└─────────────┘
    │
    ▼
┌─────────────┐
│   Softmax   │  (probability distribution)
└─────────────┘
    │
    ▼
Next Token
```

## 🧮 Key Algorithms

### 1. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

- **Q (Query)**: What the current token is looking for
- **K (Key)**: What each token has to offer
- **V (Value)**: The actual information content
- **√d_k scaling**: Prevents softmax from becoming too peaked

### 2. Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × W_O

where head_i = Attention(Q × W_Q^i, K × W_K^i, V × W_V^i)
```

Multiple attention heads learn different relationship patterns in parallel.

### 3. Causal Masking

For autoregressive generation, tokens can only attend to previous positions:

```
Position 0: can see [0]
Position 1: can see [0, 1]
Position 2: can see [0, 1, 2]
...
```

Future positions are masked with -∞ before softmax.

### 4. Positional Encoding

Since attention has no inherent order, we add position information:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 5. Training (Next-Token Prediction)

```
Loss = -log P(correct_next_token | previous_tokens)
```

The model learns to predict the next character given the context.

## 🚀 Quick Start

### Run the ChatGPT Demo

```bash
go run simple_llm.go transformer.go
```

Select mode:
- **1**: Demo mode - shows architecture, training, and generation
- **2**: Chat mode - interactive conversation

### Study the Attention Mechanism

```bash
go run study_attention.go transformer.go
```

Step-by-step walkthrough of:
- Matrix operations
- Scaled dot-product attention
- Multi-head attention
- Causal masking
- Positional encoding

### Run Attention Pattern Tests

```bash
go run transformer_test.go transformer.go
```

Tests different attention behaviors:
- Identity (self-attention)
- Uniform (global averaging)
- Sequential (local/neighbor attention)
- Random matrices

## 📁 File Structure

| File | Description |
|------|-------------|
| `transformer.go` | Core matrix operations and attention |
| `simple_llm.go` | Complete LLM with training & chat |
| `study_attention.go` | Educational walkthrough |
| `transformer_test.go` | Attention pattern tests |
| `test_inputs.json` | Test matrices for experiments |

## 💬 How Chat Works

1. **Input Processing**
   ```
   "hello" → tokenize → [12, 9, 16, 16, 19]
   ```

2. **Forward Pass**
   ```
   tokens → embeddings → +positional → decoder layers → logits
   ```

3. **Generation** (Temperature Sampling)
   ```python
   # Apply temperature to logits
   scaled_logits = logits / temperature
   
   # Top-K filtering
   top_k_logits = select_top_k(scaled_logits, k=10)
   
   # Sample from distribution
   next_token = sample(softmax(top_k_logits))
   ```

4. **Loop** until max length or end token

### Temperature Effects

| Temperature | Behavior |
|-------------|----------|
| 0.1 | Very deterministic, repetitive |
| 0.7 | Balanced creativity (default) |
| 1.0 | More random, diverse |
| 1.5+ | Very random, may be incoherent |

## 🎓 Training

The model uses **next-token prediction**:

```
Input:  "hello ther"  →  Target: "ello there"
```

For each position, minimize cross-entropy loss between predicted and actual next token.

### Training Parameters

```go
embeddingDim := 64   // Vector size for each token
numHeads := 4        // Parallel attention heads
numLayers := 1       // Decoder layers (GPT-2 has 12-48)
dFF := 128           // Feed-forward hidden size
learningRate := 0.05 // Gradient step size
epochs := 100        // Training iterations
```

## 📖 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [GPT-2](https://openai.com/research/better-language-models) - Decoder-only architecture
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation

## 🔧 Model Comparison

| Component | This Implementation | GPT-2 Small | GPT-3 |
|-----------|---------------------|-------------|-------|
| Parameters | ~50K | 117M | 175B |
| Layers | 1-2 | 12 | 96 |
| Heads | 4 | 12 | 96 |
| d_model | 64 | 768 | 12288 |
| Vocab Size | 49 | 50257 | 50257 |

This is a **learning implementation** - real LLMs need much more data and compute!

## ✨ Features

- ✅ Scaled Dot-Product Attention
- ✅ Multi-Head Attention with learned projections
- ✅ Causal Masking for autoregressive generation
- ✅ Sinusoidal Positional Encoding
- ✅ Layer Normalization (Pre-LN)
- ✅ Feed-Forward Networks with ReLU
- ✅ Residual Connections
- ✅ Cross-Entropy Loss
- ✅ Temperature Sampling
- ✅ Top-K Sampling
- ✅ Interactive Chat Interface

## 📝 License

Educational use - learn and experiment!

