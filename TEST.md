# 🧪 Testing Guide

This document explains how to test each module in the Transformer implementation.

## 📋 Test Overview

| Module | Test File | Command |
|--------|-----------|---------|
| Matrix Operations | `transformer.go` | `go run transformer_test.go transformer.go` |
| Attention Patterns | `transformer_test.go` | `go run transformer_test.go transformer.go` |
| Complete LLM | `simple_llm.go` | `go run simple_llm.go transformer.go` |
| Study Examples | `study_attention.go` | `go run study_attention.go transformer.go` |

---

## 1. Matrix Operations (`transformer.go`)

### What to Test

| Operation | Formula | Expected Behavior |
|-----------|---------|-------------------|
| `NewMatrix(r, c)` | Create r×c matrix | All zeros |
| `Transpose()` | A^T[i][j] = A[j][i] | Rows ↔ Columns |
| `Multiply(B)` | C[i][j] = Σ A[i][k]×B[k][j] | Standard matrix mult |
| `Scale(s)` | A[i][j] × s | Element-wise scaling |
| `Softmax()` | e^x_i / Σe^x_j | Row sums = 1.0 |

### Manual Test

```go
// In Go playground or test file:
A := Matrix{{1, 2}, {3, 4}}
B := Matrix{{5, 6}, {7, 8}}

// Test multiplication
C := A.Multiply(B)
// Expected: [[19, 22], [43, 50]]

// Test transpose
AT := A.Transpose()
// Expected: [[1, 3], [2, 4]]

// Test softmax
logits := Matrix{{1.0, 2.0, 3.0}}
probs := logits.Softmax()
// Expected: [[0.09, 0.24, 0.67]] (sums to 1.0)
```

### Run Test

```bash
go run study_attention.go transformer.go
```

Look for "PART 1: MATRIX OPERATIONS" section.

---

## 2. Scaled Dot-Product Attention

### What to Test

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

| Step | Operation | Verify |
|------|-----------|--------|
| 1 | QK^T | Similarity scores matrix |
| 2 | Scale by 1/√d_k | Values reduced |
| 3 | Softmax | Each row sums to 1.0 |
| 4 | Multiply by V | Weighted combination |

### Test Cases

#### Test 1: Identity Attention (Self-Focus)

```bash
go run transformer_test.go transformer.go
```

**Input:**
- Q = K = Identity matrix (8×8)
- V = Sequential values

**Expected:**
- Each token attends 100% to itself
- Output ≈ Input values

#### Test 2: Uniform Attention (Global Average)

**Input:**
- Q = K = All ones matrix

**Expected:**
- All attention weights = 1/n (equal)
- Output = Average of all V rows

#### Test 3: Sequential Attention (Local Focus)

**Input:**
- Q = K = Tridiagonal matrix (1.0 on diagonal, 0.5 adjacent)

**Expected:**
- Higher attention to self and neighbors
- Lower attention to distant tokens

### Verify Attention Weights

```go
// Attention weights must:
// 1. Be non-negative
// 2. Sum to 1.0 per row
// 3. Reflect Q-K similarity

for i := 0; i < len(weights); i++ {
    sum := 0.0
    for j := 0; j < len(weights[i]); j++ {
        assert(weights[i][j] >= 0)  // Non-negative
        sum += weights[i][j]
    }
    assert(abs(sum - 1.0) < 0.0001)  // Sums to 1
}
```

---

## 3. Multi-Head Attention

### What to Test

| Component | Test |
|-----------|------|
| Head splitting | Input correctly divided into h parts |
| Per-head attention | Each head runs independently |
| Concatenation | Outputs combined correctly |
| Output projection | Final dimension = d_model |

### Run Test

```bash
go run transformer_test.go transformer.go
```

Look for "Multi-Head Attention (2 heads)" sections.

### Verify

```
Input shape:  [seq_len, d_model]
Per head:     [seq_len, d_model/num_heads]
Output shape: [seq_len, d_model]
```

---

## 4. Causal Masking

### What to Test

Tokens should only attend to previous positions (not future).

### Test Case

```go
// For sequence length 4:
// Position 0 can see: [0]
// Position 1 can see: [0, 1]
// Position 2 can see: [0, 1, 2]
// Position 3 can see: [0, 1, 2, 3]

// Attention weights should be:
// [w00,   0,   0,   0]   <- row 0
// [w10, w11,  0,   0]   <- row 1
// [w20, w21, w22,  0]   <- row 2
// [w30, w31, w32, w33]  <- row 3
```

### Run Test

```bash
go run study_attention.go transformer.go
```

Look for "PART 4: CAUSAL MASKING" section.

### Verify

```go
// After masking, future positions should be 0:
for i := 0; i < seqLen; i++ {
    for j := i + 1; j < seqLen; j++ {
        assert(attentionWeights[i][j] == 0.0)
    }
}
```

---

## 5. Positional Encoding

### What to Test

| Property | Expected |
|----------|----------|
| Shape | [seq_len, d_model] |
| Values | Between -1 and 1 (sin/cos) |
| Uniqueness | Each position has different encoding |
| Pattern | Even dims = sin, Odd dims = cos |

### Run Test

```bash
go run study_attention.go transformer.go
```

Look for "PART 5: POSITIONAL ENCODING" section.

### Verify

```go
// Check formula:
// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

for pos := 0; pos < seqLen; pos++ {
    for i := 0; i < dModel; i++ {
        expected := 0.0
        if i % 2 == 0 {
            expected = math.Sin(float64(pos) / math.Pow(10000, float64(i)/float64(dModel)))
        } else {
            expected = math.Cos(float64(pos) / math.Pow(10000, float64(i-1)/float64(dModel)))
        }
        assert(abs(pe[pos][i] - expected) < 0.0001)
    }
}
```

---

## 6. Layer Normalization

### What to Test

| Property | Expected |
|----------|----------|
| Mean | ≈ 0 (after normalization) |
| Variance | ≈ 1 (after normalization) |
| Shape | Same as input |

### Verify

```go
// After LayerNorm, each row should have:
// mean ≈ 0, variance ≈ 1

for i := 0; i < seqLen; i++ {
    mean := 0.0
    for j := 0; j < dim; j++ {
        mean += output[i][j]
    }
    mean /= float64(dim)
    assert(abs(mean) < 0.01)  // Mean ≈ 0
    
    variance := 0.0
    for j := 0; j < dim; j++ {
        diff := output[i][j] - mean
        variance += diff * diff
    }
    variance /= float64(dim)
    assert(abs(variance - 1.0) < 0.1)  // Variance ≈ 1
}
```

---

## 7. Feed-Forward Network

### What to Test

```
FFN(x) = ReLU(x × W1 + b1) × W2 + b2
```

| Property | Expected |
|----------|----------|
| ReLU | Negative values → 0 |
| Shape | Input [n, d_model] → Output [n, d_model] |
| Hidden | Intermediate dimension = d_ff |

### Verify

```go
// ReLU test: no negative values in hidden layer
for i := 0; i < seqLen; i++ {
    for j := 0; j < dFF; j++ {
        assert(hidden[i][j] >= 0)  // ReLU output
    }
}
```

---

## 8. Complete LLM Pipeline

### Run Full Test

```bash
go run simple_llm.go transformer.go
# Select option 1 for demo mode
```

### What to Verify

| Stage | Test |
|-------|------|
| Tokenization | "hello" → tokens → "hello" |
| Embedding | tokens → [seq_len, embed_dim] |
| Forward Pass | Input → logits [seq_len, vocab_size] |
| Training | Loss decreases over epochs |
| Generation | Produces readable text |

### Expected Output

```
📝 TOKENIZATION DEMO
  Original: "hello world"
  Tokens: [1 12 9 16 16 19 4 27 19 22 16 8 2]
  Decoded: "hello world"  ← Should match!

⚡ FORWARD PASS DEMO
  Input: "hello" -> tokens: [12 9 16 16 19]
  Output logits shape: 5x49  ← [seq_len × vocab_size]

🎓 TRAINING THE MODEL
  Epoch 1/100 - Loss: 4.6254
  ...
  Epoch 100/100 - Loss: 3.4621  ← Should decrease!
```

---

## 9. Chat Mode Testing

### Run Interactive Test

```bash
go run simple_llm.go transformer.go
# Select option 2 for chat mode
```

### Test Inputs

| Input | Expected Response Type |
|-------|------------------------|
| `hello` | Greeting response |
| `how are you` | Status response |
| `bye` | Farewell response |
| `help` | Help information |
| `random text` | Generated continuation |

### Chat Test Script

```bash
# Automated test (pipe inputs)
echo -e "2\nhello\nhow are you\nbye\nquit" | go run simple_llm.go transformer.go
```

---

## 10. Performance Testing

### Memory Usage

```bash
# Check memory with verbose GC
GODEBUG=gctrace=1 go run simple_llm.go transformer.go
```

### Timing

```go
// Add timing to critical sections:
start := time.Now()
logits := llm.Forward(tokens)
fmt.Printf("Forward pass: %v\n", time.Since(start))
```

---

## 🔍 Debugging Tips

### Print Intermediate Values

```go
// In transformer.go, add prints:
func ScaledDotProductAttention(Q, K, V Matrix) Matrix {
    KT := K.Transpose()
    KT.Print("K Transpose")  // Debug print
    
    QKT := Q.Multiply(KT)
    QKT.Print("QK^T")  // Debug print
    // ...
}
```

### Check for NaN/Inf

```go
func checkMatrix(m Matrix, name string) {
    for i := range m {
        for j := range m[i] {
            if math.IsNaN(m[i][j]) || math.IsInf(m[i][j], 0) {
                fmt.Printf("WARNING: %s[%d][%d] = %v\n", name, i, j, m[i][j])
            }
        }
    }
}
```

### Verify Dimensions

```go
func assertShape(m Matrix, expectedRows, expectedCols int, name string) {
    if len(m) != expectedRows || len(m[0]) != expectedCols {
        panic(fmt.Sprintf("%s: expected %dx%d, got %dx%d", 
            name, expectedRows, expectedCols, len(m), len(m[0])))
    }
}
```

---

## ✅ Test Checklist

- [ ] Matrix multiplication produces correct results
- [ ] Softmax rows sum to 1.0
- [ ] Attention weights are non-negative
- [ ] Causal mask blocks future positions
- [ ] Positional encodings are unique per position
- [ ] Layer norm outputs have mean≈0, var≈1
- [ ] Training loss decreases over epochs
- [ ] Tokenize → Decode returns original text
- [ ] Chat responds to basic inputs
- [ ] No NaN or Inf values in outputs

---

## 📊 Test Data Files

| File | Contents | Used By |
|------|----------|---------|
| `test_inputs.json` | Identity, Uniform, Sequential matrices | `transformer_test.go` |
| `training_data.json` | Text completion examples | Reference data |

### View Test Data

```bash
cat test_inputs.json | head -50
```

---

## 🚀 Run All Tests

```bash
# Quick test - all modules
echo "Running Matrix & Attention Tests..."
go run transformer_test.go transformer.go

echo "Running Study Guide..."
go run study_attention.go transformer.go

echo "Running LLM Demo..."
echo "1" | go run simple_llm.go transformer.go
```

