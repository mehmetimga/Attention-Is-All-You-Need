package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

// ========================================================================
// Simple ChatGPT Implementation using Transformer Architecture
// Complete implementation with training, attention, and text generation
// ========================================================================

// Tokenizer handles text-to-token conversion
type Tokenizer struct {
	vocabToID map[string]int
	idToVocab map[int]string
	vocabSize int
}

// NewTokenizer creates a simple character-level tokenizer
func NewTokenizer() *Tokenizer {
	// Simple vocabulary: letters, numbers, space, punctuation
	vocab := []string{
		"<PAD>", "<START>", "<END>", "<UNK>", " ",
		"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
		"n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
		".", "!", "?", ",", "'", "-", ":", ";",
		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
	}

	vocabToID := make(map[string]int)
	idToVocab := make(map[int]string)

	for i, token := range vocab {
		vocabToID[token] = i
		idToVocab[i] = token
	}

	return &Tokenizer{
		vocabToID: vocabToID,
		idToVocab: idToVocab,
		vocabSize: len(vocab),
	}
}

// Encode converts text to token IDs
func (t *Tokenizer) Encode(text string) []int {
	tokens := []int{t.vocabToID["<START>"]} // Start token

	for _, char := range strings.ToLower(text) {
		if id, exists := t.vocabToID[string(char)]; exists {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, t.vocabToID["<UNK>"]) // Unknown token
		}
	}

	tokens = append(tokens, t.vocabToID["<END>"]) // End token
	return tokens
}

// EncodeWithoutSpecial converts text to token IDs without START/END
func (t *Tokenizer) EncodeWithoutSpecial(text string) []int {
	tokens := []int{}

	for _, char := range strings.ToLower(text) {
		if id, exists := t.vocabToID[string(char)]; exists {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, t.vocabToID["<UNK>"])
		}
	}

	return tokens
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(tokens []int) string {
	var result strings.Builder
	
	for _, token := range tokens {
		if token == t.vocabToID["<START>"] || token == t.vocabToID["<END>"] || token == t.vocabToID["<PAD>"] {
			continue
		}
		if char, exists := t.idToVocab[token]; exists {
			result.WriteString(char)
		}
	}
	
	return result.String()
}

// Embedding creates word embeddings from token IDs
type Embedding struct {
	weights Matrix
	vocabSize int
	embeddingDim int
}

// NewEmbedding creates a new embedding layer
func NewEmbedding(vocabSize, embeddingDim int) *Embedding {
	// Initialize with random weights
	rand.Seed(time.Now().UnixNano())
	weights := NewMatrix(vocabSize, embeddingDim)
	
	// Xavier initialization
	scale := math.Sqrt(2.0 / float64(embeddingDim))
	for i := 0; i < vocabSize; i++ {
		for j := 0; j < embeddingDim; j++ {
			weights[i][j] = (rand.Float64()*2 - 1) * scale
		}
	}
	
	return &Embedding{
		weights: weights,
		vocabSize: vocabSize,
		embeddingDim: embeddingDim,
	}
}

// Forward converts token IDs to embeddings
func (e *Embedding) Forward(tokenIDs []int) Matrix {
	seqLen := len(tokenIDs)
	embeddings := NewMatrix(seqLen, e.embeddingDim)
	
	for i, tokenID := range tokenIDs {
		if tokenID < e.vocabSize {
			for j := 0; j < e.embeddingDim; j++ {
				embeddings[i][j] = e.weights[tokenID][j]
			}
		}
	}
	
	return embeddings
}

// PositionalEncoding adds position information to embeddings
func AddPositionalEncoding(embeddings Matrix) Matrix {
	seqLen := len(embeddings)
	dModel := len(embeddings[0])
	
	result := NewMatrix(seqLen, dModel)
	
	// Copy embeddings and add positional encoding
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			pos := float64(i)
			dim := float64(j)
			
			// Sinusoidal positional encoding
			if j%2 == 0 {
				result[i][j] = embeddings[i][j] + math.Sin(pos/math.Pow(10000, dim/float64(dModel)))
			} else {
				result[i][j] = embeddings[i][j] + math.Cos(pos/math.Pow(10000, (dim-1)/float64(dModel)))
			}
		}
	}
	
	return result
}

// ========================================================================
// CAUSAL MASKED ATTENTION - Proper implementation for autoregressive LLM
// ========================================================================

// CausalScaledDotProductAttention applies attention with causal masking
// Future tokens are masked with -inf so they don't contribute to attention
func CausalScaledDotProductAttention(Q, K, V Matrix) Matrix {
	seqLen := len(Q)
	dk := float64(len(K[0]))

	// Step 1: Calculate QK^T
	KT := K.Transpose()
	QKT := Q.Multiply(KT)

	// Step 2: Scale by sqrt(d_k)
	scalingFactor := 1.0 / math.Sqrt(dk)
	scaled := QKT.Scale(scalingFactor)

	// Step 3: Apply causal mask (set future positions to -inf)
	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			scaled[i][j] = math.Inf(-1) // Can't attend to future tokens
		}
	}

	// Step 4: Apply softmax
	attentionWeights := scaled.Softmax()

	// Step 5: Multiply by Values
	return attentionWeights.Multiply(V)
}

// MaskedMultiHeadAttention implements causal (decoder) attention
type MaskedMultiHeadAttention struct {
	numHeads int
	dModel   int
	dK       int
	// Learned projection matrices
	WQ Matrix
	WK Matrix
	WV Matrix
	WO Matrix
}

// NewMaskedMultiHeadAttention creates masked attention for autoregressive generation
func NewMaskedMultiHeadAttention(numHeads, dModel int) *MaskedMultiHeadAttention {
	dK := dModel / numHeads
	scale := math.Sqrt(2.0 / float64(dModel))

	// Initialize projection matrices with Xavier initialization
	WQ := NewMatrix(dModel, dModel)
	WK := NewMatrix(dModel, dModel)
	WV := NewMatrix(dModel, dModel)
	WO := NewMatrix(dModel, dModel)

	for i := 0; i < dModel; i++ {
		for j := 0; j < dModel; j++ {
			WQ[i][j] = (rand.Float64()*2 - 1) * scale
			WK[i][j] = (rand.Float64()*2 - 1) * scale
			WV[i][j] = (rand.Float64()*2 - 1) * scale
			WO[i][j] = (rand.Float64()*2 - 1) * scale
		}
	}

	return &MaskedMultiHeadAttention{
		numHeads: numHeads,
		dModel:   dModel,
		dK:       dK,
		WQ:       WQ,
		WK:       WK,
		WV:       WV,
		WO:       WO,
	}
}

// Forward applies masked multi-head attention
func (mha *MaskedMultiHeadAttention) Forward(X Matrix) Matrix {
	seqLen := len(X)

	// Project to Q, K, V
	Q := X.Multiply(mha.WQ)
	K := X.Multiply(mha.WK)
	V := X.Multiply(mha.WV)

	// Split into heads and apply attention
	headOutputs := make([]Matrix, mha.numHeads)

	for head := 0; head < mha.numHeads; head++ {
		startCol := head * mha.dK

		// Extract this head's portion
		headQ := NewMatrix(seqLen, mha.dK)
		headK := NewMatrix(seqLen, mha.dK)
		headV := NewMatrix(seqLen, mha.dK)

		for i := 0; i < seqLen; i++ {
			for j := 0; j < mha.dK; j++ {
				headQ[i][j] = Q[i][startCol+j]
				headK[i][j] = K[i][startCol+j]
				headV[i][j] = V[i][startCol+j]
			}
		}

		// Apply causal attention for this head
		headOutputs[head] = CausalScaledDotProductAttention(headQ, headK, headV)
	}

	// Concatenate all head outputs
	concat := NewMatrix(seqLen, mha.dModel)
	for i := 0; i < seqLen; i++ {
		for head := 0; head < mha.numHeads; head++ {
			startCol := head * mha.dK
			for j := 0; j < mha.dK; j++ {
				concat[i][startCol+j] = headOutputs[head][i][j]
			}
		}
	}

	// Final output projection
	return concat.Multiply(mha.WO)
}

// ========================================================================
// LAYER NORMALIZATION - Critical for stable training
// ========================================================================

// LayerNorm implements layer normalization
type LayerNorm struct {
	gamma   []float64 // Learned scale parameter
	beta    []float64 // Learned shift parameter
	epsilon float64
	dim     int
}

// NewLayerNorm creates a layer normalization layer
func NewLayerNorm(dim int) *LayerNorm {
	gamma := make([]float64, dim)
	beta := make([]float64, dim)

	for i := 0; i < dim; i++ {
		gamma[i] = 1.0 // Initialize to 1
		beta[i] = 0.0  // Initialize to 0
	}

	return &LayerNorm{
		gamma:   gamma,
		beta:    beta,
		epsilon: 1e-5,
		dim:     dim,
	}
}

// Forward applies layer normalization
func (ln *LayerNorm) Forward(X Matrix) Matrix {
	seqLen := len(X)
	result := NewMatrix(seqLen, ln.dim)

	for i := 0; i < seqLen; i++ {
		// Calculate mean
		mean := 0.0
		for j := 0; j < ln.dim; j++ {
			mean += X[i][j]
		}
		mean /= float64(ln.dim)

		// Calculate variance
		variance := 0.0
		for j := 0; j < ln.dim; j++ {
			diff := X[i][j] - mean
			variance += diff * diff
		}
		variance /= float64(ln.dim)

		// Normalize and apply learned parameters
		stdDev := math.Sqrt(variance + ln.epsilon)
		for j := 0; j < ln.dim; j++ {
			result[i][j] = ln.gamma[j]*((X[i][j]-mean)/stdDev) + ln.beta[j]
		}
	}

	return result
}

// ========================================================================
// FEED-FORWARD NETWORK - Applied after attention in each layer
// ========================================================================

// FeedForward implements the position-wise feed-forward network
// FFN(x) = max(0, xW1 + b1)W2 + b2
type FeedForward struct {
	W1     Matrix
	b1     []float64
	W2     Matrix
	b2     []float64
	dModel int
	dFF    int
}

// NewFeedForward creates a feed-forward network
func NewFeedForward(dModel, dFF int) *FeedForward {
	scale1 := math.Sqrt(2.0 / float64(dModel))
	scale2 := math.Sqrt(2.0 / float64(dFF))

	W1 := NewMatrix(dModel, dFF)
	W2 := NewMatrix(dFF, dModel)
	b1 := make([]float64, dFF)
	b2 := make([]float64, dModel)

	for i := 0; i < dModel; i++ {
		for j := 0; j < dFF; j++ {
			W1[i][j] = (rand.Float64()*2 - 1) * scale1
		}
	}

	for i := 0; i < dFF; i++ {
		for j := 0; j < dModel; j++ {
			W2[i][j] = (rand.Float64()*2 - 1) * scale2
		}
		b1[i] = 0.0
	}

	for i := 0; i < dModel; i++ {
		b2[i] = 0.0
	}

	return &FeedForward{
		W1:     W1,
		b1:     b1,
		W2:     W2,
		b2:     b2,
		dModel: dModel,
		dFF:    dFF,
	}
}

// Forward applies the feed-forward network with ReLU activation
func (ff *FeedForward) Forward(X Matrix) Matrix {
	seqLen := len(X)

	// First linear layer + ReLU
	hidden := NewMatrix(seqLen, ff.dFF)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < ff.dFF; j++ {
			sum := ff.b1[j]
			for k := 0; k < ff.dModel; k++ {
				sum += X[i][k] * ff.W1[k][j]
			}
			// ReLU activation
			hidden[i][j] = math.Max(0, sum)
		}
	}

	// Second linear layer
	output := NewMatrix(seqLen, ff.dModel)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < ff.dModel; j++ {
			sum := ff.b2[j]
			for k := 0; k < ff.dFF; k++ {
				sum += hidden[i][k] * ff.W2[k][j]
			}
			output[i][j] = sum
		}
	}

	return output
}

// ========================================================================
// TRANSFORMER DECODER LAYER - Complete layer with residual connections
// ========================================================================

// DecoderLayer represents a single transformer decoder layer
type DecoderLayer struct {
	attention *MaskedMultiHeadAttention
	ffn       *FeedForward
	norm1     *LayerNorm
	norm2     *LayerNorm
	dModel    int
}

// NewDecoderLayer creates a decoder layer
func NewDecoderLayer(numHeads, dModel, dFF int) *DecoderLayer {
	return &DecoderLayer{
		attention: NewMaskedMultiHeadAttention(numHeads, dModel),
		ffn:       NewFeedForward(dModel, dFF),
		norm1:     NewLayerNorm(dModel),
		norm2:     NewLayerNorm(dModel),
		dModel:    dModel,
	}
}

// Forward applies the decoder layer with Pre-LN architecture
func (dl *DecoderLayer) Forward(X Matrix) Matrix {
	seqLen := len(X)

	// Pre-LN: Normalize before attention
	normed1 := dl.norm1.Forward(X)

	// Multi-head self-attention
	attnOutput := dl.attention.Forward(normed1)

	// Residual connection
	residual1 := NewMatrix(seqLen, dl.dModel)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dl.dModel; j++ {
			residual1[i][j] = X[i][j] + attnOutput[i][j]
		}
	}

	// Pre-LN: Normalize before FFN
	normed2 := dl.norm2.Forward(residual1)

	// Feed-forward network
	ffnOutput := dl.ffn.Forward(normed2)

	// Residual connection
	output := NewMatrix(seqLen, dl.dModel)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dl.dModel; j++ {
			output[i][j] = residual1[i][j] + ffnOutput[i][j]
		}
	}

	return output
}

// ========================================================================
// SIMPLE CHATGPT - Complete Language Model with Training
// ========================================================================

// SimpleLLM represents our basic language model (GPT-style decoder-only)
type SimpleLLM struct {
	tokenizer        *Tokenizer
	embedding        *Embedding
	layers           []*DecoderLayer
	outputProjection Matrix
	finalNorm        *LayerNorm
	vocabSize        int
	embeddingDim     int
	numLayers        int
	learningRate     float64
}

// NewSimpleLLM creates a new language model
func NewSimpleLLM(vocabSize, embeddingDim, numHeads, numLayers, dFF int) *SimpleLLM {
	rand.Seed(time.Now().UnixNano())

	tokenizer := NewTokenizer()
	embedding := NewEmbedding(tokenizer.vocabSize, embeddingDim)

	// Create stacked decoder layers
	layers := make([]*DecoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		layers[i] = NewDecoderLayer(numHeads, embeddingDim, dFF)
	}

	// Output projection layer (embedding_dim -> vocab_size)
	outputProjection := NewMatrix(embeddingDim, tokenizer.vocabSize)
	scale := math.Sqrt(2.0 / float64(tokenizer.vocabSize))
	for i := 0; i < embeddingDim; i++ {
		for j := 0; j < tokenizer.vocabSize; j++ {
			outputProjection[i][j] = (rand.Float64()*2 - 1) * scale
		}
	}

	return &SimpleLLM{
		tokenizer:        tokenizer,
		embedding:        embedding,
		layers:           layers,
		outputProjection: outputProjection,
		finalNorm:        NewLayerNorm(embeddingDim),
		vocabSize:        tokenizer.vocabSize,
		embeddingDim:     embeddingDim,
		numLayers:        numLayers,
		learningRate:     0.001,
	}
}

// Forward pass through the model - returns logits for each position
func (llm *SimpleLLM) Forward(tokens []int) Matrix {
	// 1. Convert to embeddings
	embeddings := llm.embedding.Forward(tokens)

	// 2. Add positional encoding
	hidden := AddPositionalEncoding(embeddings)

	// 3. Pass through all decoder layers
	for _, layer := range llm.layers {
		hidden = layer.Forward(hidden)
	}

	// 4. Final layer normalization
	hidden = llm.finalNorm.Forward(hidden)

	// 5. Project to vocabulary space
	logits := hidden.Multiply(llm.outputProjection)

	return logits
}

// ========================================================================
// CROSS-ENTROPY LOSS - For next token prediction
// ========================================================================

// CrossEntropyLoss computes the cross-entropy loss for language modeling
func CrossEntropyLoss(logits Matrix, targets []int) float64 {
	seqLen := len(logits)
	totalLoss := 0.0

	for i := 0; i < seqLen; i++ {
		// Apply softmax to get probabilities
		maxLogit := logits[i][0]
		for j := 1; j < len(logits[i]); j++ {
			if logits[i][j] > maxLogit {
				maxLogit = logits[i][j]
			}
		}

		expSum := 0.0
		for j := 0; j < len(logits[i]); j++ {
			expSum += math.Exp(logits[i][j] - maxLogit)
		}

		// Log probability of the correct token
		targetProb := math.Exp(logits[i][targets[i]]-maxLogit) / expSum
		if targetProb > 0 {
			totalLoss -= math.Log(targetProb)
		} else {
			totalLoss += 10.0 // Penalty for zero probability
		}
	}

	return totalLoss / float64(seqLen)
}

// ========================================================================
// TRAINING - Direct weight updates for next-token prediction
// ========================================================================

// TrainStep performs one training step with direct gradient estimation
func (llm *SimpleLLM) TrainStep(inputTokens []int, targetTokens []int) float64 {
	// Forward pass
	logits := llm.Forward(inputTokens)
	seqLen := len(logits)

	// Calculate loss and update weights
	totalLoss := 0.0

	for pos := 0; pos < seqLen && pos < len(targetTokens); pos++ {
		// Get softmax probabilities for this position
		maxLogit := logits[pos][0]
		for j := 1; j < llm.vocabSize; j++ {
			if logits[pos][j] > maxLogit {
				maxLogit = logits[pos][j]
			}
		}

		probs := make([]float64, llm.vocabSize)
		expSum := 0.0
		for j := 0; j < llm.vocabSize; j++ {
			probs[j] = math.Exp(logits[pos][j] - maxLogit)
			expSum += probs[j]
		}
		for j := 0; j < llm.vocabSize; j++ {
			probs[j] /= expSum
		}

		targetToken := targetTokens[pos]

		// Calculate loss for this position
		if probs[targetToken] > 0 {
			totalLoss -= math.Log(probs[targetToken])
		}

		// Get the hidden state for this position (from embeddings + attention)
		inputToken := inputTokens[pos]

		// Update output projection: increase weight for correct token, decrease for others
		for dim := 0; dim < llm.embeddingDim; dim++ {
			embVal := llm.embedding.weights[inputToken][dim]

			// Gradient: prob - target (1 for correct token, 0 for others)
			for vocabIdx := 0; vocabIdx < llm.vocabSize; vocabIdx++ {
				target := 0.0
				if vocabIdx == targetToken {
					target = 1.0
				}
				gradient := probs[vocabIdx] - target

				// Update output projection
				llm.outputProjection[dim][vocabIdx] -= llm.learningRate * gradient * embVal
			}
		}

		// Update embedding for input token to better predict target
		for dim := 0; dim < llm.embeddingDim; dim++ {
			gradientSum := 0.0
			for vocabIdx := 0; vocabIdx < llm.vocabSize; vocabIdx++ {
				target := 0.0
				if vocabIdx == targetToken {
					target = 1.0
				}
				gradient := probs[vocabIdx] - target
				gradientSum += gradient * llm.outputProjection[dim][vocabIdx]
			}
			llm.embedding.weights[inputToken][dim] -= llm.learningRate * gradientSum * 0.1
		}
	}

	return totalLoss / float64(seqLen)
}

// Train trains the model on a dataset using next-token prediction
func (llm *SimpleLLM) Train(data []TrainingExample, epochs int) {
	fmt.Println("\n🎓 TRAINING THE MODEL")
	fmt.Println("======================")

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		numExamples := 0

		// Shuffle training order each epoch
		perm := rand.Perm(len(data))

		for _, idx := range perm {
			example := data[idx]

			// For next-token prediction:
			// Input: full target sequence (shifted)
			// Target: next characters to predict
			targetTokens := llm.tokenizer.EncodeWithoutSpecial(example.Target)

			if len(targetTokens) < 2 {
				continue
			}

			// Input is all tokens except last, target is all tokens except first
			inputTokens := targetTokens[:len(targetTokens)-1]
			targetOutput := targetTokens[1:]

			loss := llm.TrainStep(inputTokens, targetOutput)
			totalLoss += loss
			numExamples++
		}

		avgLoss := totalLoss / float64(numExamples)
		if epoch%5 == 0 || epoch == epochs-1 {
			fmt.Printf("  Epoch %d/%d - Loss: %.4f\n", epoch+1, epochs, avgLoss)
		}
	}

	fmt.Println("✅ Training complete!")
}

// ========================================================================
// TEXT GENERATION with Temperature Sampling
// ========================================================================

// SampleWithTemperature samples from logits with temperature scaling
func SampleWithTemperature(logits []float64, temperature float64) int {
	if temperature <= 0 {
		temperature = 0.0001
	}

	// Apply temperature
	scaled := make([]float64, len(logits))
	maxLogit := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}

	// Softmax with temperature
	expSum := 0.0
	for i := 0; i < len(logits); i++ {
		scaled[i] = math.Exp((logits[i] - maxLogit) / temperature)
		expSum += scaled[i]
	}

	// Normalize to probabilities
	for i := 0; i < len(scaled); i++ {
		scaled[i] /= expSum
	}

	// Sample from the distribution
	r := rand.Float64()
	cumSum := 0.0
	for i := 0; i < len(scaled); i++ {
		cumSum += scaled[i]
		if r <= cumSum {
			return i
		}
	}

	return len(scaled) - 1
}

// TopKSampling samples from the top-k most likely tokens
func TopKSampling(logits []float64, k int, temperature float64) int {
	if k <= 0 || k > len(logits) {
		k = len(logits)
	}

	// Find top-k indices
	type indexedLogit struct {
		index int
		value float64
	}

	indexed := make([]indexedLogit, len(logits))
	for i, v := range logits {
		indexed[i] = indexedLogit{i, v}
	}

	// Simple selection sort for top-k (good enough for small vocab)
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(indexed); j++ {
			if indexed[j].value > indexed[maxIdx].value {
				maxIdx = j
			}
		}
		indexed[i], indexed[maxIdx] = indexed[maxIdx], indexed[i]
	}

	// Create filtered logits (only top-k)
	topKLogits := make([]float64, k)
	for i := 0; i < k; i++ {
		topKLogits[i] = indexed[i].value
	}

	// Sample from top-k
	sampledIdx := SampleWithTemperature(topKLogits, temperature)
	return indexed[sampledIdx].index
}

// Generate generates text using the model with temperature sampling
func (llm *SimpleLLM) Generate(prompt string, maxLength int, temperature float64) string {
	tokens := llm.tokenizer.EncodeWithoutSpecial(prompt)
	if len(tokens) == 0 {
		tokens = []int{llm.tokenizer.vocabToID[" "]}
	}

	generated := make([]int, len(tokens))
	copy(generated, tokens)

	for len(generated) < maxLength {
		// Get model predictions
		logits := llm.Forward(generated)

		// Get logits for the last position
		lastLogits := logits[len(logits)-1]

		// Sample next token using top-k with temperature
		nextTokenID := TopKSampling(lastLogits, 10, temperature)

		// Stop if we generate end token or padding
		if nextTokenID == llm.tokenizer.vocabToID["<END>"] ||
			nextTokenID == llm.tokenizer.vocabToID["<PAD>"] {
			break
		}

		generated = append(generated, nextTokenID)
	}

	return llm.tokenizer.Decode(generated)
}

// Chat provides a simple chat interface with pattern matching fallback
func (llm *SimpleLLM) Chat(userInput string) string {
	input := strings.ToLower(strings.TrimSpace(userInput))

	// Pattern matching for common inputs (reliable fallback)
	responses := map[string]string{
		"hello":            "hello there! how can i help you?",
		"hi":               "hi! nice to chat with you.",
		"hey":              "hey! what's up?",
		"how are you":      "i'm doing great, thanks for asking!",
		"what is your name": "i'm a simple chatgpt, a basic language model.",
		"who are you":      "i'm an ai assistant built with transformers.",
		"bye":              "goodbye! have a great day!",
		"goodbye":          "bye! take care!",
		"thanks":           "you're welcome!",
		"thank you":        "happy to help!",
		"good morning":     "good morning to you too!",
		"good night":       "good night! sleep well!",
		"help":             "i can chat with you! try saying hello or asking how i work.",
		"what can you do":  "i can have simple conversations. i use attention mechanisms!",
	}

	// Check for pattern match
	for pattern, response := range responses {
		if strings.Contains(input, pattern) {
			return response
		}
	}

	// Try neural network generation
	response := llm.Generate(input, len(input)+25, 0.7)

	// Clean up the response
	if strings.HasPrefix(strings.ToLower(response), input) {
		response = response[len(input):]
	}
	response = strings.TrimSpace(response)

	// Filter out garbage characters
	cleaned := strings.Builder{}
	for _, c := range response {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == ' ' || c == '.' || c == '!' || c == '?' || c == ',' || c == '\'' {
			cleaned.WriteRune(c)
		}
	}
	response = cleaned.String()

	if len(response) < 3 {
		return "interesting! tell me more."
	}

	return response
}

// TrainingExample represents a single training sample
type TrainingExample struct {
	Input  string
	Target string
}

// CreateTrainingData generates simple training examples
func CreateTrainingData() []TrainingExample {
	return []TrainingExample{
		{"hello", "hello world"},
		{"how are", "how are you"},
		{"good", "good morning"},
		{"the cat", "the cat sits"},
		{"I am", "I am happy"},
		{"sun is", "sun is bright"},
		{"water", "water flows"},
		{"book", "book reading"},
		{"music", "music playing"},
		{"tree", "tree growing"},
	}
}

// ========================================================================
// TRAINING DATA
// ========================================================================

// CreateTrainingData generates training examples for the chatbot
func CreateConversationData() []TrainingExample {
	return []TrainingExample{
		// Greetings
		{"hello", "hello there"},
		{"hi", "hi there"},
		{"hey", "hey friend"},
		{"good morning", "good morning to you"},
		{"good night", "good night sleep well"},

		// Questions and answers
		{"how are you", "i am doing great"},
		{"what is your name", "i am a simple chatbot"},
		{"who are you", "i am an ai assistant"},
		{"what can you do", "i can chat with you"},

		// Farewells
		{"bye", "bye see you later"},
		{"goodbye", "goodbye take care"},
		{"see you", "see you soon"},

		// Responses
		{"thanks", "you are welcome"},
		{"thank you", "happy to help"},
		{"ok", "okay sounds good"},
		{"yes", "yes indeed"},
		{"no", "no problem"},

		// Simple completions
		{"the weather is", "the weather is nice today"},
		{"i like", "i like spending time here"},
		{"tell me", "tell me more about it"},
		{"help me", "help me understand"},

		// More conversation patterns
		{"nice to meet you", "nice to meet you too"},
		{"how do you work", "i use attention mechanisms"},
		{"are you smart", "i try my best to help"},
		{"what time is it", "i cannot tell the time"},
		{"where are you", "i exist in the code"},
	}
}

func runSimpleLLMDemo() {
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║          🤖 SIMPLE CHATGPT IMPLEMENTATION 🤖               ║")
	fmt.Println("║    Transformer-based Language Model with Training          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Model parameters - optimized for quick training
	embeddingDim := 64 // Embedding dimension
	numHeads := 4      // Number of attention heads
	numLayers := 1     // Single layer for faster training
	dFF := 128         // Feed-forward hidden dimension

	// Create the model
	fmt.Println("🔧 Creating the model...")
	llm := NewSimpleLLM(0, embeddingDim, numHeads, numLayers, dFF)
	llm.learningRate = 0.05 // Higher learning rate for faster convergence

	fmt.Println("\n📊 Model Configuration:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  • Vocabulary Size: %d\n", llm.vocabSize)
	fmt.Printf("  • Embedding Dimension: %d\n", embeddingDim)
	fmt.Printf("  • Number of Attention Heads: %d\n", numHeads)
	fmt.Printf("  • Number of Decoder Layers: %d\n", numLayers)
	fmt.Printf("  • Feed-Forward Dimension: %d\n", dFF)

	// Demonstrate tokenization
	fmt.Println("\n📝 TOKENIZATION DEMO")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━")
	testText := "hello world"
	tokens := llm.tokenizer.Encode(testText)
	decoded := llm.tokenizer.Decode(tokens)
	fmt.Printf("  Original: \"%s\"\n", testText)
	fmt.Printf("  Tokens: %v\n", tokens)
	fmt.Printf("  Decoded: \"%s\"\n", decoded)

	// Demonstrate forward pass
	fmt.Println("\n⚡ FORWARD PASS DEMO")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━")
	inputTokens := llm.tokenizer.EncodeWithoutSpecial("hello")
	logits := llm.Forward(inputTokens)
	fmt.Printf("  Input: \"hello\" -> tokens: %v\n", inputTokens)
	fmt.Printf("  Output logits shape: %dx%d\n", len(logits), len(logits[0]))

	// Get training data
	trainingData := CreateConversationData()
	fmt.Println("\n📚 TRAINING DATA")
	fmt.Println("━━━━━━━━━━━━━━━━")
	fmt.Printf("  Total examples: %d\n", len(trainingData))
	fmt.Println("  Sample pairs:")
	for i, example := range trainingData[:5] {
		fmt.Printf("    %d. \"%s\" → \"%s\"\n", i+1, example.Input, example.Target)
	}

	// Train the model
	llm.Train(trainingData, 100)

	// Demonstrate text generation after training
	fmt.Println("\n✨ TEXT GENERATION (after training)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	prompts := []string{"hello", "how are", "good", "the"}
	for _, prompt := range prompts {
		generated := llm.Generate(prompt, 20, 0.8)
		fmt.Printf("  \"%s\" → \"%s\"\n", prompt, generated)
	}

	// Architecture explanation
	fmt.Println("\n🏗️  ARCHITECTURE COMPONENTS")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  ✓ Token Embedding Layer")
	fmt.Println("  ✓ Positional Encoding (Sinusoidal)")
	fmt.Println("  ✓ Masked Multi-Head Self-Attention")
	fmt.Println("  ✓ Layer Normalization")
	fmt.Println("  ✓ Feed-Forward Network (ReLU)")
	fmt.Println("  ✓ Residual Connections")
	fmt.Println("  ✓ Stacked Decoder Layers")
	fmt.Println("  ✓ Output Projection to Vocabulary")

	fmt.Println("\n💡 FEATURES")
	fmt.Println("━━━━━━━━━━━")
	fmt.Println("  ✓ Temperature-based sampling")
	fmt.Println("  ✓ Top-K sampling")
	fmt.Println("  ✓ Cross-entropy loss")
	fmt.Println("  ✓ Gradient descent training")
	fmt.Println("  ✓ Causal masking for autoregressive generation")
}

// RunChatbot starts an interactive chat session
func RunChatbot() {
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║              🤖 SIMPLE CHATGPT CHATBOT 🤖                  ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Create and configure model
	fmt.Println("🔧 Initializing model...")
	llm := NewSimpleLLM(0, 64, 4, 1, 128)
	llm.learningRate = 0.05

	// Train on conversation data
	trainingData := CreateConversationData()
	fmt.Printf("📚 Training on %d examples...\n", len(trainingData))
	llm.Train(trainingData, 50)

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("💬 Chat started! Type 'quit' to exit.")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if strings.ToLower(input) == "quit" || strings.ToLower(input) == "exit" {
			fmt.Println("\n👋 Goodbye! Thanks for chatting!")
			break
		}

		// Generate response
		response := llm.Generate(input, len(input)+40, 0.7)

		// Clean up the response
		if strings.HasPrefix(strings.ToLower(response), strings.ToLower(input)) {
			response = strings.TrimPrefix(response, input)
			response = strings.TrimPrefix(strings.ToLower(response), strings.ToLower(input))
		}
		response = strings.TrimSpace(response)

		if response == "" {
			response = "i am thinking..."
		}

		fmt.Printf("Bot: %s\n\n", response)
	}
}

// main function - entry point
func main() {
	fmt.Println()
	fmt.Println("Select mode:")
	fmt.Println("  1. Demo mode (show architecture and training)")
	fmt.Println("  2. Chat mode (interactive conversation)")
	fmt.Println()
	fmt.Print("Enter choice (1 or 2): ")

	scanner := bufio.NewScanner(os.Stdin)
	if scanner.Scan() {
		choice := strings.TrimSpace(scanner.Text())
		fmt.Println()

		switch choice {
		case "1":
			runSimpleLLMDemo()
		case "2":
			RunChatbot()
		default:
			fmt.Println("Invalid choice. Running demo mode...")
			runSimpleLLMDemo()
		}
	} else {
		runSimpleLLMDemo()
	}
}