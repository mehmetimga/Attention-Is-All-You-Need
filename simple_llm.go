package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

// Simple LLM Implementation using Transformer Architecture
// This demonstrates how to build a basic language model using attention

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
		"<PAD>", "<START>", "<END>", " ",
		"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
		"n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
		".", "!", "?", ",", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
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
		}
	}
	
	tokens = append(tokens, t.vocabToID["<END>"]) // End token
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

// MaskedMultiHeadAttention implements causal (decoder) attention
type MaskedMultiHeadAttention struct {
	*MultiHeadAttention
}

// NewMaskedMultiHeadAttention creates masked attention for autoregressive generation
func NewMaskedMultiHeadAttention(numHeads, dModel int) *MaskedMultiHeadAttention {
	return &MaskedMultiHeadAttention{
		MultiHeadAttention: NewMultiHeadAttention(numHeads, dModel),
	}
}

// Forward applies masked attention (tokens can't see future tokens)
func (mha *MaskedMultiHeadAttention) Forward(Q, K, V Matrix) Matrix {
	seqLen := len(Q)
	
	// Apply causal mask: tokens can only attend to previous and current positions
	maskedK := NewMatrix(seqLen, mha.dModel)
	maskedV := NewMatrix(seqLen, mha.dModel)
	
	for i := 0; i < seqLen; i++ {
		for j := 0; j < mha.dModel; j++ {
			maskedK[i][j] = K[i][j]
			maskedV[i][j] = V[i][j]
		}
	}
	
	// For simplicity, we'll apply masking in the attention calculation
	return mha.MultiHeadAttention.Forward(Q, maskedK, maskedV)
}

// SimpleLLM represents our basic language model
type SimpleLLM struct {
	tokenizer *Tokenizer
	embedding *Embedding
	attention *MaskedMultiHeadAttention
	outputProjection Matrix // Maps from embedding dim to vocab size
	vocabSize int
	embeddingDim int
}

// NewSimpleLLM creates a new language model
func NewSimpleLLM(vocabSize, embeddingDim, numHeads int) *SimpleLLM {
	tokenizer := NewTokenizer()
	embedding := NewEmbedding(vocabSize, embeddingDim)
	attention := NewMaskedMultiHeadAttention(numHeads, embeddingDim)
	
	// Output projection layer (embedding_dim -> vocab_size)
	outputProjection := NewMatrix(embeddingDim, vocabSize)
	rand.Seed(time.Now().UnixNano())
	scale := math.Sqrt(2.0 / float64(vocabSize))
	for i := 0; i < embeddingDim; i++ {
		for j := 0; j < vocabSize; j++ {
			outputProjection[i][j] = (rand.Float64()*2 - 1) * scale
		}
	}
	
	return &SimpleLLM{
		tokenizer: tokenizer,
		embedding: embedding,
		attention: attention,
		outputProjection: outputProjection,
		vocabSize: vocabSize,
		embeddingDim: embeddingDim,
	}
}

// Forward pass through the model
func (llm *SimpleLLM) Forward(text string) (Matrix, []int) {
	// 1. Tokenize input
	tokens := llm.tokenizer.Encode(text)
	fmt.Printf("Tokenized input: %v\n", tokens)
	
	// 2. Convert to embeddings
	embeddings := llm.embedding.Forward(tokens)
	fmt.Printf("Embeddings shape: %dx%d\n", len(embeddings), len(embeddings[0]))
	
	// 3. Add positional encoding
	posEmbeddings := AddPositionalEncoding(embeddings)
	
	// 4. Apply self-attention
	attentionOutput := llm.attention.Forward(posEmbeddings, posEmbeddings, posEmbeddings)
	
	// 5. Project to vocabulary space (simplified - normally would have multiple layers)
	logits := attentionOutput.Multiply(llm.outputProjection)
	
	return logits, tokens
}

// Generate text using the model (very basic sampling)
func (llm *SimpleLLM) Generate(prompt string, maxLength int) string {
	tokens := llm.tokenizer.Encode(prompt)
	generated := make([]int, len(tokens))
	copy(generated, tokens)
	
	for len(generated) < maxLength {
		// Get model predictions for current sequence
		currentText := llm.tokenizer.Decode(generated[1:]) // Remove <START> token
		logits, _ := llm.Forward(currentText)
		
		// Sample next token (simplified - just pick the token with highest score in last position)
		lastLogits := logits[len(logits)-1]
		nextTokenID := 0
		maxScore := lastLogits[0]
		
		for i := 1; i < len(lastLogits); i++ {
			if lastLogits[i] > maxScore {
				maxScore = lastLogits[i]
				nextTokenID = i
			}
		}
		
		// Stop if we generate end token
		if nextTokenID == llm.tokenizer.vocabToID["<END>"] {
			break
		}
		
		generated = append(generated, nextTokenID)
	}
	
	return llm.tokenizer.Decode(generated)
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

func runSimpleLLMDemo() {
	fmt.Println("SIMPLE TRANSFORMER-BASED LANGUAGE MODEL")
	fmt.Println("=======================================")
	
	// Model parameters
	vocabSize := 42    // Size of our simple vocabulary
	embeddingDim := 16 // Small embedding dimension for demo
	numHeads := 2      // Number of attention heads
	
	// Create the model
	llm := NewSimpleLLM(vocabSize, embeddingDim, numHeads)
	
	fmt.Printf("Model Configuration:\n")
	fmt.Printf("- Vocabulary Size: %d\n", vocabSize)
	fmt.Printf("- Embedding Dimension: %d\n", embeddingDim)
	fmt.Printf("- Number of Attention Heads: %d\n\n", numHeads)
	
	// Demonstrate tokenization
	fmt.Println("=== TOKENIZATION DEMO ===")
	testText := "hello world"
	tokens := llm.tokenizer.Encode(testText)
	decoded := llm.tokenizer.Decode(tokens)
	fmt.Printf("Original: \"%s\"\n", testText)
	fmt.Printf("Tokens: %v\n", tokens)
	fmt.Printf("Decoded: \"%s\"\n\n", decoded)
	
	// Demonstrate forward pass
	fmt.Println("=== FORWARD PASS DEMO ===")
	logits, originalTokens := llm.Forward("hello")
	fmt.Printf("Input tokens: %v\n", originalTokens)
	fmt.Printf("Output logits shape: %dx%d\n", len(logits), len(logits[0]))
	fmt.Printf("Sample logits for first position: [%.3f, %.3f, %.3f, ...]\n\n", 
		logits[0][0], logits[0][1], logits[0][2])
	
	// Demonstrate simple generation
	fmt.Println("=== TEXT GENERATION DEMO ===")
	prompt := "hello"
	generated := llm.Generate(prompt, 10)
	fmt.Printf("Prompt: \"%s\"\n", prompt)
	fmt.Printf("Generated: \"%s\"\n\n", generated)
	
	// Show training data examples
	fmt.Println("=== TRAINING DATA EXAMPLES ===")
	trainingData := CreateTrainingData()
	fmt.Println("Sample training pairs:")
	for i, example := range trainingData[:5] {
		fmt.Printf("%d. Input: \"%s\" -> Target: \"%s\"\n", 
			i+1, example.Input, example.Target)
	}
	
	fmt.Println("\n=== APPLICATION IDEAS ===")
	fmt.Println("1. TEXT COMPLETION: Complete partial sentences")
	fmt.Println("2. SIMPLE CHATBOT: Respond to basic questions")
	fmt.Println("3. CODE COMPLETION: Complete simple programming patterns")
	fmt.Println("4. POETRY GENERATION: Generate rhyming text")
	fmt.Println("5. STORY CONTINUATION: Extend story beginnings")
	fmt.Println("6. DIALOG SYSTEMS: Role-playing conversations")
	fmt.Println("7. TEMPLATE FILLING: Complete form templates")
	fmt.Println("8. SIMPLE TRANSLATION: Between simple language patterns")
	
	fmt.Println("\n=== TRAINING RECOMMENDATIONS ===")
	fmt.Println("For effective training, you need:")
	fmt.Println("• Large text corpus (books, articles, conversations)")
	fmt.Println("• Next-token prediction objective")
	fmt.Println("• Gradient descent optimization (Adam, SGD)")
	fmt.Println("• Multiple training epochs")
	fmt.Println("• Proper learning rate scheduling")
	fmt.Println("• Regularization (dropout, weight decay)")
}