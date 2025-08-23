package main

import (
	"fmt"
	"math"
)

// Matrix represents a 2D matrix for our attention calculations
type Matrix [][]float64

// NewMatrix creates a new matrix with given dimensions
func NewMatrix(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

// Transpose returns the transpose of the matrix
func (m Matrix) Transpose() Matrix {
	rows, cols := len(m), len(m[0])
	result := NewMatrix(cols, rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[j][i] = m[i][j]
		}
	}
	return result
}

// Multiply performs matrix multiplication
func (m Matrix) Multiply(other Matrix) Matrix {
	rows1, cols1 := len(m), len(m[0])
	rows2, cols2 := len(other), len(other[0])
	
	if cols1 != rows2 {
		panic("Matrix dimensions don't match for multiplication")
	}
	
	result := NewMatrix(rows1, cols2)
	for i := 0; i < rows1; i++ {
		for j := 0; j < cols2; j++ {
			for k := 0; k < cols1; k++ {
				result[i][j] += m[i][k] * other[k][j]
			}
		}
	}
	return result
}

// Scale multiplies each element by a scalar
func (m Matrix) Scale(scalar float64) Matrix {
	rows, cols := len(m), len(m[0])
	result := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = m[i][j] * scalar
		}
	}
	return result
}

// Softmax applies softmax function to each row
func (m Matrix) Softmax() Matrix {
	rows, cols := len(m), len(m[0])
	result := NewMatrix(rows, cols)
	
	for i := 0; i < rows; i++ {
		// Find max for numerical stability
		max := m[i][0]
		for j := 1; j < cols; j++ {
			if m[i][j] > max {
				max = m[i][j]
			}
		}
		
		// Calculate exp and sum
		sum := 0.0
		for j := 0; j < cols; j++ {
			result[i][j] = math.Exp(m[i][j] - max)
			sum += result[i][j]
		}
		
		// Normalize
		for j := 0; j < cols; j++ {
			result[i][j] /= sum
		}
	}
	return result
}

// Print displays the matrix
func (m Matrix) Print(name string) {
	fmt.Printf("%s:\n", name)
	for _, row := range m {
		fmt.Printf("  [")
		for j, val := range row {
			if j > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%.4f", val)
		}
		fmt.Printf("]\n")
	}
	fmt.Println()
}

// ScaledDotProductAttention implements the core attention mechanism from the paper
// Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
func ScaledDotProductAttention(Q, K, V Matrix) Matrix {
	fmt.Println("=== Scaled Dot-Product Attention ===")
	
	// Step 1: Calculate QK^T
	KT := K.Transpose()
	fmt.Println("Step 1: Calculate Q * K^T")
	Q.Print("Query (Q)")
	KT.Print("Key Transpose (K^T)")
	
	QKT := Q.Multiply(KT)
	QKT.Print("Q * K^T")
	
	// Step 2: Scale by sqrt(d_k) where d_k is the dimension of keys
	dk := float64(len(K[0])) // dimension of keys
	scalingFactor := 1.0 / math.Sqrt(dk)
	fmt.Printf("Step 2: Scale by 1/sqrt(d_k) = 1/sqrt(%.0f) = %.4f\n", dk, scalingFactor)
	
	scaled := QKT.Scale(scalingFactor)
	scaled.Print("Scaled scores")
	
	// Step 3: Apply softmax
	fmt.Println("Step 3: Apply softmax to get attention weights")
	attentionWeights := scaled.Softmax()
	attentionWeights.Print("Attention weights")
	
	// Step 4: Multiply by Values
	fmt.Println("Step 4: Multiply attention weights by Values")
	V.Print("Values (V)")
	
	result := attentionWeights.Multiply(V)
	result.Print("Final attention output")
	
	return result
}

// MultiHeadAttention implements simplified multi-head attention
type MultiHeadAttention struct {
	numHeads int
	dModel   int
	dK       int
}

func NewMultiHeadAttention(numHeads, dModel int) *MultiHeadAttention {
	return &MultiHeadAttention{
		numHeads: numHeads,
		dModel:   dModel,
		dK:       dModel / numHeads, // each head gets dModel/numHeads dimensions
	}
}

func (mha *MultiHeadAttention) Forward(Q, K, V Matrix) Matrix {
	fmt.Println("=== Multi-Head Attention ===")
	fmt.Printf("Number of heads: %d, d_model: %d, d_k per head: %d\n\n", 
		mha.numHeads, mha.dModel, mha.dK)
	
	seqLen := len(Q)
	headOutputs := make([]Matrix, mha.numHeads)
	
	// Process each head
	for head := 0; head < mha.numHeads; head++ {
		fmt.Printf("--- Head %d ---\n", head+1)
		
		// For simplicity, we'll use different slices of the input matrices
		// In a real implementation, you'd have learned projection matrices
		startCol := head * mha.dK
		
		// Project Q, K, V for this head
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
		
		// Apply attention for this head
		headOutput := ScaledDotProductAttention(headQ, headK, headV)
		headOutputs[head] = headOutput
		
		fmt.Printf("Head %d output:\n", head+1)
		headOutput.Print("")
	}
	
	// Concatenate all head outputs
	fmt.Println("Concatenating all head outputs...")
	result := NewMatrix(seqLen, mha.dModel)
	for i := 0; i < seqLen; i++ {
		for head := 0; head < mha.numHeads; head++ {
			startCol := head * mha.dK
			for j := 0; j < mha.dK; j++ {
				result[i][startCol+j] = headOutputs[head][i][j]
			}
		}
	}
	
	result.Print("Multi-head attention output")
	return result
}

func runOriginalDemo() {
	fmt.Println("Transformer Attention Mechanism Implementation")
	fmt.Println("=============================================\n")
	
	// Create sample 8-length sequences with 8-dimensional embeddings
	// This represents 8 tokens, each with 8-dimensional embeddings
	seqLen := 8
	dModel := 8
	
	// Initialize sample matrices
	// In practice, these would come from embeddings + positional encodings
	Q := Matrix{
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7},
		{0.3, 0.4, 0.1, 0.2, 0.7, 0.8, 0.5, 0.6},
		{0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6, 0.5},
		{0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4},
		{0.6, 0.5, 0.8, 0.7, 0.2, 0.1, 0.4, 0.3},
		{0.7, 0.8, 0.5, 0.6, 0.3, 0.4, 0.1, 0.2},
		{0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1},
	}
	
	K := Matrix{
		{0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1},
		{0.7, 0.8, 0.5, 0.6, 0.3, 0.4, 0.1, 0.2},
		{0.6, 0.5, 0.8, 0.7, 0.2, 0.1, 0.4, 0.3},
		{0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4},
		{0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6, 0.5},
		{0.3, 0.4, 0.1, 0.2, 0.7, 0.8, 0.5, 0.6},
		{0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7},
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
	}
	
	V := Matrix{
		{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3},
		{0.9, 1.0, 0.7, 0.8, 0.5, 0.6, 0.3, 0.4},
		{0.8, 0.7, 1.0, 0.9, 0.4, 0.3, 0.6, 0.5},
		{0.7, 0.8, 0.9, 1.0, 0.3, 0.4, 0.5, 0.6},
		{0.6, 0.5, 0.4, 0.3, 1.0, 0.9, 0.8, 0.7},
		{0.5, 0.6, 0.3, 0.4, 0.9, 1.0, 0.7, 0.8},
		{0.4, 0.3, 0.6, 0.5, 0.8, 0.7, 1.0, 0.9},
		{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
	}
	
	fmt.Printf("Working with %dx%d matrices (sequence length: %d, embedding dim: %d)\n\n", 
		seqLen, dModel, seqLen, dModel)
	
	// Demonstrate single-head attention
	fmt.Println("PART 1: Single-Head Scaled Dot-Product Attention")
	fmt.Println("================================================")
	result1 := ScaledDotProductAttention(Q, K, V)
	
	fmt.Println("\nPART 2: Multi-Head Attention")
	fmt.Println("============================")
	
	// Demonstrate multi-head attention with 2 heads
	mha := NewMultiHeadAttention(2, dModel)
	result2 := mha.Forward(Q, K, V)
	
	fmt.Println("\nSUMMARY:")
	fmt.Println("========")
	fmt.Println("This implementation demonstrates the core concepts from 'Attention Is All You Need':")
	fmt.Println("1. Scaled dot-product attention: softmax(QK^T / sqrt(d_k))V")
	fmt.Println("2. Multi-head attention: parallel attention heads that capture different relationships")
	fmt.Println("3. Each position can attend to all positions in the sequence simultaneously")
	fmt.Println("4. No recurrence needed - fully parallelizable computation")
	
	_ = result1
	_ = result2
}