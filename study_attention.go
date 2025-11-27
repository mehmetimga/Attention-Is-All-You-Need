// ============================================================================
// ATTENTION MECHANISM STUDY GUIDE
// ============================================================================
// This file contains implementations and examples for studying the core
// concepts from "Attention Is All You Need" paper.
//
// To run: go run study_attention.go transformer.go
// ============================================================================

package main

import (
	"fmt"
	"math"
	"strings"
)

// ============================================================================
// PART 1: MATRIX OPERATIONS
// ============================================================================
// These are the fundamental operations used in attention mechanisms.

// DemoMatrixOperations shows basic matrix operations used in transformers
func DemoMatrixOperations() {
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║         PART 1: MATRIX OPERATIONS                          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Create sample matrices
	A := Matrix{
		{1, 2},
		{3, 4},
	}

	B := Matrix{
		{5, 6},
		{7, 8},
	}

	fmt.Println("Matrix A:")
	A.Print("")

	fmt.Println("Matrix B:")
	B.Print("")

	// Multiplication
	fmt.Println("1. Matrix Multiplication (A × B):")
	fmt.Println("   Formula: C[i][j] = Σ A[i][k] × B[k][j]")
	C := A.Multiply(B)
	C.Print("   Result")

	// Transpose
	fmt.Println("2. Matrix Transpose (A^T):")
	fmt.Println("   Formula: A^T[i][j] = A[j][i]")
	AT := A.Transpose()
	AT.Print("   Result")

	// Scaling
	fmt.Println("3. Scalar Multiplication (A × 0.5):")
	scaled := A.Scale(0.5)
	scaled.Print("   Result")

	// Softmax
	fmt.Println("4. Softmax (row-wise):")
	fmt.Println("   Formula: softmax(x_i) = e^(x_i) / Σ e^(x_j)")
	fmt.Println("   Property: All values sum to 1.0")
	logits := Matrix{{1.0, 2.0, 3.0}}
	probs := logits.Softmax()
	probs.Print("   Result")
	fmt.Println()
}

// ============================================================================
// PART 2: SCALED DOT-PRODUCT ATTENTION
// ============================================================================
// The core attention mechanism: Attention(Q,K,V) = softmax(QK^T / √d_k)V

// DemoScaledDotProductAttention explains the attention formula step by step
func DemoScaledDotProductAttention() {
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║     PART 2: SCALED DOT-PRODUCT ATTENTION                   ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Println("Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) × V")
	fmt.Println()

	// Small example for clarity
	// 3 tokens, 4 dimensions
	Q := Matrix{
		{1.0, 0.0, 1.0, 0.0}, // Query for token 1
		{0.0, 1.0, 0.0, 1.0}, // Query for token 2
		{1.0, 1.0, 0.0, 0.0}, // Query for token 3
	}

	K := Matrix{
		{1.0, 0.0, 1.0, 0.0}, // Key for token 1
		{0.0, 1.0, 0.0, 1.0}, // Key for token 2
		{0.5, 0.5, 0.5, 0.5}, // Key for token 3
	}

	V := Matrix{
		{1.0, 0.0, 0.0, 0.0}, // Value for token 1 (represents "apple")
		{0.0, 1.0, 0.0, 0.0}, // Value for token 2 (represents "banana")
		{0.0, 0.0, 1.0, 0.0}, // Value for token 3 (represents "cherry")
	}

	fmt.Println("Input Matrices (3 tokens, 4 dimensions):")
	Q.Print("Query (Q) - What each token is looking for")
	K.Print("Key (K) - What each token offers to be found")
	V.Print("Value (V) - The actual information to retrieve")

	// Step 1: QK^T
	fmt.Println("STEP 1: Calculate QK^T (similarity scores)")
	fmt.Println("─────────────────────────────────────────")
	KT := K.Transpose()
	QKT := Q.Multiply(KT)
	QKT.Print("QK^T (raw attention scores)")
	fmt.Println("  Interpretation: QKT[i][j] = how much token i attends to token j")
	fmt.Println()

	// Step 2: Scale by sqrt(d_k)
	fmt.Println("STEP 2: Scale by 1/√d_k")
	fmt.Println("────────────────────────")
	dk := float64(len(K[0])) // d_k = 4
	scalingFactor := 1.0 / math.Sqrt(dk)
	fmt.Printf("  d_k = %v, scaling factor = 1/√%v = %.4f\n", int(dk), int(dk), scalingFactor)
	fmt.Println("  Why scale? To prevent softmax from becoming too peaked")
	fmt.Println("  (large dot products → very small gradients after softmax)")
	scaled := QKT.Scale(scalingFactor)
	scaled.Print("Scaled scores")
	fmt.Println()

	// Step 3: Softmax
	fmt.Println("STEP 3: Apply Softmax (row-wise)")
	fmt.Println("─────────────────────────────────")
	fmt.Println("  Convert scores to probabilities (each row sums to 1.0)")
	attentionWeights := scaled.Softmax()
	attentionWeights.Print("Attention weights")

	// Verify row sums
	fmt.Println("  Verification (row sums):")
	for i := 0; i < len(attentionWeights); i++ {
		sum := 0.0
		for j := 0; j < len(attentionWeights[i]); j++ {
			sum += attentionWeights[i][j]
		}
		fmt.Printf("    Row %d sum: %.4f\n", i, sum)
	}
	fmt.Println()

	// Step 4: Multiply by V
	fmt.Println("STEP 4: Multiply by Values")
	fmt.Println("──────────────────────────")
	fmt.Println("  Output = weighted sum of value vectors")
	output := attentionWeights.Multiply(V)
	output.Print("Final output")

	fmt.Println("  Interpretation:")
	fmt.Println("    - Token 1 mostly attends to itself → output ≈ V[0]")
	fmt.Println("    - Token 2 mostly attends to itself → output ≈ V[1]")
	fmt.Println("    - Token 3 attends to a mix → output = weighted combo")
	fmt.Println()
}

// ============================================================================
// PART 3: MULTI-HEAD ATTENTION
// ============================================================================
// Multiple attention heads capture different types of relationships

// DemoMultiHeadAttention explains how multiple heads work
func DemoMultiHeadAttention() {
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║         PART 3: MULTI-HEAD ATTENTION                       ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Println("Why multiple heads?")
	fmt.Println("───────────────────")
	fmt.Println("  • Each head can focus on different aspects")
	fmt.Println("  • Head 1: might learn syntactic relationships")
	fmt.Println("  • Head 2: might learn semantic relationships")
	fmt.Println("  • Head 3: might learn positional patterns")
	fmt.Println()

	fmt.Println("How it works:")
	fmt.Println("─────────────")
	fmt.Println("  1. Project Q, K, V into h different subspaces")
	fmt.Println("  2. Run attention in parallel on each subspace")
	fmt.Println("  3. Concatenate all head outputs")
	fmt.Println("  4. Project back to original dimension")
	fmt.Println()

	// Example with 2 heads
	numHeads := 2
	dModel := 8 // total embedding dimension
	dK := dModel / numHeads

	fmt.Printf("Example: %d heads, d_model=%d, d_k per head=%d\n\n", numHeads, dModel, dK)

	// Create sample input
	X := Matrix{
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7},
		{0.3, 0.4, 0.1, 0.2, 0.7, 0.8, 0.5, 0.6},
	}

	X.Print("Input X (3 tokens, 8 dimensions)")

	fmt.Println("Head 1 processes dimensions [0:4]")
	fmt.Println("Head 2 processes dimensions [4:8]")
	fmt.Println()

	// Show head 1 input
	head1X := NewMatrix(3, 4)
	head2X := NewMatrix(3, 4)
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			head1X[i][j] = X[i][j]
			head2X[i][j] = X[i][j+4]
		}
	}

	head1X.Print("Head 1 input (dims 0-3)")
	head2X.Print("Head 2 input (dims 4-7)")

	fmt.Println("Each head runs scaled dot-product attention independently,")
	fmt.Println("then outputs are concatenated back to d_model dimensions.")
	fmt.Println()
}

// ============================================================================
// PART 4: CAUSAL MASKING (for Language Models)
// ============================================================================
// Prevents tokens from attending to future positions

// DemoCausalMasking shows how masking works for autoregressive generation
func DemoCausalMasking() {
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║         PART 4: CAUSAL MASKING                             ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Println("Why causal masking?")
	fmt.Println("───────────────────")
	fmt.Println("  • For language models (GPT), we generate text left-to-right")
	fmt.Println("  • Token at position i should NOT see tokens at positions > i")
	fmt.Println("  • This prevents 'cheating' during training")
	fmt.Println()

	// Example attention scores
	scores := Matrix{
		{1.0, 0.5, 0.3, 0.1},
		{0.5, 1.0, 0.5, 0.3},
		{0.3, 0.5, 1.0, 0.5},
		{0.1, 0.3, 0.5, 1.0},
	}

	fmt.Println("Before masking (raw attention scores):")
	scores.Print("")

	fmt.Println("Applying causal mask (set future positions to -∞):")
	fmt.Println("  • Position 0 can only see position 0")
	fmt.Println("  • Position 1 can see positions 0, 1")
	fmt.Println("  • Position 2 can see positions 0, 1, 2")
	fmt.Println("  • etc.")
	fmt.Println()

	// Apply mask
	masked := NewMatrix(4, 4)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if j > i {
				masked[i][j] = math.Inf(-1) // Can't attend to future
			} else {
				masked[i][j] = scores[i][j]
			}
		}
	}

	// Show masked scores (replace -inf with -∞ for display)
	fmt.Println("Masked scores (−∞ for future positions):")
	for i := 0; i < 4; i++ {
		fmt.Print("  [")
		for j := 0; j < 4; j++ {
			if math.IsInf(masked[i][j], -1) {
				fmt.Print("  -∞  ")
			} else {
				fmt.Printf(" %.2f ", masked[i][j])
			}
		}
		fmt.Println("]")
	}
	fmt.Println()

	fmt.Println("After softmax (−∞ becomes 0):")
	softmaxed := masked.Softmax()
	softmaxed.Print("")

	fmt.Println("  Notice: Each row only has non-zero weights for current")
	fmt.Println("  and previous positions!")
	fmt.Println()
}

// ============================================================================
// PART 5: POSITIONAL ENCODING
// ============================================================================
// Injects position information since attention has no inherent ordering

// DemoPositionalEncoding shows how position information is added
func DemoPositionalEncoding() {
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║         PART 5: POSITIONAL ENCODING                        ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Println("Why positional encoding?")
	fmt.Println("────────────────────────")
	fmt.Println("  • Attention treats input as a SET (no order)")
	fmt.Println("  • 'The cat sat' and 'sat cat The' would be identical!")
	fmt.Println("  • We must explicitly add position information")
	fmt.Println()

	fmt.Println("Sinusoidal Positional Encoding:")
	fmt.Println("────────────────────────────────")
	fmt.Println("  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))")
	fmt.Println("  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
	fmt.Println()

	fmt.Println("Why sinusoids?")
	fmt.Println("  • Can represent any position (even longer than training)")
	fmt.Println("  • PE(pos+k) can be represented as linear function of PE(pos)")
	fmt.Println("  • Allows model to learn relative positions")
	fmt.Println()

	// Show encoding for first few positions
	dModel := 8
	seqLen := 4

	fmt.Printf("Example: Positional encodings for %d positions, %d dimensions:\n\n", seqLen, dModel)

	pe := NewMatrix(seqLen, dModel)
	for pos := 0; pos < seqLen; pos++ {
		for i := 0; i < dModel; i++ {
			if i%2 == 0 {
				pe[pos][i] = math.Sin(float64(pos) / math.Pow(10000, float64(i)/float64(dModel)))
			} else {
				pe[pos][i] = math.Cos(float64(pos) / math.Pow(10000, float64(i-1)/float64(dModel)))
			}
		}
	}

	pe.Print("Positional Encodings")

	fmt.Println("  Each position has a unique 'fingerprint'")
	fmt.Println("  These are ADDED to the token embeddings")
	fmt.Println()
}

// ============================================================================
// PART 6: ATTENTION PATTERNS VISUALIZATION
// ============================================================================

// DemoAttentionPatterns shows different attention behaviors
func DemoAttentionPatterns() {
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║         PART 6: ATTENTION PATTERNS                         ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Println("Different Q/K patterns create different attention behaviors:")
	fmt.Println()

	// Pattern 1: Self-attention (identity)
	fmt.Println("1. SELF-ATTENTION (Identity Q=K)")
	fmt.Println("   " + strings.Repeat("─", 40))
	identity := Matrix{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}
	identityScores := identity.Multiply(identity.Transpose())
	identityWeights := identityScores.Softmax()
	fmt.Println("   Each token attends mainly to itself:")
	visualizeAttention(identityWeights)

	// Pattern 2: Uniform attention
	fmt.Println("2. UNIFORM ATTENTION (all ones)")
	fmt.Println("   " + strings.Repeat("─", 40))
	uniform := Matrix{
		{1, 1, 1, 1},
		{1, 1, 1, 1},
		{1, 1, 1, 1},
		{1, 1, 1, 1},
	}
	uniformScores := uniform.Multiply(uniform.Transpose())
	uniformWeights := uniformScores.Softmax()
	fmt.Println("   Each token attends equally to all:")
	visualizeAttention(uniformWeights)

	// Pattern 3: Local attention
	fmt.Println("3. LOCAL ATTENTION (diagonal pattern)")
	fmt.Println("   " + strings.Repeat("─", 40))
	local := Matrix{
		{1.0, 0.5, 0.0, 0.0},
		{0.5, 1.0, 0.5, 0.0},
		{0.0, 0.5, 1.0, 0.5},
		{0.0, 0.0, 0.5, 1.0},
	}
	localScores := local.Multiply(local.Transpose())
	localWeights := localScores.Softmax()
	fmt.Println("   Each token attends to neighbors:")
	visualizeAttention(localWeights)
	fmt.Println()
}

// visualizeAttention prints a visual representation of attention weights
func visualizeAttention(weights Matrix) {
	symbols := []string{"░", "▒", "▓", "█"}
	for i := 0; i < len(weights); i++ {
		fmt.Print("   ")
		for j := 0; j < len(weights[i]); j++ {
			w := weights[i][j]
			var sym string
			if w < 0.15 {
				sym = symbols[0]
			} else if w < 0.25 {
				sym = symbols[1]
			} else if w < 0.35 {
				sym = symbols[2]
			} else {
				sym = symbols[3]
			}
			fmt.Print(sym + sym)
		}
		fmt.Println()
	}
	fmt.Println()
}

// ============================================================================
// MAIN - Run all demonstrations
// ============================================================================

func main() {
	fmt.Println()
	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║     TRANSFORMER ATTENTION MECHANISM - STUDY GUIDE          ║")
	fmt.Println("║     Based on 'Attention Is All You Need' (2017)            ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("This guide walks through the core concepts of the attention")
	fmt.Println("mechanism used in Transformers (GPT, BERT, etc.)")
	fmt.Println()
	fmt.Println(strings.Repeat("═", 60))
	fmt.Println()

	DemoMatrixOperations()
	DemoScaledDotProductAttention()
	DemoMultiHeadAttention()
	DemoCausalMasking()
	DemoPositionalEncoding()
	DemoAttentionPatterns()

	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      SUMMARY                               ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Key Concepts:")
	fmt.Println("  1. Attention = softmax(QK^T / √d_k) × V")
	fmt.Println("  2. Q (Query): What am I looking for?")
	fmt.Println("  3. K (Key): What do I have to offer?")
	fmt.Println("  4. V (Value): What information do I contain?")
	fmt.Println("  5. Multi-head: Multiple attention patterns in parallel")
	fmt.Println("  6. Causal mask: Prevent seeing future tokens")
	fmt.Println("  7. Positional encoding: Add position information")
	fmt.Println()
	fmt.Println("To see the complete working example:")
	fmt.Println("  go run simple_llm.go transformer.go")
	fmt.Println()
}

