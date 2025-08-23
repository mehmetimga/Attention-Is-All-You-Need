// Test runner for transformer attention mechanisms
package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"strings"
	"time"
)

// TestInput represents the structure of our test input JSON
type TestInput struct {
	Description string      `json:"description"`
	Q           [][]float64 `json:"Q"`
	K           [][]float64 `json:"K"`
	V           [][]float64 `json:"V"`
}

// TestInputs represents all test scenarios
type TestInputs struct {
	IdentityTest   TestInput `json:"identity_test"`
	UniformTest    TestInput `json:"uniform_test"`
	SequentialTest TestInput `json:"sequential_test"`
}

// LoadTestInputs loads test data from JSON file
func LoadTestInputs(filename string) (*TestInputs, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var testInputs TestInputs
	err = json.Unmarshal(data, &testInputs)
	if err != nil {
		return nil, err
	}

	return &testInputs, nil
}

// GenerateRandomMatrix creates a matrix with random values between min and max
func GenerateRandomMatrix(rows, cols int, min, max float64) Matrix {
	rand.Seed(time.Now().UnixNano())
	matrix := NewMatrix(rows, cols)
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix[i][j] = min + rand.Float64()*(max-min)
		}
	}
	return matrix
}

// AnalyzeAttentionWeights provides insights into attention patterns
func AnalyzeAttentionWeights(weights Matrix, name string) {
	fmt.Printf("\n--- Analysis of %s ---\n", name)
	rows := len(weights)
	
	// Find max attention for each position
	for i := 0; i < rows; i++ {
		maxVal := weights[i][0]
		maxPos := 0
		
		for j := 1; j < len(weights[i]); j++ {
			if weights[i][j] > maxVal {
				maxVal = weights[i][j]
				maxPos = j
			}
		}
		
		fmt.Printf("Token %d attends most to Token %d (weight: %.4f)\n", i, maxPos, maxVal)
	}
	
	// Calculate attention distribution variance
	totalVariance := 0.0
	for i := 0; i < rows; i++ {
		mean := 1.0 / float64(len(weights[i])) // uniform would be 1/n
		variance := 0.0
		
		for j := 0; j < len(weights[i]); j++ {
			diff := weights[i][j] - mean
			variance += diff * diff
		}
		variance /= float64(len(weights[i]))
		totalVariance += variance
	}
	
	avgVariance := totalVariance / float64(rows)
	fmt.Printf("Average attention variance: %.6f\n", avgVariance)
	if avgVariance < 0.001 {
		fmt.Println("→ Nearly uniform attention (all tokens equally important)")
	} else if avgVariance > 0.01 {
		fmt.Println("→ Highly focused attention (some tokens much more important)")
	} else {
		fmt.Println("→ Moderately focused attention")
	}
}

// TestIdentityMatrices tests with identity matrices
func TestIdentityMatrices() {
	fmt.Printf("\n" + strings.Repeat("=", 80) + "\n")
	fmt.Printf("IDENTITY MATRIX TEST\n")
	fmt.Printf(strings.Repeat("=", 80) + "\n")
	fmt.Println("Expected: Each token should attend most strongly to itself")
	fmt.Println("This simulates perfect self-attention where each position")
	fmt.Println("only cares about its own content.\n")

	testInputs, err := LoadTestInputs("test_inputs.json")
	if err != nil {
		fmt.Printf("Error loading test inputs: %v\n", err)
		return
	}

	test := testInputs.IdentityTest
	Q := Matrix(test.Q)
	K := Matrix(test.K)
	V := Matrix(test.V)

	// Calculate QK^T to show the pattern
	KT := K.Transpose()
	QKT := Q.Multiply(KT)
	fmt.Println("Q × K^T (before scaling and softmax):")
	QKT.Print("Raw attention scores")
	
	// Apply scaled dot-product attention
	fmt.Println("\nRunning Scaled Dot-Product Attention...")
	_ = ScaledDotProductAttention(Q, K, V)
	
	fmt.Println("EXPLANATION:")
	fmt.Println("- Identity Q and K matrices create perfect diagonal attention")
	fmt.Println("- Each position (i,i) gets score 1.0, all others get 0.0")
	fmt.Println("- After softmax, each token attends with ~100% weight to itself")
	fmt.Println("- Output values are essentially the same as input values")
}

// TestUniformAttention tests uniform attention scenario  
func TestUniformAttention() {
	fmt.Printf("\n" + strings.Repeat("=", 80) + "\n")
	fmt.Printf("UNIFORM ATTENTION TEST\n")
	fmt.Printf(strings.Repeat("=", 80) + "\n")
	fmt.Println("Expected: All tokens receive equal attention weights")
	fmt.Println("This simulates global average pooling where each position")
	fmt.Println("considers all positions equally.\n")

	testInputs, err := LoadTestInputs("test_inputs.json")
	if err != nil {
		fmt.Printf("Error loading test inputs: %v\n", err)
		return
	}

	test := testInputs.UniformTest
	Q := Matrix(test.Q)
	K := Matrix(test.K)
	V := Matrix(test.V)

	// Calculate QK^T to show the pattern
	KT := K.Transpose()
	QKT := Q.Multiply(KT)
	fmt.Println("Q × K^T (before scaling and softmax):")
	QKT.Print("Raw attention scores")
	
	fmt.Println("\nRunning Scaled Dot-Product Attention...")
	_ = ScaledDotProductAttention(Q, K, V)
	
	fmt.Println("EXPLANATION:")
	fmt.Println("- All Q and K values are 1.0, so Q×K^T creates uniform 8.0 scores")
	fmt.Println("- After softmax, all positions get equal weight (1/8 = 0.125)")
	fmt.Println("- Output is average of all value vectors")
	fmt.Println("- Each output position contains [0.125, 0.125, ..., 0.125]")
}

// TestSequentialAttention tests neighbor-focused attention
func TestSequentialAttention() {
	fmt.Printf("\n" + strings.Repeat("=", 80) + "\n")
	fmt.Printf("SEQUENTIAL ATTENTION TEST\n")
	fmt.Printf(strings.Repeat("=", 80) + "\n")
	fmt.Println("Expected: Each token attends more to itself and neighbors")
	fmt.Println("This simulates local attention patterns similar to convolution")
	fmt.Println("but with learned attention weights.\n")

	testInputs, err := LoadTestInputs("test_inputs.json")
	if err != nil {
		fmt.Printf("Error loading test inputs: %v\n", err)
		return
	}

	test := testInputs.SequentialTest
	Q := Matrix(test.Q)
	K := Matrix(test.K)
	V := Matrix(test.V)

	// Show input matrices briefly
	fmt.Println("Key matrix pattern (tridiagonal):")
	K.Print("Key Matrix (K)")

	// Calculate QK^T to show the pattern
	KT := K.Transpose()
	QKT := Q.Multiply(KT)
	fmt.Println("\nQ × K^T (shows local attention pattern):")
	QKT.Print("Raw attention scores")
	
	fmt.Println("\nRunning Scaled Dot-Product Attention...")
	_ = ScaledDotProductAttention(Q, K, V)
	
	// Analyze the attention pattern
	scaled := QKT.Scale(1.0 / 2.828) // sqrt(8)
	attentionWeights := scaled.Softmax()
	AnalyzeAttentionWeights(attentionWeights, "Sequential Attention Weights")
	
	fmt.Println("\nEXPLANATION:")
	fmt.Println("- Q and K have tridiagonal structure (1.0 on diagonal, 0.5 on adjacent)")
	fmt.Println("- This creates higher scores for self and immediate neighbors")
	fmt.Println("- V matrix has high values (10.0) on diagonal, low (1.0) elsewhere")
	fmt.Println("- Result emphasizes local context while allowing some global information")
}

// TestRandomMatrices generates and tests random matrices
func TestRandomMatrices() {
	fmt.Printf("\n" + strings.Repeat("=", 80) + "\n")
	fmt.Printf("RANDOM MATRIX TEST\n")
	fmt.Printf(strings.Repeat("=", 80) + "\n")
	fmt.Println("Testing with random matrices to show general behavior")
	fmt.Println("This represents realistic scenarios with learned embeddings.\n")

	// Generate random matrices
	Q := GenerateRandomMatrix(8, 8, 0.0, 1.0)
	K := GenerateRandomMatrix(8, 8, 0.0, 1.0)
	V := GenerateRandomMatrix(8, 8, 0.0, 1.0)

	fmt.Println("Generated random matrices (first 3 rows shown):")
	for i := 0; i < 3; i++ {
		fmt.Printf("Q[%d]: [%.3f, %.3f, %.3f, %.3f, ...]\n", i, Q[i][0], Q[i][1], Q[i][2], Q[i][3])
	}

	fmt.Println("\nRunning Scaled Dot-Product Attention...")
	// Test attention but suppress detailed output
	result := ScaledDotProductAttentionQuiet(Q, K, V)
	result.Print("Random Matrix Attention Output")
	
	// Calculate and analyze attention weights
	KT := K.Transpose()
	QKT := Q.Multiply(KT)
	scaled := QKT.Scale(1.0 / 2.828) // sqrt(8)
	attentionWeights := scaled.Softmax()
	AnalyzeAttentionWeights(attentionWeights, "Random Matrix Attention")
	
	fmt.Println("\nEXPLANATION:")
	fmt.Println("- Random matrices create varied attention patterns")
	fmt.Println("- Some positions may attend strongly to specific others")
	fmt.Println("- This demonstrates how learned embeddings create different focus")
	fmt.Println("- Attention variance shows how 'focused' vs 'distributed' the attention is")
}

// ScaledDotProductAttentionQuiet - same as original but without verbose output
func ScaledDotProductAttentionQuiet(Q, K, V Matrix) Matrix {
	KT := K.Transpose()
	QKT := Q.Multiply(KT)
	dk := float64(len(K[0]))
	scalingFactor := 1.0 / math.Sqrt(dk)
	scaled := QKT.Scale(scalingFactor)
	attentionWeights := scaled.Softmax()
	result := attentionWeights.Multiply(V)
	return result
}

func main() {
	fmt.Println("TRANSFORMER ATTENTION COMPREHENSIVE TESTING")
	fmt.Println("===========================================")
	fmt.Println("This test suite demonstrates different attention mechanisms")
	fmt.Println("and their behaviors with various input patterns.\n")

	// Test 1: Identity matrices
	TestIdentityMatrices()

	// Test 2: Uniform attention
	TestUniformAttention() 

	// Test 3: Sequential attention
	TestSequentialAttention()

	// Test 4: Random matrices
	TestRandomMatrices()

	fmt.Printf("\n" + strings.Repeat("=", 80) + "\n")
	fmt.Printf("SUMMARY OF ATTENTION PATTERNS\n")
	fmt.Printf(strings.Repeat("=", 80) + "\n")
	fmt.Println("1. IDENTITY: Perfect self-attention → tokens only focus on themselves")
	fmt.Println("2. UNIFORM: Equal attention → global average pooling behavior") 
	fmt.Println("3. SEQUENTIAL: Local attention → emphasizes neighbors (like CNN)")
	fmt.Println("4. RANDOM: Varied patterns → realistic learned behavior")
	fmt.Println("\nKey Insights:")
	fmt.Println("- Attention weights always sum to 1.0 (softmax property)")
	fmt.Println("- Different Q/K patterns create different focus behaviors")
	fmt.Println("- Values determine what information gets aggregated")
	fmt.Println("- Multi-head attention captures multiple relationship types")
	fmt.Printf(strings.Repeat("=", 80) + "\n")
}