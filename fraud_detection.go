package main

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// Fraud Detection using Transformer Architecture
// The attention mechanism can identify suspicious patterns in transaction sequences

// TransactionTokenizer converts transaction data to tokens
type TransactionTokenizer struct {
	vocabToID map[string]int
	idToVocab map[int]string
	vocabSize int
}

// NewTransactionTokenizer creates tokenizer for financial transactions
func NewTransactionTokenizer() *TransactionTokenizer {
	// Vocabulary for transaction features
	vocab := []string{
		"<PAD>", "<START>", "<END>", "<UNK>",
		// Transaction types
		"debit", "credit", "transfer", "withdrawal", "deposit", "payment", "refund",
		// Amount ranges (bucketed)
		"tiny", "small", "medium", "large", "huge", "massive",
		// Time features
		"morning", "afternoon", "evening", "night", "weekend", "weekday",
		// Location/merchant types
		"online", "atm", "grocery", "gas", "restaurant", "retail", "bank",
		// Risk indicators
		"foreign", "new_device", "unusual_location", "high_velocity", "round_amount",
		// User behavior patterns
		"frequent_user", "new_user", "inactive_user", "verified", "unverified",
	}
	
	vocabToID := make(map[string]int)
	idToVocab := make(map[int]string)
	
	for i, token := range vocab {
		vocabToID[token] = i
		idToVocab[i] = token
	}
	
	return &TransactionTokenizer{
		vocabToID: vocabToID,
		idToVocab: idToVocab,
		vocabSize: len(vocab),
	}
}

// Transaction represents a financial transaction
type Transaction struct {
	Amount       float64   `json:"amount"`
	Type         string    `json:"type"`
	Timestamp    time.Time `json:"timestamp"`
	Location     string    `json:"location"`
	MerchantType string    `json:"merchant_type"`
	UserID       string    `json:"user_id"`
	IsFraud      bool      `json:"is_fraud"`
}

// EncodeTransaction converts transaction to tokens
func (tt *TransactionTokenizer) EncodeTransaction(tx Transaction) []int {
	tokens := []int{tt.vocabToID["<START>"]}
	
	// Transaction type
	if id, exists := tt.vocabToID[tx.Type]; exists {
		tokens = append(tokens, id)
	}
	
	// Amount bucket
	amountBucket := tt.getAmountBucket(tx.Amount)
	if id, exists := tt.vocabToID[amountBucket]; exists {
		tokens = append(tokens, id)
	}
	
	// Time of day
	hour := tx.Timestamp.Hour()
	timeOfDay := tt.getTimeOfDay(hour)
	if id, exists := tt.vocabToID[timeOfDay]; exists {
		tokens = append(tokens, id)
	}
	
	// Day type
	dayType := tt.getDayType(tx.Timestamp)
	if id, exists := tt.vocabToID[dayType]; exists {
		tokens = append(tokens, id)
	}
	
	// Merchant type
	if id, exists := tt.vocabToID[tx.MerchantType]; exists {
		tokens = append(tokens, id)
	}
	
	// Location indicator
	if id, exists := tt.vocabToID[tx.Location]; exists {
		tokens = append(tokens, id)
	}
	
	tokens = append(tokens, tt.vocabToID["<END>"])
	return tokens
}

func (tt *TransactionTokenizer) getAmountBucket(amount float64) string {
	switch {
	case amount < 10:
		return "tiny"
	case amount < 100:
		return "small"
	case amount < 1000:
		return "medium"
	case amount < 10000:
		return "large"
	case amount < 100000:
		return "huge"
	default:
		return "massive"
	}
}

func (tt *TransactionTokenizer) getTimeOfDay(hour int) string {
	switch {
	case hour < 6:
		return "night"
	case hour < 12:
		return "morning"
	case hour < 18:
		return "afternoon"
	default:
		return "evening"
	}
}

func (tt *TransactionTokenizer) getDayType(t time.Time) string {
	if t.Weekday() == time.Saturday || t.Weekday() == time.Sunday {
		return "weekend"
	}
	return "weekday"
}

// FraudDetector uses transformer attention for fraud detection
type FraudDetector struct {
	tokenizer *TransactionTokenizer
	llm       *SimpleLLM
	threshold float64
}

// NewFraudDetector creates a fraud detection system
func NewFraudDetector() *FraudDetector {
	tokenizer := NewTransactionTokenizer()
	
	// Create LLM with transaction vocabulary
	llm := NewSimpleLLM(tokenizer.vocabSize, 32, 4) // Larger model for complex patterns
	
	return &FraudDetector{
		tokenizer: tokenizer,
		llm:       llm,
		threshold: 0.5, // Fraud probability threshold
	}
}

// DetectFraud analyzes a sequence of transactions
func (fd *FraudDetector) DetectFraud(transactions []Transaction) []FraudResult {
	results := make([]FraudResult, len(transactions))
	
	// Process transactions in sequence (last 5 transactions for context)
	for i, tx := range transactions {
		// Get context window of previous transactions
		start := 0
		if i >= 4 {
			start = i - 4
		}
		context := transactions[start : i+1]
		
		// Convert to token sequence
		var allTokens []int
		for _, contextTx := range context {
			txTokens := fd.tokenizer.EncodeTransaction(contextTx)
			allTokens = append(allTokens, txTokens...)
		}
		
		// Get attention-based analysis
		fraudProb := fd.analyzeFraudProbability(allTokens)
		
		results[i] = FraudResult{
			TransactionID: fmt.Sprintf("tx_%d", i),
			FraudScore:    fraudProb,
			IsFraud:       fraudProb > fd.threshold,
			RiskFactors:   fd.identifyRiskFactors(tx),
		}
	}
	
	return results
}

type FraudResult struct {
	TransactionID string   `json:"transaction_id"`
	FraudScore    float64  `json:"fraud_score"`
	IsFraud       bool     `json:"is_fraud"`
	RiskFactors   []string `json:"risk_factors"`
}

func (fd *FraudDetector) analyzeFraudProbability(tokens []int) float64 {
	if len(tokens) == 0 {
		return 0.0
	}
	
	// Convert tokens to text for LLM analysis
	text := fd.tokensToText(tokens)
	
	// Get model output logits
	logits, _ := fd.llm.Forward(text)
	
	// Use last position logits as fraud indicators
	if len(logits) > 0 {
		lastLogits := logits[len(logits)-1]
		
		// Simple fraud scoring based on attention patterns
		// In practice, you'd train this end-to-end
		var fraudScore float64
		for _, logit := range lastLogits {
			fraudScore += math.Abs(logit) // Unusual patterns = higher absolute values
		}
		
		// Normalize to 0-1 range
		fraudScore = fraudScore / float64(len(lastLogits))
		fraudScore = math.Min(fraudScore, 1.0)
		
		return fraudScore
	}
	
	return 0.0
}

func (fd *FraudDetector) tokensToText(tokens []int) string {
	var parts []string
	for _, token := range tokens {
		if text, exists := fd.tokenizer.idToVocab[token]; exists {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, " ")
}

func (fd *FraudDetector) identifyRiskFactors(tx Transaction) []string {
	var factors []string
	
	// Amount-based risk
	if tx.Amount > 10000 {
		factors = append(factors, "high_amount")
	}
	
	// Round amount suspicion
	if tx.Amount == math.Trunc(tx.Amount) && tx.Amount > 100 {
		factors = append(factors, "round_amount")
	}
	
	// Time-based risk
	hour := tx.Timestamp.Hour()
	if hour < 6 || hour > 22 {
		factors = append(factors, "unusual_time")
	}
	
	// Weekend activity for business transactions
	if tx.Timestamp.Weekday() == time.Saturday || tx.Timestamp.Weekday() == time.Sunday {
		if tx.MerchantType == "bank" {
			factors = append(factors, "weekend_banking")
		}
	}
	
	return factors
}

// CreateFraudTrainingData generates sample training data
func CreateFraudTrainingData() []FraudTrainingExample {
	return []FraudTrainingExample{
		{
			Sequence: "debit large evening weekend atm",
			Label:    "normal",
			Score:    0.2,
		},
		{
			Sequence: "transfer massive night foreign new_device",
			Label:    "fraud",
			Score:    0.9,
		},
		{
			Sequence: "payment small morning weekday grocery verified",
			Label:    "normal",
			Score:    0.1,
		},
		{
			Sequence: "withdrawal huge night atm foreign unverified",
			Label:    "fraud",
			Score:    0.85,
		},
		{
			Sequence: "credit medium afternoon weekday online frequent_user",
			Label:    "normal",
			Score:    0.15,
		},
		{
			Sequence: "debit massive evening new_device high_velocity round_amount",
			Label:    "fraud",
			Score:    0.95,
		},
	}
}

type FraudTrainingExample struct {
	Sequence string  `json:"sequence"`
	Label    string  `json:"label"`
	Score    float64 `json:"score"`
}

// Demo function
func runFraudDetectionDemo() {
	fmt.Println("FRAUD DETECTION WITH TRANSFORMER ATTENTION")
	fmt.Println("==========================================")
	
	detector := NewFraudDetector()
	
	// Sample transactions
	transactions := []Transaction{
		{
			Amount:       50.0,
			Type:         "payment",
			Timestamp:    time.Now(),
			Location:     "online",
			MerchantType: "grocery",
			UserID:       "user123",
			IsFraud:      false,
		},
		{
			Amount:       15000.0,
			Type:         "transfer",
			Timestamp:    time.Date(2024, 1, 1, 2, 30, 0, 0, time.UTC),
			Location:     "foreign",
			MerchantType: "bank",
			UserID:       "user123",
			IsFraud:      true,
		},
		{
			Amount:       25.75,
			Type:         "debit",
			Timestamp:    time.Now(),
			Location:     "atm",
			MerchantType: "bank",
			UserID:       "user123",
			IsFraud:      false,
		},
	}
	
	fmt.Println("Analyzing transaction sequence...")
	results := detector.DetectFraud(transactions)
	
	for i, result := range results {
		fmt.Printf("\nTransaction %d:\n", i+1)
		fmt.Printf("  Amount: $%.2f\n", transactions[i].Amount)
		fmt.Printf("  Type: %s\n", transactions[i].Type)
		fmt.Printf("  Fraud Score: %.3f\n", result.FraudScore)
		fmt.Printf("  Is Fraud: %v\n", result.IsFraud)
		fmt.Printf("  Risk Factors: %v\n", result.RiskFactors)
	}
	
	fmt.Println("\n=== TRAINING DATA FOR FRAUD DETECTION ===")
	trainingData := CreateFraudTrainingData()
	
	fmt.Println("Sample training examples:")
	for i, example := range trainingData {
		fmt.Printf("%d. Sequence: \"%s\"\n", i+1, example.Sequence)
		fmt.Printf("   Label: %s (Score: %.2f)\n\n", example.Label, example.Score)
	}
	
	fmt.Println("=== KEY ADVANTAGES OF TRANSFORMER FOR FRAUD DETECTION ===")
	fmt.Println("1. SEQUENCE AWARENESS: Detects patterns across transaction history")
	fmt.Println("2. ATTENTION MECHANISM: Focuses on suspicious combinations")
	fmt.Println("3. CONTEXTUAL UNDERSTANDING: Considers time, location, amount patterns")
	fmt.Println("4. SCALABILITY: Processes multiple transactions simultaneously")
	fmt.Println("5. ADAPTABILITY: Learns new fraud patterns through training")
}

func main() {
	runFraudDetectionDemo()
}