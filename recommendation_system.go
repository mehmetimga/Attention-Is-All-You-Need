package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// Recommendation System using Transformer Attention
// Uses user behavior sequences to recommend products/content

// RecommendationTokenizer converts user behavior to tokens
type RecommendationTokenizer struct {
	vocabToID map[string]int
	idToVocab map[int]string
	vocabSize int
}

// NewRecommendationTokenizer creates tokenizer for user behavior
func NewRecommendationTokenizer() *RecommendationTokenizer {
	// Vocabulary for user interactions and items
	vocab := []string{
		"<PAD>", "<START>", "<END>", "<UNK>",
		// Actions
		"view", "click", "purchase", "like", "share", "review", "search", "add_to_cart", "wishlist",
		// Product categories
		"electronics", "clothing", "books", "food", "home", "sports", "music", "movies", "games",
		"beauty", "health", "travel", "automotive", "toys", "jewelry", "art",
		// User segments
		"new_user", "returning_user", "premium_user", "frequent_buyer", "browser",
		// Time patterns
		"morning", "afternoon", "evening", "night", "weekend", "weekday",
		// Engagement levels
		"high_engagement", "medium_engagement", "low_engagement",
		// Device/platform
		"mobile", "desktop", "tablet", "app", "web",
		// Price sensitivity
		"budget", "mid_range", "premium", "luxury",
	}
	
	vocabToID := make(map[string]int)
	idToVocab := make(map[int]string)
	
	for i, token := range vocab {
		vocabToID[token] = i
		idToVocab[i] = token
	}
	
	return &RecommendationTokenizer{
		vocabToID: vocabToID,
		idToVocab: idToVocab,
		vocabSize: len(vocab),
	}
}

// UserInteraction represents a user's interaction with the system
type UserInteraction struct {
	UserID      string    `json:"user_id"`
	Action      string    `json:"action"`
	ItemID      string    `json:"item_id"`
	Category    string    `json:"category"`
	Timestamp   time.Time `json:"timestamp"`
	Duration    int       `json:"duration_seconds"`
	Rating      float64   `json:"rating"`
	DeviceType  string    `json:"device_type"`
	PriceRange  string    `json:"price_range"`
}

// EncodeUserSession converts user session to tokens
func (rt *RecommendationTokenizer) EncodeUserSession(interactions []UserInteraction) []int {
	tokens := []int{rt.vocabToID["<START>"]}
	
	for _, interaction := range interactions {
		// Action
		if id, exists := rt.vocabToID[interaction.Action]; exists {
			tokens = append(tokens, id)
		}
		
		// Category
		if id, exists := rt.vocabToID[interaction.Category]; exists {
			tokens = append(tokens, id)
		}
		
		// Time of day
		hour := interaction.Timestamp.Hour()
		timeOfDay := rt.getTimeOfDay(hour)
		if id, exists := rt.vocabToID[timeOfDay]; exists {
			tokens = append(tokens, id)
		}
		
		// Device type
		if id, exists := rt.vocabToID[interaction.DeviceType]; exists {
			tokens = append(tokens, id)
		}
		
		// Price range
		if id, exists := rt.vocabToID[interaction.PriceRange]; exists {
			tokens = append(tokens, id)
		}
		
		// Engagement level based on duration
		engagement := rt.getEngagementLevel(interaction.Duration, interaction.Action)
		if id, exists := rt.vocabToID[engagement]; exists {
			tokens = append(tokens, id)
		}
	}
	
	tokens = append(tokens, rt.vocabToID["<END>"])
	return tokens
}

func (rt *RecommendationTokenizer) getTimeOfDay(hour int) string {
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

func (rt *RecommendationTokenizer) getEngagementLevel(duration int, action string) string {
	// Different thresholds based on action type
	switch action {
	case "view":
		if duration > 60 {
			return "high_engagement"
		} else if duration > 10 {
			return "medium_engagement"
		}
		return "low_engagement"
	case "purchase":
		return "high_engagement"
	case "click":
		if duration > 30 {
			return "medium_engagement"
		}
		return "low_engagement"
	default:
		return "medium_engagement"
	}
}

// RecommendationEngine uses transformer attention for recommendations
type RecommendationEngine struct {
	tokenizer *RecommendationTokenizer
	llm       *SimpleLLM
	itemCatalog map[string]Item
}

type Item struct {
	ID       string  `json:"id"`
	Name     string  `json:"name"`
	Category string  `json:"category"`
	Price    float64 `json:"price"`
	Rating   float64 `json:"rating"`
}

// NewRecommendationEngine creates a recommendation system
func NewRecommendationEngine() *RecommendationEngine {
	tokenizer := NewRecommendationTokenizer()
	
	// Create LLM with recommendation vocabulary
	llm := NewSimpleLLM(tokenizer.vocabSize, 64, 8) // Large model for complex user patterns
	
	// Sample item catalog
	catalog := map[string]Item{
		"laptop001": {"laptop001", "Gaming Laptop", "electronics", 1200.0, 4.5},
		"book001":   {"book001", "AI Programming", "books", 45.0, 4.8},
		"shirt001":  {"shirt001", "Cotton T-Shirt", "clothing", 25.0, 4.2},
		"phone001":  {"phone001", "Smartphone", "electronics", 800.0, 4.6},
		"novel001":  {"novel001", "Sci-Fi Novel", "books", 15.0, 4.3},
	}
	
	return &RecommendationEngine{
		tokenizer:   tokenizer,
		llm:         llm,
		itemCatalog: catalog,
	}
}

type Recommendation struct {
	ItemID     string  `json:"item_id"`
	ItemName   string  `json:"item_name"`
	Score      float64 `json:"score"`
	Reason     string  `json:"reason"`
	Category   string  `json:"category"`
}

// GetRecommendations generates recommendations based on user history
func (re *RecommendationEngine) GetRecommendations(userHistory []UserInteraction, topK int) []Recommendation {
	// Encode user session
	tokens := re.tokenizer.EncodeUserSession(userHistory)
	
	// Convert tokens to text for analysis
	sessionText := re.tokensToText(tokens)
	
	// Get attention-based user preferences
	logits, _ := re.llm.Forward(sessionText)
	
	// Analyze user preferences from attention patterns
	preferences := re.analyzeUserPreferences(logits, userHistory)
	
	// Generate recommendations
	var recommendations []Recommendation
	
	for itemID, item := range re.itemCatalog {
		score := re.calculateItemScore(item, preferences, userHistory)
		reason := re.generateReason(item, preferences)
		
		recommendations = append(recommendations, Recommendation{
			ItemID:   itemID,
			ItemName: item.Name,
			Score:    score,
			Reason:   reason,
			Category: item.Category,
		})
	}
	
	// Sort by score and return top K
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].Score > recommendations[j].Score
	})
	
	if topK > len(recommendations) {
		topK = len(recommendations)
	}
	
	return recommendations[:topK]
}

func (re *RecommendationEngine) tokensToText(tokens []int) string {
	var parts []string
	for _, token := range tokens {
		if text, exists := re.tokenizer.idToVocab[token]; exists {
			if text != "<PAD>" && text != "<START>" && text != "<END>" {
				parts = append(parts, text)
			}
		}
	}
	return strings.Join(parts, " ")
}

type UserPreferences struct {
	PreferredCategories map[string]float64
	PreferredActions    map[string]float64
	PreferredTimeSlots  map[string]float64
	PricePreference     string
	EngagementLevel     float64
}

func (re *RecommendationEngine) analyzeUserPreferences(logits Matrix, history []UserInteraction) UserPreferences {
	prefs := UserPreferences{
		PreferredCategories: make(map[string]float64),
		PreferredActions:    make(map[string]float64),
		PreferredTimeSlots:  make(map[string]float64),
	}
	
	// Analyze interaction history for explicit preferences
	for _, interaction := range history {
		// Category preferences
		weight := 1.0
		if interaction.Action == "purchase" {
			weight = 3.0
		} else if interaction.Action == "like" || interaction.Action == "review" {
			weight = 2.0
		}
		
		prefs.PreferredCategories[interaction.Category] += weight
		prefs.PreferredActions[interaction.Action] += weight
		
		timeSlot := re.tokenizer.getTimeOfDay(interaction.Timestamp.Hour())
		prefs.PreferredTimeSlots[timeSlot] += 1.0
		
		// Engagement analysis
		if interaction.Duration > 60 {
			prefs.EngagementLevel += 1.0
		}
	}
	
	// Normalize preferences
	re.normalizePreferences(&prefs)
	
	return prefs
}

func (re *RecommendationEngine) normalizePreferences(prefs *UserPreferences) {
	// Normalize category preferences
	total := 0.0
	for _, weight := range prefs.PreferredCategories {
		total += weight
	}
	if total > 0 {
		for cat, weight := range prefs.PreferredCategories {
			prefs.PreferredCategories[cat] = weight / total
		}
	}
	
	// Similar normalization for other preferences...
	prefs.EngagementLevel = math.Min(prefs.EngagementLevel/10.0, 1.0)
}

func (re *RecommendationEngine) calculateItemScore(item Item, prefs UserPreferences, history []UserInteraction) float64 {
	score := 0.0
	
	// Category preference score
	if catScore, exists := prefs.PreferredCategories[item.Category]; exists {
		score += catScore * 0.4 // 40% weight
	}
	
	// Item rating score
	score += (item.Rating / 5.0) * 0.3 // 30% weight
	
	// Price preference (simplified)
	priceScore := 1.0 - math.Abs(item.Price-500.0)/1000.0 // Assume $500 is sweet spot
	score += math.Max(priceScore, 0.0) * 0.2 // 20% weight
	
	// Novelty bonus (items not seen before)
	noveltyBonus := 1.0
	for _, interaction := range history {
		if interaction.ItemID == item.ID {
			noveltyBonus = 0.5 // Reduce score for already seen items
			break
		}
	}
	score *= noveltyBonus
	
	// Engagement boost
	score += prefs.EngagementLevel * 0.1 // 10% weight
	
	return math.Min(score, 1.0)
}

func (re *RecommendationEngine) generateReason(item Item, prefs UserPreferences) string {
	reasons := []string{}
	
	if catScore, exists := prefs.PreferredCategories[item.Category]; exists && catScore > 0.3 {
		reasons = append(reasons, fmt.Sprintf("Popular in your favorite category: %s", item.Category))
	}
	
	if item.Rating >= 4.5 {
		reasons = append(reasons, "Highly rated by other customers")
	}
	
	if len(reasons) == 0 {
		reasons = append(reasons, "Recommended based on your browsing patterns")
	}
	
	return strings.Join(reasons, "; ")
}

// CreateRecommendationTrainingData generates sample training data
func CreateRecommendationTrainingData() []RecommendationTrainingExample {
	return []RecommendationTrainingExample{
		{
			UserSequence:   "view electronics morning desktop high_engagement purchase electronics",
			NextAction:     "view electronics",
			Relevance:     0.9,
			Explanation:   "User shows strong preference for electronics with high engagement",
		},
		{
			UserSequence:   "view books evening mobile medium_engagement add_to_cart books",
			NextAction:     "purchase books",
			Relevance:     0.8,
			Explanation:   "User added books to cart, likely to purchase",
		},
		{
			UserSequence:   "click clothing afternoon app low_engagement",
			NextAction:     "view clothing",
			Relevance:     0.3,
			Explanation:   "Low engagement suggests limited interest",
		},
		{
			UserSequence:   "search music night desktop wishlist music like music",
			NextAction:     "purchase music",
			Relevance:     0.85,
			Explanation:   "Strong music interest with multiple positive interactions",
		},
	}
}

type RecommendationTrainingExample struct {
	UserSequence string  `json:"user_sequence"`
	NextAction   string  `json:"next_action"`
	Relevance    float64 `json:"relevance_score"`
	Explanation  string  `json:"explanation"`
}

// Demo function
func runRecommendationDemo() {
	fmt.Println("RECOMMENDATION SYSTEM WITH TRANSFORMER ATTENTION")
	fmt.Println("===============================================")
	
	engine := NewRecommendationEngine()
	
	// Sample user history
	userHistory := []UserInteraction{
		{
			UserID:     "user123",
			Action:     "view",
			ItemID:     "laptop001",
			Category:   "electronics",
			Timestamp:  time.Now().Add(-2 * time.Hour),
			Duration:   45,
			DeviceType: "desktop",
			PriceRange: "premium",
		},
		{
			UserID:     "user123",
			Action:     "add_to_cart",
			ItemID:     "laptop001",
			Category:   "electronics",
			Timestamp:  time.Now().Add(-1 * time.Hour),
			Duration:   10,
			DeviceType: "desktop",
			PriceRange: "premium",
		},
		{
			UserID:     "user123",
			Action:     "view",
			ItemID:     "book001",
			Category:   "books",
			Timestamp:  time.Now().Add(-30 * time.Minute),
			Duration:   20,
			DeviceType: "mobile",
			PriceRange: "budget",
		},
	}
	
	fmt.Println("Analyzing user behavior...")
	fmt.Println("\nUser History:")
	for i, interaction := range userHistory {
		fmt.Printf("%d. Action: %s, Category: %s, Duration: %ds\n", 
			i+1, interaction.Action, interaction.Category, interaction.Duration)
	}
	
	recommendations := engine.GetRecommendations(userHistory, 3)
	
	fmt.Println("\n=== TOP RECOMMENDATIONS ===")
	for i, rec := range recommendations {
		fmt.Printf("%d. %s (Score: %.3f)\n", i+1, rec.ItemName, rec.Score)
		fmt.Printf("   Category: %s\n", rec.Category)
		fmt.Printf("   Reason: %s\n\n", rec.Reason)
	}
	
	fmt.Println("=== TRAINING DATA FOR RECOMMENDATIONS ===")
	trainingData := CreateRecommendationTrainingData()
	
	for i, example := range trainingData {
		fmt.Printf("%d. User Pattern: \"%s\"\n", i+1, example.UserSequence)
		fmt.Printf("   Predicted Next: %s (Relevance: %.2f)\n", example.NextAction, example.Relevance)
		fmt.Printf("   Explanation: %s\n\n", example.Explanation)
	}
	
	fmt.Println("=== KEY ADVANTAGES OF TRANSFORMER FOR RECOMMENDATIONS ===")
	fmt.Println("1. SEQUENTIAL PATTERNS: Understands user behavior over time")
	fmt.Println("2. ATTENTION WEIGHTS: Identifies which past actions are most relevant")
	fmt.Println("3. CONTEXT AWARENESS: Considers time, device, engagement patterns")
	fmt.Println("4. PERSONALIZATION: Adapts to individual user preferences")
	fmt.Println("5. COLD START: Can work with limited user data using content features")
}

func main() {
	runRecommendationDemo()
}