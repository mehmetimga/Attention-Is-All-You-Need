package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// Application examples for the Simple LLM

// TextCompleter demonstrates text completion functionality
type TextCompleter struct {
	llm *SimpleLLM
}

func NewTextCompleter(llm *SimpleLLM) *TextCompleter {
	return &TextCompleter{llm: llm}
}

func (tc *TextCompleter) Complete(prompt string) string {
	return tc.llm.Generate(prompt, 15)
}

// SimpleChatbot demonstrates conversational AI
type SimpleChatbot struct {
	llm *SimpleLLM
	context []string
}

func NewSimpleChatbot(llm *SimpleLLM) *SimpleChatbot {
	return &SimpleChatbot{
		llm: llm,
		context: make([]string, 0),
	}
}

func (bot *SimpleChatbot) Respond(userInput string) string {
	// Add user input to context
	bot.context = append(bot.context, userInput)
	
	// Keep only last 3 exchanges for simple context
	if len(bot.context) > 3 {
		bot.context = bot.context[len(bot.context)-3:]
	}
	
	// Create prompt from context
	prompt := strings.Join(bot.context, " ")
	response := bot.llm.Generate(prompt, 12)
	
	// Add response to context
	bot.context = append(bot.context, response)
	
	return response
}

// CodeCompleter demonstrates code completion
type CodeCompleter struct {
	llm *SimpleLLM
}

func NewCodeCompleter(llm *SimpleLLM) *CodeCompleter {
	return &CodeCompleter{llm: llm}
}

func (cc *CodeCompleter) CompleteCode(codeFragment string) string {
	return cc.llm.Generate(codeFragment, 20)
}

// StoryGenerator demonstrates creative text generation
type StoryGenerator struct {
	llm *SimpleLLM
}

func NewStoryGenerator(llm *SimpleLLM) *StoryGenerator {
	return &StoryGenerator{llm: llm}
}

func (sg *StoryGenerator) GenerateStory(prompt string, length int) string {
	return sg.llm.Generate(prompt, length)
}

// TemplateSystem demonstrates form filling
type TemplateSystem struct {
	llm *SimpleLLM
	templates map[string]string
}

func NewTemplateSystem(llm *SimpleLLM) *TemplateSystem {
	templates := map[string]string{
		"email": "dear sir i am writing to",
		"letter": "hello friend i hope you",
		"report": "the results show that",
		"summary": "in conclusion we found",
	}
	
	return &TemplateSystem{
		llm: llm,
		templates: templates,
	}
}

func (ts *TemplateSystem) FillTemplate(templateName string) string {
	if template, exists := ts.templates[templateName]; exists {
		return ts.llm.Generate(template, 25)
	}
	return "template not found"
}

// Interactive demo function
func runInteractiveDemo(llm *SimpleLLM) {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("INTERACTIVE LLM DEMO - Choose an application:")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("1. Text Completion")
	fmt.Println("2. Simple Chatbot")
	fmt.Println("3. Code Completion")  
	fmt.Println("4. Story Generation")
	fmt.Println("5. Template Filling")
	fmt.Println("6. Exit")
	fmt.Print("\nSelect option (1-6): ")
	
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	choice := scanner.Text()
	
	switch choice {
	case "1":
		textCompletionDemo(llm, scanner)
	case "2":
		chatbotDemo(llm, scanner)
	case "3":
		codeCompletionDemo(llm, scanner)
	case "4":
		storyGenerationDemo(llm, scanner)
	case "5":
		templateFillingDemo(llm, scanner)
	case "6":
		fmt.Println("Goodbye!")
		return
	default:
		fmt.Println("Invalid choice. Try again.")
	}
	
	// Ask if user wants to try another application
	fmt.Print("\nTry another application? (y/n): ")
	scanner.Scan()
	if strings.ToLower(scanner.Text()) == "y" {
		runInteractiveDemo(llm)
	}
}

func textCompletionDemo(llm *SimpleLLM, scanner *bufio.Scanner) {
	completer := NewTextCompleter(llm)
	
	fmt.Println("\n--- TEXT COMPLETION DEMO ---")
	fmt.Println("Enter partial text to complete (or 'quit' to exit):")
	
	for {
		fmt.Print("You: ")
		scanner.Scan()
		input := scanner.Text()
		
		if input == "quit" {
			break
		}
		
		completion := completer.Complete(input)
		fmt.Printf("Completed: %s\n\n", completion)
	}
}

func chatbotDemo(llm *SimpleLLM, scanner *bufio.Scanner) {
	bot := NewSimpleChatbot(llm)
	
	fmt.Println("\n--- CHATBOT DEMO ---")
	fmt.Println("Chat with the bot (or 'quit' to exit):")
	
	for {
		fmt.Print("You: ")
		scanner.Scan()
		input := scanner.Text()
		
		if input == "quit" {
			break
		}
		
		response := bot.Respond(input)
		fmt.Printf("Bot: %s\n\n", response)
	}
}

func codeCompletionDemo(llm *SimpleLLM, scanner *bufio.Scanner) {
	coder := NewCodeCompleter(llm)
	
	fmt.Println("\n--- CODE COMPLETION DEMO ---")
	fmt.Println("Enter code fragments to complete (or 'quit' to exit):")
	
	examples := []string{
		"for i in",
		"if x equals", 
		"def function",
		"import",
		"print hello",
	}
	
	fmt.Println("Try these examples:")
	for _, example := range examples {
		fmt.Printf("  %s\n", example)
	}
	fmt.Println()
	
	for {
		fmt.Print("Code: ")
		scanner.Scan()
		input := scanner.Text()
		
		if input == "quit" {
			break
		}
		
		completion := coder.CompleteCode(input)
		fmt.Printf("Completed: %s\n\n", completion)
	}
}

func storyGenerationDemo(llm *SimpleLLM, scanner *bufio.Scanner) {
	generator := NewStoryGenerator(llm)
	
	fmt.Println("\n--- STORY GENERATION DEMO ---")
	fmt.Println("Enter story beginnings (or 'quit' to exit):")
	
	examples := []string{
		"once upon",
		"in the dark",
		"the brave knight",
		"long ago",
		"suddenly there",
	}
	
	fmt.Println("Try these prompts:")
	for _, example := range examples {
		fmt.Printf("  %s\n", example)
	}
	fmt.Println()
	
	for {
		fmt.Print("Story start: ")
		scanner.Scan()
		input := scanner.Text()
		
		if input == "quit" {
			break
		}
		
		story := generator.GenerateStory(input, 30)
		fmt.Printf("Story: %s\n\n", story)
	}
}

func templateFillingDemo(llm *SimpleLLM, scanner *bufio.Scanner) {
	templateSys := NewTemplateSystem(llm)
	
	fmt.Println("\n--- TEMPLATE FILLING DEMO ---")
	fmt.Println("Available templates:")
	fmt.Println("  email - Generate email text")
	fmt.Println("  letter - Generate letter text") 
	fmt.Println("  report - Generate report text")
	fmt.Println("  summary - Generate summary text")
	fmt.Println()
	
	for {
		fmt.Print("Template name (or 'quit' to exit): ")
		scanner.Scan()
		input := scanner.Text()
		
		if input == "quit" {
			break
		}
		
		filled := templateSys.FillTemplate(input)
		fmt.Printf("Generated: %s\n\n", filled)
	}
}

// TrainingDatasets provides different types of training data
type TrainingDatasets struct{}

func (td *TrainingDatasets) GetConversationalData() []TrainingExample {
	return []TrainingExample{
		{"hello", "hello how are you"},
		{"goodbye", "goodbye see you later"},
		{"thank you", "thank you very much"},
		{"good morning", "good morning have nice day"},
		{"how are you", "how are you doing today"},
		{"nice weather", "nice weather for walking"},
		{"see you later", "see you later take care"},
		{"have a good day", "have a good day friend"},
	}
}

func (td *TrainingDatasets) GetFactualData() []TrainingExample {
	return []TrainingExample{
		{"water freezes", "water freezes at zero degrees"},
		{"earth revolves", "earth revolves around sun"},
		{"cats are", "cats are domestic animals"},
		{"fire produces", "fire produces heat and light"},
		{"plants need", "plants need water and sunlight"},
		{"birds can", "birds can fly in sky"},
		{"fish live", "fish live in water"},
		{"sun provides", "sun provides light and warmth"},
	}
}

func (td *TrainingDatasets) GetCreativeData() []TrainingExample {
	return []TrainingExample{
		{"once upon time", "once upon time there lived"},
		{"in forest", "in forest deep and dark"},
		{"brave knight", "brave knight rode his horse"},
		{"magic castle", "magic castle stood on hill"},
		{"wise wizard", "wise wizard cast his spell"},
		{"beautiful princess", "beautiful princess sang sweetly"},
		{"dark dragon", "dark dragon guarded treasure"},
		{"happy ending", "happy ending for everyone"},
	}
}

func (td *TrainingDatasets) GetTechnicalData() []TrainingExample {
	return []TrainingExample{
		{"function takes", "function takes input parameters"},
		{"loop iterates", "loop iterates through array"},
		{"variable stores", "variable stores data value"},
		{"if statement", "if statement checks condition"},
		{"return value", "return value from function"},
		{"import library", "import library for functionality"},
		{"class defines", "class defines object blueprint"},
		{"method performs", "method performs specific task"},
	}
}

func main() {
	fmt.Println("SIMPLE LLM APPLICATIONS DEMO")
	fmt.Println("============================")
	
	// Create a simple LLM
	llm := NewSimpleLLM(42, 16, 2)
	
	fmt.Println("Available Training Data Types:")
	datasets := &TrainingDatasets{}
	
	fmt.Println("\n1. CONVERSATIONAL DATA:")
	conv := datasets.GetConversationalData()
	for i, ex := range conv[:3] {
		fmt.Printf("   %d. '%s' -> '%s'\n", i+1, ex.Input, ex.Target)
	}
	
	fmt.Println("\n2. FACTUAL DATA:")
	fact := datasets.GetFactualData()
	for i, ex := range fact[:3] {
		fmt.Printf("   %d. '%s' -> '%s'\n", i+1, ex.Input, ex.Target)
	}
	
	fmt.Println("\n3. CREATIVE DATA:")
	creative := datasets.GetCreativeData()
	for i, ex := range creative[:3] {
		fmt.Printf("   %d. '%s' -> '%s'\n", i+1, ex.Input, ex.Target)
	}
	
	fmt.Println("\n4. TECHNICAL DATA:")
	tech := datasets.GetTechnicalData()
	for i, ex := range tech[:3] {
		fmt.Printf("   %d. '%s' -> '%s'\n", i+1, ex.Input, ex.Target)
	}
	
	// Run interactive demo
	runInteractiveDemo(llm)
}