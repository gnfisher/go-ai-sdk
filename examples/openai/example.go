package openai

import (
	"context"
	"fmt"
	"log"
	"os"

	ai "github.com/gnfisher/go-ai-sdk"
	"github.com/gnfisher/go-ai-sdk/providers/openai"
)

// OpenAIExample demonstrates how to use the SDK with the OpenAI provider
func OpenAIExample() {
	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create an OpenAI provider
	openaiProvider := openai.New(
		openai.WithAPIKey(apiKey),
	)

	// Create a client and register the OpenAI provider
	client := ai.NewClient(
		ai.WithProvider(ai.ProviderOpenAI),
		ai.WithModel("gpt-3.5-turbo"),
	)
	client.RegisterProvider(ai.ProviderOpenAI, openaiProvider)

	// Example 1: Get a text response
	response, err := client.GetText(
		context.Background(),
		ai.WithMessages(
			ai.SystemMessage("You are a helpful assistant."),
			ai.UserMessage("What is the capital of France?"),
		),
	)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println("Text response:")
	fmt.Println(response)
	fmt.Println()

	// Example 2: Get a structured response
	type CapitalInfo struct {
		Capital    string `json:"capital"`
		Country    string `json:"country"`
		Continent  string `json:"continent"`
		Population int    `json:"population"`
	}

	var capitalInfo CapitalInfo
	err = client.GetObject(
		context.Background(),
		&capitalInfo,
		ai.WithMessages(
			ai.SystemMessage("You are a helpful assistant that provides information about capital cities."),
			ai.UserMessage("Provide information about Paris."),
		),
	)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Println("Structured response:")
	fmt.Printf("Capital: %s\n", capitalInfo.Capital)
	fmt.Printf("Country: %s\n", capitalInfo.Country)
	fmt.Printf("Continent: %s\n", capitalInfo.Continent)
	fmt.Printf("Population: %d\n", capitalInfo.Population)
}
