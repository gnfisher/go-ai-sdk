package main

import (
	"context"
	"fmt"
	"os"

	ai "github.com/gnfisher/go-ai-sdk"
	"github.com/gnfisher/go-ai-sdk/providers/anthropic"
)

type Person struct {
	Name    string   `json:"name"`
	Age     int      `json:"age"`
	Hobbies []string `json:"hobbies"`
}

func main() {
	// Get API key from environment variable
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("ANTHROPIC_API_KEY environment variable is required")
		os.Exit(1)
	}

	// Create Anthropic provider
	anthropicProvider := anthropic.New(
		anthropic.WithAPIKey(apiKey),
	)

	// Create AI client
	client := ai.NewClient()

	// Register Anthropic provider
	client.RegisterProvider(ai.ProviderAnthropic, anthropicProvider)

	// Example 1: Get text response
	textExample(client)

	// Example 2: Get structured response
	objectExample(client)
}

func textExample(client *ai.Client) {
	fmt.Println("=== Text Response Example ===")

	response, err := client.GetText(
		context.Background(),
		ai.WithProvider(ai.ProviderAnthropic),
		ai.WithModel("claude-3-haiku-20240307"),
		ai.WithMessages(
			ai.SystemMessage("You are a helpful assistant."),
			ai.UserMessage("Tell me a short joke about programming."),
		),
	)

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("Response:")
	fmt.Println(response)
	fmt.Println()
}

func objectExample(client *ai.Client) {
	fmt.Println("=== Structured Response Example ===")

	var person Person

	err := client.GetObject(
		context.Background(),
		&person,
		ai.WithProvider(ai.ProviderAnthropic),
		ai.WithModel("claude-3-haiku-20240307"),
		ai.WithMessages(
			ai.UserMessage("Create a fictional person with name, age, and hobbies."),
		),
	)

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("Response:")
	fmt.Printf("Name: %s\n", person.Name)
	fmt.Printf("Age: %d\n", person.Age)
	fmt.Printf("Hobbies: %v\n", person.Hobbies)
}
