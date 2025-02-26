package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	ai "github.com/gnfisher/go-ai-sdk"
	"github.com/gnfisher/go-ai-sdk/providers/openai"
)

type WeatherParams struct {
	Location string `json:"location"`
	Unit     string `json:"unit,omitempty"`
}

type WeatherResponse struct {
	Temperature int    `json:"temperature"`
	Condition   string `json:"condition"`
}

func main() {
	// Get API key from environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("OPENAI_API_KEY environment variable is required")
		os.Exit(1)
	}

	// Create OpenAI provider
	openaiProvider := openai.New(
		openai.WithAPIKey(apiKey),
	)

	// Create AI client
	client := ai.NewClient()

	// Register OpenAI provider
	client.RegisterProvider(ai.ProviderOpenAI, openaiProvider)

	// Define the weather function
	weatherFunction := ai.FunctionDefinition{
		Name:        "get_weather",
		Description: "Get the current weather for a location",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"location": {
					"type": "string",
					"description": "The city and state or country, e.g. San Francisco, CA"
				},
				"unit": {
					"type": "string", 
					"enum": ["celsius", "fahrenheit"],
					"description": "The unit of temperature"
				}
			},
			"required": ["location"]
		}`),
	}

	// Get tool calls from the model
	toolCalls, err := client.GetToolCalls(
		context.Background(),
		ai.WithProvider(ai.ProviderOpenAI),
		ai.WithModel("gpt-4"),
		ai.WithMessages(
			ai.UserMessage("What's the weather in San Francisco?"),
		),
		ai.WithTools(weatherFunction),
	)

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Process tool calls
	fmt.Printf("Received %d tool calls\n", len(toolCalls))

	var toolResponses []ai.Message
	for _, toolCall := range toolCalls {
		fmt.Printf("Tool Call ID: %s\n", toolCall.ID)
		fmt.Printf("Tool Name: %s\n", toolCall.Tool.Name)
		fmt.Printf("Arguments: %s\n\n", string(toolCall.Tool.Arguments))

		// Process the tool call based on the function name
		if toolCall.Tool.Name == "get_weather" {
			// Parse arguments
			var params WeatherParams
			if err := json.Unmarshal(toolCall.Tool.Arguments, &params); err != nil {
				fmt.Printf("Error parsing arguments: %v\n", err)
				continue
			}

			// In a real application, you would call an actual weather API here
			// For this example, we'll simulate a response
			weatherResp := simulateWeatherAPI(params)
			weatherRespJSON, _ := json.Marshal(weatherResp)

			// Create tool response message
			toolResponses = append(toolResponses, ai.ToolResultMessage(
				toolCall.ID,
				string(weatherRespJSON),
			))
		}
	}

	// If we have any tool responses, send them back to the model for a final answer
	if len(toolResponses) > 0 {
		messages := []ai.Message{
			ai.UserMessage("What's the weather in San Francisco?"),
		}
		// Add assistant message with tool calls
		assistantMsg := ai.AssistantMessage("")
		assistantMsg.ToolCalls = toolCalls
		messages = append(messages, assistantMsg)

		// Add tool responses
		messages = append(messages, toolResponses...)

		// Get final response
		response, err := client.GetText(
			context.Background(),
			ai.WithProvider(ai.ProviderOpenAI),
			ai.WithModel("gpt-4"),
			ai.WithMessages(messages...),
		)

		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return
		}

		fmt.Println("Final Response:")
		fmt.Println(response)
	}
}

// Simulate a weather API call
func simulateWeatherAPI(params WeatherParams) WeatherResponse {
	// In a real application, you would call an actual weather API here
	return WeatherResponse{
		Temperature: 22,
		Condition:   "Sunny with some clouds",
	}
}
