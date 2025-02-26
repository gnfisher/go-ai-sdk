package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gnfisher/go-ai-sdk"
)

// mockServer creates a test server that returns a predefined response
func mockServer(statusCode int, responseBody string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(statusCode)
		if _, err := w.Write([]byte(responseBody)); err != nil {
			panic(err)
		}
	}))
}

func TestNew(t *testing.T) {
	// Test with default options
	provider := New()
	if provider.apiURL != defaultAPIURL {
		t.Errorf("Expected API URL to be %s, got %s", defaultAPIURL, provider.apiURL)
	}
	if provider.client != http.DefaultClient {
		t.Errorf("Expected client to be http.DefaultClient")
	}

	// Test with custom options
	customURL := "https://custom.openai.com/v1"
	customClient := &http.Client{}
	provider = New(
		WithAPIKey("test-key"),
		WithAPIURL(customURL),
		WithHTTPClient(customClient),
	)

	if provider.apiKey != "test-key" {
		t.Errorf("Expected API key to be test-key, got %s", provider.apiKey)
	}
	if provider.apiURL != customURL {
		t.Errorf("Expected API URL to be %s, got %s", customURL, provider.apiURL)
	}
	if provider.client != customClient {
		t.Errorf("Expected client to be customClient")
	}
}

func TestGetText(t *testing.T) {
	// Test missing API key
	provider := New()
	_, err := provider.GetText(context.Background(), &ai.Config{
		Model: "test-model",
	})
	if err != ErrEmptyAPIKey {
		t.Errorf("Expected ErrEmptyAPIKey, got %v", err)
	}

	// Test successful response
	mockResponse := Response{
		Choices: []Choice{
			{
				Message: Message{
					Content: "Hello, world!",
				},
			},
		},
	}
	mockResponseJSON, _ := json.Marshal(mockResponse)

	server := mockServer(http.StatusOK, string(mockResponseJSON))
	defer server.Close()

	provider = New(
		WithAPIKey("test-key"),
		WithAPIURL(server.URL),
	)

	result, err := provider.GetText(context.Background(), &ai.Config{
		Model: "test-model",
		Messages: []ai.Message{
			{
				Role:    ai.RoleUser,
				Content: "Hello",
			},
		},
		Temperature: 0.7,
		MaxTokens:   100,
	})

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result != "Hello, world!" {
		t.Errorf("Expected 'Hello, world!', got %s", result)
	}

	// Test error response
	errorResponse := Response{
		Error: &Error{
			Message: "Invalid model",
			Type:    "invalid_request_error",
		},
	}
	errorResponseJSON, _ := json.Marshal(errorResponse)

	server = mockServer(http.StatusBadRequest, string(errorResponseJSON))
	defer server.Close()

	provider = New(
		WithAPIKey("test-key"),
		WithAPIURL(server.URL),
	)

	_, err = provider.GetText(context.Background(), &ai.Config{
		Model: "invalid-model",
	})

	if err == nil {
		t.Errorf("Expected error, got nil")
	}
}

func TestGetObject(t *testing.T) {
	type TestResponse struct {
		Message string `json:"message"`
	}

	// Test missing API key
	provider := New()
	var resp TestResponse
	err := provider.GetObject(context.Background(), &ai.Config{
		Model: "test-model",
	}, &resp)
	if err != ErrEmptyAPIKey {
		t.Errorf("Expected ErrEmptyAPIKey, got %v", err)
	}

	// Test successful response
	mockResponse := Response{
		Choices: []Choice{
			{
				Message: Message{
					Content: `{"message": "Hello, world!"}`,
				},
			},
		},
	}
	mockResponseJSON, _ := json.Marshal(mockResponse)

	server := mockServer(http.StatusOK, string(mockResponseJSON))
	defer server.Close()

	provider = New(
		WithAPIKey("test-key"),
		WithAPIURL(server.URL),
	)

	err = provider.GetObject(context.Background(), &ai.Config{
		Model: "test-model",
		Messages: []ai.Message{
			{
				Role:    ai.RoleUser,
				Content: "Hello",
			},
		},
	}, &resp)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if resp.Message != "Hello, world!" {
		t.Errorf("Expected 'Hello, world!', got %s", resp.Message)
	}

	// Test response with JSON in code block
	mockResponse = Response{
		Choices: []Choice{
			{
				Message: Message{
					Content: "```json\n{\"message\": \"Hello, world!\"}\n```",
				},
			},
		},
	}
	mockResponseJSON, _ = json.Marshal(mockResponse)

	server = mockServer(http.StatusOK, string(mockResponseJSON))
	defer server.Close()

	provider = New(
		WithAPIKey("test-key"),
		WithAPIURL(server.URL),
	)

	resp = TestResponse{}
	err = provider.GetObject(context.Background(), &ai.Config{
		Model: "test-model",
		Messages: []ai.Message{
			{
				Role:    ai.RoleUser,
				Content: "Hello",
			},
		},
	}, &resp)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if resp.Message != "Hello, world!" {
		t.Errorf("Expected 'Hello, world!', got %s", resp.Message)
	}
}

func TestGetToolCalls(t *testing.T) {
	// Test missing API key
	provider := New()
	_, err := provider.GetToolCalls(context.Background(), &ai.Config{
		Model: "test-model",
		Tools: []ai.FunctionDefinition{
			{
				Name:        "get_weather",
				Description: "Gets weather information",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		},
	})
	if err != ErrEmptyAPIKey {
		t.Errorf("Expected ErrEmptyAPIKey, got %v", err)
	}

	// Test no tools specified
	provider = New(WithAPIKey("test-key"))
	_, err = provider.GetToolCalls(context.Background(), &ai.Config{
		Model: "test-model",
	})
	if err == nil || err.Error() != "no tools specified" {
		t.Errorf("Expected 'no tools specified' error, got %v", err)
	}

	// Test successful response with tool calls
	mockResponse := Response{
		Choices: []Choice{
			{
				Message: Message{
					Role:    "assistant",
					Content: "",
					ToolCalls: []ToolCall{
						{
							ID:   "call_abc123",
							Type: "function",
							Function: ToolFunction{
								Name:      "get_weather",
								Arguments: json.RawMessage(`{"location":"San Francisco","unit":"celsius"}`),
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
	}
	mockResponseJSON, _ := json.Marshal(mockResponse)

	server := mockServer(http.StatusOK, string(mockResponseJSON))
	defer server.Close()

	provider = New(
		WithAPIKey("test-key"),
		WithAPIURL(server.URL),
	)

	result, err := provider.GetToolCalls(context.Background(), &ai.Config{
		Model: "test-model",
		Messages: []ai.Message{
			{
				Role:    ai.RoleUser,
				Content: "What's the weather in San Francisco?",
			},
		},
		Tools: []ai.FunctionDefinition{
			{
				Name:        "get_weather",
				Description: "Gets weather information",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		},
	})

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if len(result) != 1 {
		t.Errorf("Expected 1 tool call, got %d", len(result))
		return
	}

	if result[0].ID != "call_abc123" {
		t.Errorf("Expected tool call ID 'call_abc123', got %s", result[0].ID)
	}

	if result[0].Type != "function" {
		t.Errorf("Expected tool call type 'function', got %s", result[0].Type)
	}

	if result[0].Tool.Name != "get_weather" {
		t.Errorf("Expected tool name 'get_weather', got %s", result[0].Tool.Name)
	}

	// Test response with no tool calls
	mockResponse = Response{
		Choices: []Choice{
			{
				Message: Message{
					Role:      "assistant",
					Content:   "I don't need to use a tool for this.",
					ToolCalls: []ToolCall{},
				},
				FinishReason: "stop",
			},
		},
	}
	mockResponseJSON, _ = json.Marshal(mockResponse)

	server = mockServer(http.StatusOK, string(mockResponseJSON))
	defer server.Close()

	provider = New(
		WithAPIKey("test-key"),
		WithAPIURL(server.URL),
	)

	result, err = provider.GetToolCalls(context.Background(), &ai.Config{
		Model: "test-model",
		Messages: []ai.Message{
			{
				Role:    ai.RoleUser,
				Content: "Hello, how are you?",
			},
		},
		Tools: []ai.FunctionDefinition{
			{
				Name:        "get_weather",
				Description: "Gets weather information",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		},
	})

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if len(result) != 0 {
		t.Errorf("Expected 0 tool calls, got %d", len(result))
	}

	// Test error response
	errorResponse := Response{
		Error: &Error{
			Message: "Invalid model",
			Type:    "invalid_request_error",
		},
	}
	errorResponseJSON, _ := json.Marshal(errorResponse)

	server = mockServer(http.StatusBadRequest, string(errorResponseJSON))
	defer server.Close()

	provider = New(
		WithAPIKey("test-key"),
		WithAPIURL(server.URL),
	)

	_, err = provider.GetToolCalls(context.Background(), &ai.Config{
		Model: "invalid-model",
		Tools: []ai.FunctionDefinition{
			{
				Name:        "get_weather",
				Description: "Gets weather information",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		},
	})

	if err == nil {
		t.Errorf("Expected error, got nil")
	}
}
