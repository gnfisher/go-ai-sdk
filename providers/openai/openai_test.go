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
		w.Write([]byte(responseBody))
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