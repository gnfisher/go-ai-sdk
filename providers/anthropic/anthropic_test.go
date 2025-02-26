package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gnfisher/go-ai-sdk"
)

type TestStruct struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestGetText(t *testing.T) {
	tests := []struct {
		name           string
		config         *ai.Config
		mockResponse   *Response
		mockStatusCode int
		expectError    bool
		expectedResult string
	}{
		{
			name: "successful response",
			config: &ai.Config{
				Model:    "claude-3-haiku-20240307",
				Messages: []ai.Message{ai.UserMessage("Hello")},
			},
			mockResponse: &Response{
				ID:   "msg_123",
				Type: "message",
				Role: "assistant",
				Content: []Content{
					{Type: "text", Text: "Hello! How can I help you today?"},
				},
				Model:      "claude-3-haiku-20240307",
				StopReason: "end_turn",
			},
			mockStatusCode: http.StatusOK,
			expectError:    false,
			expectedResult: "Hello! How can I help you today?",
		},
		{
			name: "error response",
			config: &ai.Config{
				Model:    "claude-3-haiku-20240307",
				Messages: []ai.Message{ai.UserMessage("Hello")},
			},
			mockResponse: &Response{
				Error: &Error{
					Type:    "invalid_request_error",
					Message: "Invalid API key",
				},
			},
			mockStatusCode: http.StatusUnauthorized,
			expectError:    true,
			expectedResult: "",
		},
		{
			name: "empty response",
			config: &ai.Config{
				Model:    "claude-3-haiku-20240307",
				Messages: []ai.Message{ai.UserMessage("Hello")},
			},
			mockResponse:   &Response{},
			mockStatusCode: http.StatusOK,
			expectError:    true,
			expectedResult: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a mock server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Check request headers
				if r.Header.Get("x-api-key") != "test-key" {
					t.Errorf("Expected x-api-key header to be 'test-key', got %s", r.Header.Get("x-api-key"))
				}

				if r.Header.Get("anthropic-version") != "2023-06-01" {
					t.Errorf("Expected anthropic-version header to be '2023-06-01', got %s", r.Header.Get("anthropic-version"))
				}

				// Set response status code
				w.WriteHeader(tt.mockStatusCode)

				// Write response body
				if tt.mockResponse != nil {
					if err := json.NewEncoder(w).Encode(tt.mockResponse); err != nil {
						t.Fatalf("failed to encode response: %v", err)
					}
				}
			}))
			defer server.Close()

			// Create provider with mock server URL
			provider := New(
				WithAPIKey("test-key"),
				WithAPIURL(server.URL),
			)

			// Call GetText
			result, err := provider.GetText(context.Background(), tt.config)

			// Check error
			if (err != nil) != tt.expectError {
				t.Errorf("GetText() error = %v, expectError %v", err, tt.expectError)
				return
			}

			// Check result
			if result != tt.expectedResult {
				t.Errorf("GetText() = %v, want %v", result, tt.expectedResult)
			}
		})
	}
}

func TestGetObject(t *testing.T) {
	tests := []struct {
		name           string
		config         *ai.Config
		mockResponse   *Response
		mockStatusCode int
		expectError    bool
		expectedResult *TestStruct
	}{
		{
			name: "successful response",
			config: &ai.Config{
				Model:    "claude-3-haiku-20240307",
				Messages: []ai.Message{ai.UserMessage("Get me a person")},
			},
			mockResponse: &Response{
				ID:   "msg_123",
				Type: "message",
				Role: "assistant",
				Content: []Content{
					{Type: "text", Text: `{"name":"John Doe","age":30}`},
				},
				Model:      "claude-3-haiku-20240307",
				StopReason: "end_turn",
			},
			mockStatusCode: http.StatusOK,
			expectError:    false,
			expectedResult: &TestStruct{
				Name: "John Doe",
				Age:  30,
			},
		},
		{
			name: "response with markdown",
			config: &ai.Config{
				Model:    "claude-3-haiku-20240307",
				Messages: []ai.Message{ai.UserMessage("Get me a person")},
			},
			mockResponse: &Response{
				ID:   "msg_123",
				Type: "message",
				Role: "assistant",
				Content: []Content{
					{Type: "text", Text: "```json\n{\"name\":\"John Doe\",\"age\":30}\n```"},
				},
				Model:      "claude-3-haiku-20240307",
				StopReason: "end_turn",
			},
			mockStatusCode: http.StatusOK,
			expectError:    false,
			expectedResult: &TestStruct{
				Name: "John Doe",
				Age:  30,
			},
		},
		{
			name: "error response",
			config: &ai.Config{
				Model:    "claude-3-haiku-20240307",
				Messages: []ai.Message{ai.UserMessage("Get me a person")},
			},
			mockResponse: &Response{
				Error: &Error{
					Type:    "invalid_request_error",
					Message: "Invalid API key",
				},
			},
			mockStatusCode: http.StatusUnauthorized,
			expectError:    true,
			expectedResult: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a mock server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Check request headers
				if r.Header.Get("x-api-key") != "test-key" {
					t.Errorf("Expected x-api-key header to be 'test-key', got %s", r.Header.Get("x-api-key"))
				}

				if r.Header.Get("anthropic-version") != "2023-06-01" {
					t.Errorf("Expected anthropic-version header to be '2023-06-01', got %s", r.Header.Get("anthropic-version"))
				}

				// Set response status code
				w.WriteHeader(tt.mockStatusCode)

				// Write response body
				if tt.mockResponse != nil {
					if err := json.NewEncoder(w).Encode(tt.mockResponse); err != nil {
						t.Fatalf("failed to encode response: %v", err)
					}
				}
			}))
			defer server.Close()

			// Create provider with mock server URL
			provider := New(
				WithAPIKey("test-key"),
				WithAPIURL(server.URL),
			)

			// Create result object
			result := &TestStruct{}

			// Call GetObject
			err := provider.GetObject(context.Background(), tt.config, result)

			// Check error
			if (err != nil) != tt.expectError {
				t.Errorf("GetObject() error = %v, expectError %v", err, tt.expectError)
				return
			}

			// Check result
			if tt.expectedResult != nil {
				if result.Name != tt.expectedResult.Name || result.Age != tt.expectedResult.Age {
					t.Errorf("GetObject() = %+v, want %+v", result, tt.expectedResult)
				}
			}
		})
	}
}

func TestConvertMessages(t *testing.T) {
	tests := []struct {
		name              string
		input             []ai.Message
		expectedMessages  []Message
		expectedSystemMsg string
	}{
		{
			name: "messages with system message",
			input: []ai.Message{
				ai.SystemMessage("You are a helpful assistant"),
				ai.UserMessage("Hello"),
				ai.AssistantMessage("Hi there"),
			},
			expectedMessages: []Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there"},
			},
			expectedSystemMsg: "You are a helpful assistant",
		},
		{
			name: "messages without system message",
			input: []ai.Message{
				ai.UserMessage("Hello"),
				ai.AssistantMessage("Hi there"),
			},
			expectedMessages: []Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there"},
			},
			expectedSystemMsg: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages, systemMsg := convertMessages(tt.input)

			// Check system message
			if systemMsg != tt.expectedSystemMsg {
				t.Errorf("convertMessages() systemMsg = %v, want %v", systemMsg, tt.expectedSystemMsg)
			}

			// Check messages
			if len(messages) != len(tt.expectedMessages) {
				t.Errorf("convertMessages() messages length = %v, want %v", len(messages), len(tt.expectedMessages))
				return
			}

			for i, msg := range messages {
				if msg.Role != tt.expectedMessages[i].Role || msg.Content != tt.expectedMessages[i].Content {
					t.Errorf("convertMessages() message[%d] = %+v, want %+v", i, msg, tt.expectedMessages[i])
				}
			}
		})
	}
}

func TestProviderOptions(t *testing.T) {
	// Test default values
	provider := New()
	if provider.apiURL != defaultAPIURL {
		t.Errorf("Expected default apiURL to be %s, got %s", defaultAPIURL, provider.apiURL)
	}
	if provider.apiKey != "" {
		t.Errorf("Expected default apiKey to be empty, got %s", provider.apiKey)
	}
	if provider.client != http.DefaultClient {
		t.Errorf("Expected default client to be http.DefaultClient")
	}

	// Test with options
	customClient := &http.Client{}
	provider = New(
		WithAPIKey("test-key"),
		WithAPIURL("https://custom-url.com"),
		WithHTTPClient(customClient),
	)

	if provider.apiKey != "test-key" {
		t.Errorf("Expected apiKey to be 'test-key', got %s", provider.apiKey)
	}
	if provider.apiURL != "https://custom-url.com" {
		t.Errorf("Expected apiURL to be 'https://custom-url.com', got %s", provider.apiURL)
	}
	if provider.client != customClient {
		t.Errorf("Expected client to be the custom client")
	}
}
