package ai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

// MockProvider implements LLMProvider for testing
type MockProvider struct {
	GetTextFunc      func(ctx context.Context, config *Config) (string, error)
	GetObjectFunc    func(ctx context.Context, config *Config, target interface{}) error
	GetToolCallsFunc func(ctx context.Context, config *Config) ([]ToolCall, error)
}

func (m *MockProvider) GetText(ctx context.Context, config *Config) (string, error) {
	return m.GetTextFunc(ctx, config)
}

func (m *MockProvider) GetObject(ctx context.Context, config *Config, target interface{}) error {
	return m.GetObjectFunc(ctx, config, target)
}

func (m *MockProvider) GetToolCalls(ctx context.Context, config *Config) ([]ToolCall, error) {
	return m.GetToolCallsFunc(ctx, config)
}

func TestNewClient(t *testing.T) {
	client := NewClient()

	if client.defaults.Temperature != 0.7 {
		t.Errorf("Expected default temperature to be 0.7, got %f", client.defaults.Temperature)
	}

	if client.defaults.MaxTokens != 1000 {
		t.Errorf("Expected default max tokens to be 1000, got %d", client.defaults.MaxTokens)
	}

	// Test with options
	client = NewClient(
		WithProvider(ProviderOpenAI),
		WithModel("gpt-4"),
		WithTemperature(0.5),
		WithMaxTokens(500),
	)

	if client.defaults.Provider != ProviderOpenAI {
		t.Errorf("Expected default provider to be OpenAI, got %s", client.defaults.Provider)
	}

	if client.defaults.Model != "gpt-4" {
		t.Errorf("Expected default model to be gpt-4, got %s", client.defaults.Model)
	}

	if client.defaults.Temperature != 0.5 {
		t.Errorf("Expected default temperature to be 0.5, got %f", client.defaults.Temperature)
	}

	if client.defaults.MaxTokens != 500 {
		t.Errorf("Expected default max tokens to be 500, got %d", client.defaults.MaxTokens)
	}
}

func TestRegisterProvider(t *testing.T) {
	client := NewClient()
	mockProvider := &MockProvider{}

	client.RegisterProvider(ProviderOpenAI, mockProvider)

	if _, ok := client.providers[ProviderOpenAI]; !ok {
		t.Errorf("Expected provider to be registered")
	}
}

func TestGetText(t *testing.T) {
	mockProvider := &MockProvider{
		GetTextFunc: func(ctx context.Context, config *Config) (string, error) {
			if config.Model != "test-model" {
				return "", errors.New("unexpected model")
			}
			return "Hello, world!", nil
		},
	}

	client := NewClient()
	client.RegisterProvider(ProviderOpenAI, mockProvider)

	// Test with no model
	_, err := client.GetText(context.Background(), WithProvider(ProviderOpenAI))
	if !errors.Is(err, ErrModelNotSpecified) {
		t.Errorf("Expected ErrModelNotSpecified, got %v", err)
	}

	// Test with unsupported provider
	_, err = client.GetText(context.Background(), WithProvider("unsupported"), WithModel("test-model"))
	if !errors.Is(err, ErrProviderNotSupported) {
		t.Errorf("Expected ErrProviderNotSupported, got %v", err)
	}

	// Test with valid config
	result, err := client.GetText(context.Background(),
		WithProvider(ProviderOpenAI),
		WithModel("test-model"),
	)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if result != "Hello, world!" {
		t.Errorf("Expected 'Hello, world!', got %s", result)
	}
}

func TestGetObject(t *testing.T) {
	type TestResponse struct {
		Message string `json:"message"`
	}

	mockProvider := &MockProvider{
		GetObjectFunc: func(ctx context.Context, config *Config, target interface{}) error {
			if config.Model != "test-model" {
				return errors.New("unexpected model")
			}

			// Cast target to the expected type
			resp, ok := target.(*TestResponse)
			if !ok {
				return errors.New("unexpected target type")
			}

			// Set the value
			resp.Message = "Hello, world!"
			return nil
		},
	}

	client := NewClient()
	client.RegisterProvider(ProviderOpenAI, mockProvider)

	// Test with no model
	var resp TestResponse
	err := client.GetObject(context.Background(), &resp, WithProvider(ProviderOpenAI))
	if !errors.Is(err, ErrModelNotSpecified) {
		t.Errorf("Expected ErrModelNotSpecified, got %v", err)
	}

	// Test with unsupported provider
	err = client.GetObject(context.Background(), &resp, WithProvider("unsupported"), WithModel("test-model"))
	if !errors.Is(err, ErrProviderNotSupported) {
		t.Errorf("Expected ErrProviderNotSupported, got %v", err)
	}

	// Test with valid config
	err = client.GetObject(context.Background(), &resp,
		WithProvider(ProviderOpenAI),
		WithModel("test-model"),
	)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if resp.Message != "Hello, world!" {
		t.Errorf("Expected 'Hello, world!', got %s", resp.Message)
	}
}

func TestGetToolCalls(t *testing.T) {
	mockTool := FunctionDefinition{
		Name:        "get_weather",
		Description: "Gets the weather for a location",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}`),
	}

	mockToolCall := ToolCall{
		ID:   "call_123",
		Type: "function",
		Tool: Tool{
			Name:      "get_weather",
			Arguments: json.RawMessage(`{"location":"San Francisco","unit":"celsius"}`),
		},
	}

	mockProvider := &MockProvider{
		GetToolCallsFunc: func(ctx context.Context, config *Config) ([]ToolCall, error) {
			if config.Model != "test-model" {
				return nil, errors.New("unexpected model")
			}

			// Check that tools are passed correctly
			if len(config.Tools) == 0 {
				return nil, errors.New("no tools specified")
			}

			if config.Tools[0].Name != "get_weather" {
				return nil, errors.New("unexpected tool")
			}

			return []ToolCall{mockToolCall}, nil
		},
	}

	client := NewClient()
	client.RegisterProvider(ProviderOpenAI, mockProvider)

	// Test with no model
	_, err := client.GetToolCalls(context.Background(),
		WithProvider(ProviderOpenAI),
		WithTools(mockTool),
	)
	if !errors.Is(err, ErrModelNotSpecified) {
		t.Errorf("Expected ErrModelNotSpecified, got %v", err)
	}

	// Test with no tools
	_, err = client.GetToolCalls(context.Background(),
		WithProvider(ProviderOpenAI),
		WithModel("test-model"),
	)
	if err == nil || err.Error() != "no tools specified" {
		t.Errorf("Expected 'no tools specified' error, got %v", err)
	}

	// Test with unsupported provider
	_, err = client.GetToolCalls(context.Background(),
		WithProvider("unsupported"),
		WithModel("test-model"),
		WithTools(mockTool),
	)
	if !errors.Is(err, ErrProviderNotSupported) {
		t.Errorf("Expected ErrProviderNotSupported, got %v", err)
	}

	// Test with valid config
	result, err := client.GetToolCalls(context.Background(),
		WithProvider(ProviderOpenAI),
		WithModel("test-model"),
		WithTools(mockTool),
	)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if len(result) != 1 {
		t.Errorf("Expected 1 tool call, got %d", len(result))
	}

	if result[0].ID != "call_123" {
		t.Errorf("Expected tool call ID 'call_123', got %s", result[0].ID)
	}

	if result[0].Tool.Name != "get_weather" {
		t.Errorf("Expected tool name 'get_weather', got %s", result[0].Tool.Name)
	}
}
