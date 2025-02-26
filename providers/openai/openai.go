package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gnfisher/go-ai-sdk"
)

const (
	defaultAPIURL = "https://api.openai.com/v1/chat/completions"
)

var (
	ErrEmptyAPIKey     = errors.New("OpenAI API key is empty")
	ErrInvalidResponse = errors.New("invalid response from OpenAI API")
)

// Provider implements the ai.LLMProvider interface for OpenAI
type Provider struct {
	apiKey string
	apiURL string
	client *http.Client
}

// Option is a function that configures the OpenAI provider
type Option func(*Provider)

// WithAPIKey sets the API key for the OpenAI provider
func WithAPIKey(apiKey string) Option {
	return func(p *Provider) {
		p.apiKey = apiKey
	}
}

// WithAPIURL sets the API URL for the OpenAI provider
func WithAPIURL(apiURL string) Option {
	return func(p *Provider) {
		p.apiURL = apiURL
	}
}

// WithHTTPClient sets the HTTP client for the OpenAI provider
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.client = client
	}
}

// New creates a new OpenAI provider
func New(options ...Option) *Provider {
	provider := &Provider{
		apiURL: defaultAPIURL,
		client: http.DefaultClient,
	}

	for _, opt := range options {
		opt(provider)
	}

	return provider
}

// Message represents an OpenAI chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Request represents a request to the OpenAI API
type Request struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

// Response represents a response from the OpenAI API
type Response struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int      `json:"created"`
	Choices []Choice `json:"choices"`
	Error   *Error   `json:"error,omitempty"`
}

// Choice represents a choice in the OpenAI API response
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Error represents an error in the OpenAI API response
type Error struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    string `json:"code"`
}

// convertMessages converts ai.Message to openai.Message
func convertMessages(messages []ai.Message) []Message {
	result := make([]Message, len(messages))
	for i, msg := range messages {
		result[i] = Message{
			Role:    string(msg.Role),
			Content: msg.Content,
		}
	}
	return result
}

// GetText gets a text response from the OpenAI API
func (p *Provider) GetText(ctx context.Context, config *ai.Config) (string, error) {
	if p.apiKey == "" {
		return "", ErrEmptyAPIKey
	}

	openaiMessages := convertMessages(config.Messages)

	reqBody := Request{
		Model:       config.Model,
		Messages:    openaiMessages,
		Temperature: config.Temperature,
		MaxTokens:   config.MaxTokens,
	}

	reqJSON, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.apiURL, bytes.NewBuffer(reqJSON))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp Response
		if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error != nil {
			return "", fmt.Errorf("OpenAI API error: %s", errResp.Error.Message)
		}
		return "", fmt.Errorf("OpenAI API returned status code %d: %s", resp.StatusCode, body)
	}

	var openAIResp Response
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(openAIResp.Choices) == 0 || openAIResp.Choices[0].Message.Content == "" {
		return "", ErrInvalidResponse
	}

	return openAIResp.Choices[0].Message.Content, nil
}

// GetObject gets a structured response from the OpenAI API
func (p *Provider) GetObject(ctx context.Context, config *ai.Config, target interface{}) error {
	if p.apiKey == "" {
		return ErrEmptyAPIKey
	}

	// Extract the type information from the target
	targetType := fmt.Sprintf("%T", target)

	// Create a system message instructing the model to return JSON
	systemMsg := ai.Message{
		Role:    ai.RoleSystem,
		Content: fmt.Sprintf("You are a helpful assistant that responds with JSON matching the %s type. Your response should be valid JSON and nothing else.", targetType),
	}

	// Prepare messages, adding the system message if not already present
	messages := config.Messages
	hasSystemMsg := false
	for _, msg := range messages {
		if msg.Role == ai.RoleSystem {
			hasSystemMsg = true
			break
		}
	}

	if !hasSystemMsg {
		messages = append([]ai.Message{systemMsg}, messages...)
	}

	// Get the text response
	textResp, err := p.GetText(ctx, &ai.Config{
		Provider:    config.Provider,
		Model:       config.Model,
		Messages:    messages,
		MaxTokens:   config.MaxTokens,
		Temperature: config.Temperature,
	})
	if err != nil {
		return err
	}

	// Clean the response to ensure it's valid JSON
	jsonStr := strings.TrimSpace(textResp)

	// If response starts with ``` (markdown code block), clean it up
	if strings.HasPrefix(jsonStr, "```json") {
		jsonStr = strings.TrimPrefix(jsonStr, "```json")
		if idx := strings.LastIndex(jsonStr, "```"); idx != -1 {
			jsonStr = jsonStr[:idx]
		}
	} else if strings.HasPrefix(jsonStr, "```") {
		jsonStr = strings.TrimPrefix(jsonStr, "```")
		if idx := strings.LastIndex(jsonStr, "```"); idx != -1 {
			jsonStr = jsonStr[:idx]
		}
	}

	jsonStr = strings.TrimSpace(jsonStr)

	// Unmarshal the JSON into the target
	if err := json.Unmarshal([]byte(jsonStr), target); err != nil {
		return fmt.Errorf("failed to unmarshal JSON response: %w: %s", err, jsonStr)
	}

	return nil
}
