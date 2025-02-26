package anthropic

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
	defaultAPIURL    = "https://api.anthropic.com/v1/messages"
	anthropicVersion = "2023-06-01"
)

var (
	ErrEmptyAPIKey     = errors.New("Anthropic API key is empty")
	ErrInvalidResponse = errors.New("invalid response from Anthropic API")
)

// Provider implements the ai.LLMProvider interface for Anthropic
type Provider struct {
	apiKey string
	apiURL string
	client *http.Client
}

// Option is a function that configures the Anthropic provider
type Option func(*Provider)

// WithAPIKey sets the API key for the Anthropic provider
func WithAPIKey(apiKey string) Option {
	return func(p *Provider) {
		p.apiKey = apiKey
	}
}

// WithAPIURL sets the API URL for the Anthropic provider
func WithAPIURL(apiURL string) Option {
	return func(p *Provider) {
		p.apiURL = apiURL
	}
}

// WithHTTPClient sets the HTTP client for the Anthropic provider
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.client = client
	}
}

// New creates a new Anthropic provider
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

// Message represents an Anthropic message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Request represents a request to the Anthropic API
type Request struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	System      string    `json:"system,omitempty"`
}

// Content represents content in the Anthropic API response
type Content struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Response represents a response from the Anthropic API
type Response struct {
	ID         string    `json:"id"`
	Type       string    `json:"type"`
	Role       string    `json:"role"`
	Content    []Content `json:"content"`
	Model      string    `json:"model"`
	StopReason string    `json:"stop_reason"`
	Error      *Error    `json:"error,omitempty"`
}

// Error represents an error in the Anthropic API response
type Error struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// convertMessages converts ai.Message to anthropic.Message and extracts system message
func convertMessages(messages []ai.Message) ([]Message, string) {
	var systemMessage string
	var result []Message

	for _, msg := range messages {
		if msg.Role == ai.RoleSystem {
			systemMessage = msg.Content
			continue
		}

		// Map ai.MessageRole to Anthropic roles
		role := string(msg.Role)
		if msg.Role == ai.RoleAssistant {
			role = "assistant"
		} else if msg.Role == ai.RoleUser {
			role = "user"
		}

		result = append(result, Message{
			Role:    role,
			Content: msg.Content,
		})
	}

	return result, systemMessage
}

// GetText gets a text response from the Anthropic API
func (p *Provider) GetText(ctx context.Context, config *ai.Config) (string, error) {
	if p.apiKey == "" {
		return "", ErrEmptyAPIKey
	}

	anthropicMessages, systemMessage := convertMessages(config.Messages)

	reqBody := Request{
		Model:       config.Model,
		Messages:    anthropicMessages,
		Temperature: config.Temperature,
		MaxTokens:   config.MaxTokens,
		System:      systemMessage,
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
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)

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
			return "", fmt.Errorf("Anthropic API error: %s", errResp.Error.Message)
		}
		return "", fmt.Errorf("Anthropic API returned status code %d: %s", resp.StatusCode, body)
	}

	var anthropicResp Response
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(anthropicResp.Content) == 0 || anthropicResp.Content[0].Text == "" {
		return "", ErrInvalidResponse
	}

	return anthropicResp.Content[0].Text, nil
}

// GetObject gets a structured response from the Anthropic API
func (p *Provider) GetObject(ctx context.Context, config *ai.Config, target interface{}) error {
	if p.apiKey == "" {
		return ErrEmptyAPIKey
	}

	// Extract the type information from the target
	targetType := fmt.Sprintf("%T", target)

	// Create a system message instructing the model to return JSON
	systemMsg := fmt.Sprintf("You are a helpful assistant that responds with JSON matching the %s type. Your response should be valid JSON and nothing else.", targetType)

	// Prepare messages
	messages := config.Messages
	anthropicMessages, existingSystemMsg := convertMessages(messages)

	// If there's already a system message, append our JSON instruction
	if existingSystemMsg != "" {
		systemMsg = existingSystemMsg + "\n\n" + systemMsg
	}

	// Get the text response
	reqBody := Request{
		Model:       config.Model,
		Messages:    anthropicMessages,
		Temperature: config.Temperature,
		MaxTokens:   config.MaxTokens,
		System:      systemMsg,
	}

	reqJSON, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.apiURL, bytes.NewBuffer(reqJSON))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp Response
		if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error != nil {
			return fmt.Errorf("Anthropic API error: %s", errResp.Error.Message)
		}
		return fmt.Errorf("Anthropic API returned status code %d: %s", resp.StatusCode, body)
	}

	var anthropicResp Response
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(anthropicResp.Content) == 0 || anthropicResp.Content[0].Text == "" {
		return ErrInvalidResponse
	}

	// Clean the response to ensure it's valid JSON
	jsonStr := strings.TrimSpace(anthropicResp.Content[0].Text)

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
