package ai

import (
	"context"
)

// Provider represents the LLM service provider
type Provider string

const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
)

// MessageRole represents the role of a message in a conversation
type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
)

// Message represents a single message in a conversation
type Message struct {
	Role    MessageRole `json:"role"`
	Content string      `json:"content"`
}

// SystemMessage creates a new system message
func SystemMessage(content string) Message {
	return Message{
		Role:    RoleSystem,
		Content: content,
	}
}

// UserMessage creates a new user message
func UserMessage(content string) Message {
	return Message{
		Role:    RoleUser,
		Content: content,
	}
}

// AssistantMessage creates a new assistant message
func AssistantMessage(content string) Message {
	return Message{
		Role:    RoleAssistant,
		Content: content,
	}
}

// LLMProvider defines the interface that all LLM providers must implement
type LLMProvider interface {
	GetText(ctx context.Context, config *Config) (string, error)
	GetObject(ctx context.Context, config *Config, target interface{}) error
}

// Config holds the configuration for a request to an LLM provider
type Config struct {
	Provider    Provider
	Model       string
	Messages    []Message
	MaxTokens   int
	Temperature float64
}

// Option is a function that modifies a Config
type Option func(*Config)

// WithProvider sets the provider for the request
func WithProvider(provider Provider) Option {
	return func(c *Config) {
		c.Provider = provider
	}
}

// WithModel sets the model for the request
func WithModel(model string) Option {
	return func(c *Config) {
		c.Model = model
	}
}

// WithMessages sets the messages for the request
func WithMessages(messages ...Message) Option {
	return func(c *Config) {
		c.Messages = messages
	}
}

// WithMaxTokens sets the maximum number of tokens to generate
func WithMaxTokens(maxTokens int) Option {
	return func(c *Config) {
		c.MaxTokens = maxTokens
	}
}

// WithTemperature sets the temperature for the request
func WithTemperature(temperature float64) Option {
	return func(c *Config) {
		c.Temperature = temperature
	}
}