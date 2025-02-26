package ai

import (
	"context"
	"encoding/json"
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
	RoleTool      MessageRole = "tool" // Used for tool responses
)

// Message represents a single message in a conversation
type Message struct {
	Role       MessageRole `json:"role"`
	Content    string      `json:"content"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"` // For tool response messages
}

// ToolCall represents a call to a tool
type ToolCall struct {
	ID   string `json:"id"`
	Type string `json:"type"`
	Tool Tool   `json:"tool"`
}

// Tool represents a tool that can be called by the LLM
type Tool struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"` // Raw JSON for flexibility
}

// FunctionDefinition represents a function that can be called by the LLM
type FunctionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"` // Expected to be a JSON Schema
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

// ToolResultMessage creates a new tool result message
func ToolResultMessage(toolCallID string, content string) Message {
	return Message{
		Role:       RoleTool,
		Content:    content,
		ToolCallID: toolCallID,
	}
}

// LLMProvider defines the interface that all LLM providers must implement
type LLMProvider interface {
	GetText(ctx context.Context, config *Config) (string, error)
	GetObject(ctx context.Context, config *Config, target interface{}) error
	GetToolCalls(ctx context.Context, config *Config) ([]ToolCall, error)
}

// Config holds the configuration for a request to an LLM provider
type Config struct {
	Provider    Provider
	Model       string
	Messages    []Message
	MaxTokens   int
	Temperature float64
	Tools       []FunctionDefinition
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

// WithTools sets the tools for the request
func WithTools(tools ...FunctionDefinition) Option {
	return func(c *Config) {
		c.Tools = tools
	}
}
