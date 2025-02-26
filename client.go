package ai

import (
	"context"
	"errors"
	"fmt"
)

var (
	// ErrProviderNotSupported is returned when an unsupported provider is specified
	ErrProviderNotSupported = errors.New("provider not supported")

	// ErrModelNotSpecified is returned when no model is specified
	ErrModelNotSpecified = errors.New("model not specified")
)

// Client is the main entry point for the go-ai-sdk
type Client struct {
	providers map[Provider]LLMProvider
	defaults  *Config
}

// NewClient creates a new client with default configuration
func NewClient(options ...Option) *Client {
	defaults := &Config{
		Temperature: 0.7,
		MaxTokens:   1000,
	}

	for _, opt := range options {
		opt(defaults)
	}

	return &Client{
		providers: make(map[Provider]LLMProvider),
		defaults:  defaults,
	}
}

// RegisterProvider registers a provider with the client
func (c *Client) RegisterProvider(provider Provider, impl LLMProvider) {
	c.providers[provider] = impl
}

// mergeConfig creates a new config by merging the defaults with the provided options
func (c *Client) mergeConfig(options ...Option) *Config {
	// Start with the defaults
	config := &Config{
		Provider:    c.defaults.Provider,
		Model:       c.defaults.Model,
		MaxTokens:   c.defaults.MaxTokens,
		Temperature: c.defaults.Temperature,
	}

	// Copy messages (if any)
	if len(c.defaults.Messages) > 0 {
		config.Messages = make([]Message, len(c.defaults.Messages))
		copy(config.Messages, c.defaults.Messages)
	}

	// Apply the options
	for _, opt := range options {
		opt(config)
	}

	return config
}

// GetText gets a text response from the specified provider
func (c *Client) GetText(ctx context.Context, options ...Option) (string, error) {
	config := c.mergeConfig(options...)

	if config.Model == "" {
		return "", ErrModelNotSpecified
	}

	provider, ok := c.providers[config.Provider]
	if !ok {
		return "", fmt.Errorf("%w: %s", ErrProviderNotSupported, config.Provider)
	}

	return provider.GetText(ctx, config)
}

// GetObject gets a structured response from the specified provider
func (c *Client) GetObject(ctx context.Context, target interface{}, options ...Option) error {
	config := c.mergeConfig(options...)

	if config.Model == "" {
		return ErrModelNotSpecified
	}

	provider, ok := c.providers[config.Provider]
	if !ok {
		return fmt.Errorf("%w: %s", ErrProviderNotSupported, config.Provider)
	}

	return provider.GetObject(ctx, config, target)
}
