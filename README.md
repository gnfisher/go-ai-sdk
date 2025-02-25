# Go AI SDK

A Go package for simplified interactions with LLM services like OpenAI, Anthropic, and others.

## Features

- Simple, consistent interface for multiple LLM providers
- Text completion with `getText` API
- Structured responses with `getObject` API
- Tool/function calling support
- Configurable parameters (temperature, log probs, etc.)

## Installation

```bash
go get github.com/gnfisher/go-ai-sdk
```

## Basic Usage

```go
package main

import (
    "context"
    "fmt"
    
    ai "github.com/gnfisher/go-ai-sdk"
)

func main() {
    client := ai.NewClient()
    
    // Get a text response
    response, err := client.GetText(
        context.Background(),
        ai.WithProvider(ai.ProviderOpenAI),
        ai.WithModel("gpt-4"),
        ai.WithMessages(
            ai.SystemMessage("You are a helpful assistant."),
            ai.UserMessage("Tell me a joke."),
        ),
    )
    
    if err != nil {
        panic(err)
    }
    
    fmt.Println(response)
}
```

## License

MIT