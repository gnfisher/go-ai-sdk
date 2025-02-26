package main

import (
	"fmt"
	"os"

	openaiexample "github.com/gnfisher/go-ai-sdk/examples/openai"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "openai":
		openaiexample.OpenAIExample()
	default:
		fmt.Printf("Unknown example: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("Usage: go run examples/main.go [example]")
	fmt.Println("Available examples:")
	fmt.Println("  openai    - Run the OpenAI provider example")
}
