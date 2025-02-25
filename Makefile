.PHONY: build test lint fmt clean

# Default target
all: fmt lint test build

# Build the application
build:
	go build ./...

# Run all tests
test:
	go test -v ./...

# Run specific test
test-one:
	@if [ -z "$(TEST)" ]; then echo "Usage: make test-one TEST=TestName"; exit 1; fi
	go test -v ./... -run $(TEST)

# Run linter
lint:
	@which golangci-lint > /dev/null || (echo "Installing golangci-lint..." && go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest)
	golangci-lint run

# Format code
fmt:
	go fmt ./...
	gofmt -s -w .

# Clean build artifacts
clean:
	go clean
	rm -f $(BINARY_NAME)

# Show help
help:
	@echo "Available targets:"
	@echo "  all        : Run fmt, lint, test, and build"
	@echo "  build      : Build the application"
	@echo "  test       : Run all tests"
	@echo "  test-one   : Run a specific test (Usage: make test-one TEST=TestName)"
	@echo "  lint       : Run linter"
	@echo "  fmt        : Format code"
	@echo "  clean      : Clean build artifacts"