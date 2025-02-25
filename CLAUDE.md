# Claude Guidelines for Go-AI-SDK

## Build & Test Commands
- Build: `make build` or `go build ./...`
- Test all: `make test` or `go test ./...`
- Test single: `make test-one TEST=TestName` or `go test ./path/to/package -run TestName`
- Lint: `make lint` or `golangci-lint run`
- Format: `make fmt` or `go fmt ./...`

## Development Process
- Always check and update PLANNING.md to track progress and plans
- Check off completed tasks in PLANNING.md as we go
- Add new steps/tasks to PLANNING.md when necessary
- Work in small, tight iterations with feedback for each step
- Each iteration should result in something committable with passing tests
- Prioritize concrete progress that can be validated

## Code Style
- Follow standard Go conventions from Effective Go
- Errors: use descriptive error messages, consider custom error types
- Function naming: use CamelCase, no underscores
- Imports: group standard library, third-party, then local imports
- Comments: document public functions with meaningful GoDoc
- Tests: table-driven tests preferred
- Error handling: check errors immediately, don't use panic

## Project Structure
- Core interfaces in root package
- Provider-specific implementations in subpackages
- Tests alongside implementation files