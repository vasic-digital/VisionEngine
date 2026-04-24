# VisionEngine Makefile
# Computer vision and LLM Vision for UI analysis and navigation

.PHONY: help build test test-race test-vision vet lint clean fmt tidy

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all packages (no OpenCV required)
	go build ./...

build-vision: ## Build with OpenCV support
	go build -tags vision ./...

test: ## Run all tests (no OpenCV required)
	go test ./... -count=1

test-race: ## Run all tests with race detector
	go test ./... -race -count=1

test-vision: ## Run tests including OpenCV tests
	go test -tags vision ./... -race -count=1

test-verbose: ## Run tests with verbose output
	go test ./... -v -race -count=1

test-coverage: ## Run tests with coverage report
	go test ./... -race -count=1 -coverprofile=coverage.out
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

vet: ## Run go vet
	go vet ./...

lint: ## Run static analysis
	@if command -v golangci-lint &> /dev/null; then \
		golangci-lint run; \
	else \
		echo "golangci-lint not installed, running go vet instead"; \
		go vet ./...; \
	fi

fmt: ## Format all Go files
	gofmt -s -w .

tidy: ## Run go mod tidy
	go mod tidy

clean: ## Remove build artifacts
	rm -f coverage.out coverage.html
	go clean -cache -testcache

check: vet test-race ## Run all checks (vet + tests with race detector)

all: tidy fmt vet test-race ## Run full CI pipeline

# Definition of Done gates — portable drop-in from HelixAgent
.PHONY: no-silent-skips no-silent-skips-warn demo-all demo-all-warn demo-one ci-validate-all

no-silent-skips:
	@bash scripts/no-silent-skips.sh

no-silent-skips-warn:
	@NO_SILENT_SKIPS_WARN_ONLY=1 bash scripts/no-silent-skips.sh

demo-all:
	@bash scripts/demo-all.sh

demo-all-warn:
	@DEMO_ALL_WARN_ONLY=1 DEMO_ALLOW_TODO=1 bash scripts/demo-all.sh

demo-one:
	@DEMO_MODULES="$(MOD)" bash scripts/demo-all.sh

ci-validate-all: no-silent-skips-warn demo-all-warn
	@echo "ci-validate-all: all gates executed"
