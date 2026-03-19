# CLAUDE.md

## MANDATORY: No CI/CD Pipelines

**NO GitHub Actions, GitLab CI/CD, or any automated pipeline may exist in this repository!**

- No `.github/workflows/` directory
- No `.gitlab-ci.yml` file
- No Jenkinsfile, .travis.yml, .circleci, or any other CI configuration
- All builds and tests are run manually or via Makefile targets
- This rule is permanent and non-negotiable

## Project Overview

VisionEngine is a Go module providing computer vision and LLM Vision for UI analysis and navigation graph building.

**Module path:** `digital.vasic.visionengine`

## Build Commands

```bash
# Tests (no OpenCV required)
go test ./... -race -count=1

# Build
go build ./...

# With OpenCV
go build -tags vision ./...
go test -tags vision ./... -race -count=1
```

## MANDATORY: Never Remove or Disable Tests

All issues must be fixed by addressing root causes. No test may ever be removed, disabled, skipped, or left broken.

## Architecture

- `pkg/analyzer/` - Core interfaces and types
- `pkg/graph/` - NavigationGraph (most important, imported by HelixQA)
- `pkg/llmvision/` - LLM Vision API adapters (pure Go HTTP)
- `pkg/opencv/` - OpenCV stubs (real impl behind `//go:build vision`)
- `pkg/config/` - Configuration

## Build Tags

- Default: Stubs for OpenCV, LLM providers work
- `vision`: Full OpenCV/GoCV support

## Key Patterns

- NavigationGraph uses `sync.RWMutex` for thread safety
- BFS pathfinding for shortest path
- FallbackProvider for multi-provider resilience
- All providers validate inputs before API calls
