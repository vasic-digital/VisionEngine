# AGENTS.md — VisionEngine

## MANDATORY: No CI/CD Pipelines

**NO GitHub Actions, GitLab CI/CD, or any automated pipeline may exist in this repository!**

- No `.github/workflows/` directory
- No `.gitlab-ci.yml` file
- No Jenkinsfile, .travis.yml, .circleci, or any other CI configuration
- All builds and tests are run manually or via Makefile targets
- This rule is permanent and non-negotiable

## For AI Agents Working on This Codebase

### Module Purpose
VisionEngine provides computer vision (GoCV) and LLM Vision capabilities for UI analysis, navigation graph building, and video frame extraction.

### Key Packages
- `pkg/analyzer` — Analyzer interface, ScreenAnalysis, UIElement, VisualIssue types
- `pkg/graph` — NavigationGraph with BFS pathfinding, DOT/JSON/Mermaid export
- `pkg/llmvision` — VisionProvider interface, 4 LLM adapters (OpenAI, Anthropic, Gemini, Qwen)
- `pkg/opencv` — GoCV stubs (real impl behind `//go:build vision` tag)
- `pkg/config` — Configuration via environment variables

### Build Tags
OpenCV code is gated behind `//go:build vision`. Default `go test ./...` works without OpenCV.

### Testing
```bash
go test ./... -race -count=1          # Without OpenCV (default)
go test -tags vision ./... -race      # With OpenCV (requires OpenCV 4.x)
```

### Key Interfaces
- `analyzer.Analyzer` — screen analysis (6 methods)
- `graph.NavigationGraph` — directed graph (10 methods, thread-safe)
- `llmvision.VisionProvider` — LLM vision API (4 methods)
