# VisionEngine Architecture

## Module Structure

```
VisionEngine/
├── pkg/
│   ├── analyzer/    # Core interfaces and types
│   ├── graph/       # NavigationGraph (imported by HelixQA)
│   ├── llmvision/   # LLM Vision API adapters (Gemini, Anthropic, OpenAI, Qwen, Ollama)
│   ├── remote/      # Remote Ollama deployment via SSH (auto-install, model pull, lifecycle)
│   ├── opencv/      # OpenCV stubs (real impl behind build tag)
│   └── config/      # Configuration
├── go.mod
├── Makefile
└── Upstreams/       # Remote sync scripts
```

## Two-Layer Analysis Pipeline

```mermaid
graph TD
    A[Screenshot] --> B[Layer 1: GoCV - Mechanical]
    A --> C[Layer 2: LLM Vision - Intelligent]
    B --> D[Element Detection]
    B --> E[SSIM Comparison]
    B --> F[Color Analysis]
    C --> G[Screen Identification]
    C --> H[UI Comprehension]
    C --> I[Issue Detection]
    D --> J[Combined Analysis]
    G --> J
```

## NavigationGraph

```mermaid
classDiagram
    class NavigationGraph {
        <<interface>>
        +AddScreen(ScreenIdentity) string
        +AddTransition(from, to, Action)
        +CurrentScreen() string
        +SetCurrent(screenID)
        +PathTo(targetID) []Transition
        +UnvisitedScreens() []string
        +Coverage() float64
        +Export() GraphSnapshot
    }
    class navGraph {
        -screens map[string]*ScreenNode
        -transitions []Transition
        -adjacency map[string][]Transition
        -current string
        -mu sync.RWMutex
    }
    NavigationGraph <|.. navGraph
```

## Thread Safety

- NavigationGraph uses `sync.RWMutex` for all operations
- FallbackProvider uses `sync.RWMutex` for provider list access
- All concurrent operations are race-detector safe

## Build Tags

- Default build: No OpenCV, stubs return errors, LLM providers work
- `vision` tag: Full OpenCV support via GoCV
