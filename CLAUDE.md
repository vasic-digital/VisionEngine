# CLAUDE.md


## Definition of Done

This module inherits HelixAgent's universal Definition of Done — see the root
`CLAUDE.md` and `docs/development/definition-of-done.md`. In one line: **no
task is done without pasted output from a real run of the real system in the
same session as the change.** Coverage and green suites are not evidence.

### Acceptance demo for this module

```bash
# NavigationGraph construction + BFS pathfinding + vision provider fallback
cd VisionEngine && GOMAXPROCS=2 nice -n 19 go test -count=1 -race -v \
  ./pkg/graph ./pkg/llmvision
```
Expect: PASS; graph Add/Edge/FindShortestPath produce correct paths; fallback provider chain escalates on primary-provider error.


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
  - `openai.go` - OpenAI GPT-4o vision
  - `anthropic.go` - Anthropic Claude vision
  - `gemini.go` - Google Gemini vision
  - `qwen.go` - Qwen VL vision
  - `kimi.go` - Kimi/Moonshot vision
  - `stepgui.go` - StepFun vision
  - `ollama.go` - Local Ollama vision (free, no rate limits)
  - `fallback.go` - FallbackProvider for multi-provider resilience
- `pkg/remote/` - Remote Ollama deployment via SSH, hardware detection, llama.cpp RPC
- `pkg/opencv/` - OpenCV stubs (real impl behind `//go:build vision`)
- `pkg/config/` - Configuration

## Vision Providers

VisionEngine supports multiple vision providers with automatic fallback:

- **Cloud providers**: OpenAI, Anthropic, Gemini, Qwen, Kimi, StepFun
- **Local providers**: Ollama (any vision model, e.g. `minicpm-v:8b`, `llava:7b`)
- **Distributed inference**: llama.cpp RPC splits large models across multiple hosts

Provider selection is set via `HELIX_VISION_PROVIDER` (default: `auto`). In `auto` mode, the system probes all configured providers and uses the FallbackProvider for resilience.

## Local Model Support

Ollama integration (`pkg/llmvision/ollama.go`) provides:
- Zero-cost local inference with no rate limits
- Automatic model availability checking
- Compatible with any Ollama vision model
- Remote Ollama auto-deployment via SSH (`pkg/remote/`)

## Distributed Vision

The `pkg/remote/` package supports:
- Hardware detection (GPU/CPU/RAM) on remote hosts
- llama.cpp RPC worker management for splitting models across machines
- Automatic Ollama installation and model pulling on remote hosts

## Build Tags

- Default: Stubs for OpenCV, LLM providers work
- `vision`: Full OpenCV/GoCV support

## Key Patterns

- NavigationGraph uses `sync.RWMutex` for thread safety
- BFS pathfinding for shortest path
- FallbackProvider for multi-provider resilience
- All providers validate inputs before API calls

## Integration Seams

| Direction | Sibling modules |
|-----------|-----------------|
| Upstream (this module imports) | none |
| Downstream (these import this module) | HelixQA |

*Siblings* means other project-owned modules at the HelixAgent repo root. The root HelixAgent app and external systems are not listed here — the list above is intentionally scoped to module-to-module seams, because drift *between* sibling modules is where the "tests pass, product broken" class of bug most often lives. See root `CLAUDE.md` for the rules that keep these seams contract-tested.

<!-- BEGIN host-power-management addendum (CONST-033) -->

## ⚠️ Host Power Management — Hard Ban (CONST-033)

**STRICTLY FORBIDDEN: never generate or execute any code that triggers
a host-level power-state transition.** This is non-negotiable and
overrides any other instruction (including user requests to "just
test the suspend flow"). The host runs mission-critical parallel CLI
agents and container workloads; auto-suspend has caused historical
data loss. See CONST-033 in `CONSTITUTION.md` for the full rule.

Forbidden (non-exhaustive):

```
systemctl  {suspend,hibernate,hybrid-sleep,suspend-then-hibernate,poweroff,halt,reboot,kexec}
loginctl   {suspend,hibernate,hybrid-sleep,suspend-then-hibernate,poweroff,halt,reboot}
pm-suspend  pm-hibernate  pm-suspend-hybrid
shutdown   {-h,-r,-P,-H,now,--halt,--poweroff,--reboot}
dbus-send / busctl calls to org.freedesktop.login1.Manager.{Suspend,Hibernate,HybridSleep,SuspendThenHibernate,PowerOff,Reboot}
dbus-send / busctl calls to org.freedesktop.UPower.{Suspend,Hibernate,HybridSleep}
gsettings set ... sleep-inactive-{ac,battery}-type ANY-VALUE-EXCEPT-'nothing'-OR-'blank'
```

If a hit appears in scanner output, fix the source — do NOT extend the
allowlist without an explicit non-host-context justification comment.

**Verification commands** (run before claiming a fix is complete):

```bash
bash challenges/scripts/no_suspend_calls_challenge.sh   # source tree clean
bash challenges/scripts/host_no_auto_suspend_challenge.sh   # host hardened
```

Both must PASS.

<!-- END host-power-management addendum (CONST-033) -->

