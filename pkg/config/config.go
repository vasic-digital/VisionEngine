// Copyright 2026 Milos Vasic. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package config provides configuration for VisionEngine.
package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

var (
	// ErrInvalidConfig is returned when configuration is invalid.
	ErrInvalidConfig = errors.New("invalid configuration")
)

// Config holds the VisionEngine configuration.
type Config struct {
	// VisionProvider selects the primary vision provider: "openai", "anthropic", "gemini", "qwen", "astica", "auto".
	VisionProvider string `json:"vision_provider"`
	// OpenCVEnabled indicates if OpenCV features are enabled.
	OpenCVEnabled bool `json:"opencv_enabled"`
	// SSIMThreshold is the SSIM threshold for screen comparison.
	SSIMThreshold float64 `json:"ssim_threshold"`
	// MaxImageSize is the maximum image size in bytes.
	MaxImageSize int `json:"max_image_size"`

	// API keys for vision providers.
	OpenAIAPIKey    string `json:"-"`
	AnthropicAPIKey string `json:"-"`
	GoogleAPIKey    string `json:"-"`
	QwenAPIKey      string `json:"-"`
	DeepSeekAPIKey  string `json:"-"`
	GroqAPIKey      string `json:"-"`
	KimiAPIKey      string `json:"-"`
	StepfunAPIKey   string `json:"-"`
	AsticaAPIKey    string `json:"-"`

	// Provider-specific models.
	OpenAIModel    string `json:"openai_model,omitempty"`
	AnthropicModel string `json:"anthropic_model,omitempty"`
	GeminiModel    string `json:"gemini_model,omitempty"`
	QwenModel      string `json:"qwen_model,omitempty"`
	KimiModel      string `json:"kimi_model,omitempty"`
	StepGUIModel   string `json:"stepgui_model,omitempty"`
	AsticaModel    string `json:"astica_model,omitempty"`

	// Timeouts in seconds.
	TimeoutSecs int `json:"timeout_secs"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		VisionProvider: "auto",
		OpenCVEnabled:  true,
		SSIMThreshold:  0.95,
		MaxImageSize:   4096 * 4096 * 4, // ~64 MB
		TimeoutSecs:    60,
	}
}

// LoadFromEnv loads configuration from environment variables.
func LoadFromEnv() Config {
	cfg := DefaultConfig()

	if v := os.Getenv("HELIX_VISION_PROVIDER"); v != "" {
		cfg.VisionProvider = v
	}
	if v := os.Getenv("HELIX_VISION_OPENCV_ENABLED"); v != "" {
		cfg.OpenCVEnabled = strings.EqualFold(v, "true") || v == "1"
	}
	if v := os.Getenv("HELIX_VISION_SSIM_THRESHOLD"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.SSIMThreshold = f
		}
	}
	if v := os.Getenv("HELIX_VISION_MAX_IMAGE_SIZE"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.MaxImageSize = i
		}
	}

	cfg.OpenAIAPIKey = os.Getenv("OPENAI_API_KEY")
	if cfg.OpenAIAPIKey == "" {
		cfg.OpenAIAPIKey = os.Getenv("OPENROUTER_API_KEY")
	}
	cfg.AnthropicAPIKey = os.Getenv("ANTHROPIC_API_KEY")
	cfg.GoogleAPIKey = os.Getenv("GOOGLE_API_KEY")
	if cfg.GoogleAPIKey == "" {
		cfg.GoogleAPIKey = os.Getenv("GEMINI_API_KEY")
	}
	cfg.QwenAPIKey = os.Getenv("QWEN_API_KEY")
	cfg.DeepSeekAPIKey = os.Getenv("DEEPSEEK_API_KEY")
	cfg.GroqAPIKey = os.Getenv("GROQ_API_KEY")
	cfg.KimiAPIKey = os.Getenv("KIMI_API_KEY")
	if cfg.KimiAPIKey == "" {
		cfg.KimiAPIKey = os.Getenv("MOONSHOT_API_KEY")
	}
	cfg.StepfunAPIKey = os.Getenv("STEPFUN_API_KEY")
	cfg.AsticaAPIKey = os.Getenv("ASTICA_API_KEY")

	if v := os.Getenv("HELIX_VISION_OPENAI_MODEL"); v != "" {
		cfg.OpenAIModel = v
	}
	if v := os.Getenv("HELIX_VISION_ANTHROPIC_MODEL"); v != "" {
		cfg.AnthropicModel = v
	}
	if v := os.Getenv("HELIX_VISION_GEMINI_MODEL"); v != "" {
		cfg.GeminiModel = v
	}
	if v := os.Getenv("HELIX_VISION_QWEN_MODEL"); v != "" {
		cfg.QwenModel = v
	}
	if v := os.Getenv("HELIX_VISION_KIMI_MODEL"); v != "" {
		cfg.KimiModel = v
	}
	if v := os.Getenv("HELIX_VISION_STEPGUI_MODEL"); v != "" {
		cfg.StepGUIModel = v
	}
	if v := os.Getenv("HELIX_VISION_ASTICA_MODEL"); v != "" {
		cfg.AsticaModel = v
	}
	if v := os.Getenv("HELIX_VISION_TIMEOUT"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.TimeoutSecs = i
		}
	}

	return cfg
}

// Validate checks that the configuration is valid.
func (c Config) Validate() error {
	validProviders := map[string]bool{
		"auto": true, "openai": true, "anthropic": true, "gemini": true, "qwen": true,
		"kimi": true, "stepgui": true, "astica": true,
	}
	if !validProviders[c.VisionProvider] {
		return fmt.Errorf("%w: unknown vision provider %q", ErrInvalidConfig, c.VisionProvider)
	}
	if c.SSIMThreshold < 0 || c.SSIMThreshold > 1 {
		return fmt.Errorf("%w: SSIM threshold must be between 0 and 1, got %f", ErrInvalidConfig, c.SSIMThreshold)
	}
	if c.MaxImageSize <= 0 {
		return fmt.Errorf("%w: max image size must be positive, got %d", ErrInvalidConfig, c.MaxImageSize)
	}
	if c.TimeoutSecs <= 0 {
		return fmt.Errorf("%w: timeout must be positive, got %d", ErrInvalidConfig, c.TimeoutSecs)
	}

	// If a specific provider is selected, check API key
	switch c.VisionProvider {
	case "openai":
		if c.OpenAIAPIKey == "" {
			return fmt.Errorf("%w: OPENAI_API_KEY required for openai provider", ErrInvalidConfig)
		}
	case "anthropic":
		if c.AnthropicAPIKey == "" {
			return fmt.Errorf("%w: ANTHROPIC_API_KEY required for anthropic provider", ErrInvalidConfig)
		}
	case "gemini":
		if c.GoogleAPIKey == "" {
			return fmt.Errorf("%w: GOOGLE_API_KEY required for gemini provider", ErrInvalidConfig)
		}
	case "qwen":
		if c.QwenAPIKey == "" {
			return fmt.Errorf("%w: QWEN_API_KEY required for qwen provider", ErrInvalidConfig)
		}
	case "kimi":
		if c.KimiAPIKey == "" {
			return fmt.Errorf("%w: KIMI_API_KEY or MOONSHOT_API_KEY required for kimi provider", ErrInvalidConfig)
		}
	case "stepgui":
		if c.StepfunAPIKey == "" {
			return fmt.Errorf("%w: STEPFUN_API_KEY required for stepgui provider", ErrInvalidConfig)
		}
	case "astica":
		if c.AsticaAPIKey == "" {
			return fmt.Errorf("%w: ASTICA_API_KEY required for astica provider", ErrInvalidConfig)
		}
	}

	return nil
}

// AvailableProviders returns a list of provider names that have API keys configured.
func (c Config) AvailableProviders() []string {
	var providers []string
	if c.OpenAIAPIKey != "" {
		providers = append(providers, "openai")
	}
	if c.AnthropicAPIKey != "" {
		providers = append(providers, "anthropic")
	}
	if c.GoogleAPIKey != "" {
		providers = append(providers, "gemini")
	}
	if c.QwenAPIKey != "" {
		providers = append(providers, "qwen")
	}
	if c.DeepSeekAPIKey != "" {
		providers = append(providers, "deepseek")
	}
	if c.GroqAPIKey != "" {
		providers = append(providers, "groq")
	}
	if c.KimiAPIKey != "" {
		providers = append(providers, "kimi")
	}
	if c.StepfunAPIKey != "" {
		providers = append(providers, "stepgui")
	}
	if c.AsticaAPIKey != "" {
		providers = append(providers, "astica")
	}
	return providers
}
