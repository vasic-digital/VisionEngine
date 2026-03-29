// Copyright 2026 Milos Vasic. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package llmvision

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	ollamaDefaultBaseURL = "http://localhost:11434"
	ollamaDefaultModel   = "llava:7b"
	ollamaMaxImageSize   = 20 * 1024 * 1024 // 20 MB
	ollamaDefaultTimeout = 120 * time.Second
)

// OllamaProvider implements VisionProvider for Ollama local inference.
type OllamaProvider struct {
	config     ProviderConfig
	httpClient *http.Client
}

// NewOllamaProvider creates a new Ollama vision provider.
func NewOllamaProvider(config ProviderConfig) (*OllamaProvider, error) {
	if config.BaseURL == "" {
		config.BaseURL = ollamaDefaultBaseURL
	}
	if config.Model == "" {
		config.Model = ollamaDefaultModel
	}
	if config.MaxImageSize == 0 {
		config.MaxImageSize = ollamaMaxImageSize
	}
	timeout := ollamaDefaultTimeout
	if config.TimeoutSecs > 0 {
		timeout = time.Duration(config.TimeoutSecs) * time.Second
	}
	return &OllamaProvider{
		config: config,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// Name returns "ollama".
func (p *OllamaProvider) Name() string {
	return "ollama"
}

// SupportsVision returns true.
func (p *OllamaProvider) SupportsVision() bool {
	return true
}

// MaxImageSize returns the max image size.
func (p *OllamaProvider) MaxImageSize() int {
	return p.config.MaxImageSize
}

// AnalyzeImage sends an image to Ollama for analysis.
func (p *OllamaProvider) AnalyzeImage(ctx context.Context, image []byte, prompt string) (string, error) {
	if err := validateImage(image, p.config.MaxImageSize); err != nil {
		return "", err
	}
	if err := validatePrompt(prompt); err != nil {
		return "", err
	}

	encoded := base64.StdEncoding.EncodeToString(image)
	body := map[string]any{
		"model":  p.config.Model,
		"stream": false,
		"messages": []map[string]any{
			{
				"role":    "user",
				"content": prompt,
				"images":  []string{encoded},
			},
		},
	}

	return p.sendRequest(ctx, body)
}

// CompareImages sends two images to Ollama for comparison.
func (p *OllamaProvider) CompareImages(ctx context.Context, img1, img2 []byte, prompt string) (string, error) {
	if err := validateImage(img1, p.config.MaxImageSize); err != nil {
		return "", err
	}
	if err := validateImage(img2, p.config.MaxImageSize); err != nil {
		return "", err
	}
	if err := validatePrompt(prompt); err != nil {
		return "", err
	}

	enc1 := base64.StdEncoding.EncodeToString(img1)
	enc2 := base64.StdEncoding.EncodeToString(img2)
	body := map[string]any{
		"model":  p.config.Model,
		"stream": false,
		"messages": []map[string]any{
			{
				"role":    "user",
				"content": prompt,
				"images":  []string{enc1, enc2},
			},
		},
	}

	return p.sendRequest(ctx, body)
}

func (p *OllamaProvider) sendRequest(ctx context.Context, body map[string]any) (string, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.config.BaseURL+"/api/chat", bytes.NewReader(jsonBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrProviderUnavailable, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		return "", ErrRateLimited
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%w: status %d: %s", ErrProviderUnavailable, resp.StatusCode, string(respBody))
	}

	var result struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("%w: %v", ErrInvalidResponse, err)
	}
	if result.Message.Content == "" {
		return "", fmt.Errorf("%w: empty content in response", ErrInvalidResponse)
	}

	return result.Message.Content, nil
}
