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
	kimiDefaultBaseURL = "https://api.moonshot.cn/v1"
	kimiDefaultModel   = "kimi-k2.5"
	kimiMaxImageSize   = 20 * 1024 * 1024 // 20 MB
)

// KimiProvider implements VisionProvider for Moonshot AI Kimi K2.5.
// Uses OpenAI-compatible API format.
type KimiProvider struct {
	config     ProviderConfig
	httpClient *http.Client
}

// NewKimiProvider creates a new Kimi vision provider.
func NewKimiProvider(config ProviderConfig) (*KimiProvider, error) {
	if config.APIKey == "" {
		return nil, ErrNoAPIKey
	}
	if config.BaseURL == "" {
		config.BaseURL = kimiDefaultBaseURL
	}
	if config.Model == "" {
		config.Model = kimiDefaultModel
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 4096
	}
	if config.MaxImageSize == 0 {
		config.MaxImageSize = kimiMaxImageSize
	}
	timeout := 60 * time.Second
	if config.TimeoutSecs > 0 {
		timeout = time.Duration(config.TimeoutSecs) * time.Second
	}
	return &KimiProvider{
		config: config,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// Name returns "kimi".
func (p *KimiProvider) Name() string {
	return "kimi"
}

// SupportsVision returns true.
func (p *KimiProvider) SupportsVision() bool {
	return true
}

// MaxImageSize returns the max image size.
func (p *KimiProvider) MaxImageSize() int {
	return p.config.MaxImageSize
}

// AnalyzeImage sends an image to Kimi K2.5 for analysis.
func (p *KimiProvider) AnalyzeImage(ctx context.Context, image []byte, prompt string) (string, error) {
	if err := validateImage(image, p.config.MaxImageSize); err != nil {
		return "", err
	}
	if err := validatePrompt(prompt); err != nil {
		return "", err
	}

	encoded := base64.StdEncoding.EncodeToString(image)
	body := map[string]any{
		"model": p.config.Model,
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": prompt},
					{
						"type": "image_url",
						"image_url": map[string]string{
							"url": fmt.Sprintf("data:image/png;base64,%s", encoded),
						},
					},
				},
			},
		},
		"max_tokens": p.config.MaxTokens,
	}

	return p.sendRequest(ctx, body)
}

// CompareImages sends two images to Kimi K2.5 for comparison.
func (p *KimiProvider) CompareImages(ctx context.Context, img1, img2 []byte, prompt string) (string, error) {
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
		"model": p.config.Model,
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": prompt},
					{
						"type": "image_url",
						"image_url": map[string]string{
							"url": fmt.Sprintf("data:image/png;base64,%s", enc1),
						},
					},
					{
						"type": "image_url",
						"image_url": map[string]string{
							"url": fmt.Sprintf("data:image/png;base64,%s", enc2),
						},
					},
				},
			},
		},
		"max_tokens": p.config.MaxTokens,
	}

	return p.sendRequest(ctx, body)
}

func (p *KimiProvider) sendRequest(ctx context.Context, body map[string]any) (string, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.config.BaseURL+"/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.config.APIKey)

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
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("%w: %v", ErrInvalidResponse, err)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("%w: no choices in response", ErrInvalidResponse)
	}

	return result.Choices[0].Message.Content, nil
}
