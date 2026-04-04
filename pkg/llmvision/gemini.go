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
	geminiDefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
	geminiDefaultModel   = "gemini-2.5-flash"
	geminiMaxImageSize   = 20 * 1024 * 1024 // 20 MB
)

// GeminiProvider implements VisionProvider for Google Gemini.
type GeminiProvider struct {
	config     ProviderConfig
	httpClient *http.Client
}

// NewGeminiProvider creates a new Gemini vision provider.
func NewGeminiProvider(config ProviderConfig) (*GeminiProvider, error) {
	if config.APIKey == "" {
		return nil, ErrNoAPIKey
	}
	if config.BaseURL == "" {
		config.BaseURL = geminiDefaultBaseURL
	}
	if config.Model == "" {
		config.Model = geminiDefaultModel
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 4096
	}
	if config.MaxImageSize == 0 {
		config.MaxImageSize = geminiMaxImageSize
	}
	timeout := 60 * time.Second
	if config.TimeoutSecs > 0 {
		timeout = time.Duration(config.TimeoutSecs) * time.Second
	}
	return &GeminiProvider{
		config: config,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// Name returns "gemini".
func (p *GeminiProvider) Name() string {
	return "gemini"
}

// SupportsVision returns true.
func (p *GeminiProvider) SupportsVision() bool {
	return true
}

// MaxImageSize returns the max image size.
func (p *GeminiProvider) MaxImageSize() int {
	return p.config.MaxImageSize
}

// AnalyzeImage sends an image to Gemini for analysis.
func (p *GeminiProvider) AnalyzeImage(ctx context.Context, image []byte, prompt string) (string, error) {
	if err := validateImage(image, p.config.MaxImageSize); err != nil {
		return "", err
	}
	if err := validatePrompt(prompt); err != nil {
		return "", err
	}

	encoded := base64.StdEncoding.EncodeToString(image)
	body := map[string]any{
		"contents": []map[string]any{
			{
				"parts": []map[string]any{
					{"text": prompt},
					{
						"inline_data": map[string]string{
							"mime_type": "image/png",
							"data":      encoded,
						},
					},
				},
			},
		},
		"generationConfig": map[string]any{
			"maxOutputTokens": p.config.MaxTokens,
		},
	}

	return p.sendRequest(ctx, body)
}

// CompareImages sends two images to Gemini for comparison.
func (p *GeminiProvider) CompareImages(ctx context.Context, img1, img2 []byte, prompt string) (string, error) {
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
		"contents": []map[string]any{
			{
				"parts": []map[string]any{
					{"text": prompt},
					{
						"inline_data": map[string]string{
							"mime_type": "image/png",
							"data":      enc1,
						},
					},
					{
						"inline_data": map[string]string{
							"mime_type": "image/png",
							"data":      enc2,
						},
					},
				},
			},
		},
		"generationConfig": map[string]any{
			"maxOutputTokens": p.config.MaxTokens,
		},
	}

	return p.sendRequest(ctx, body)
}

func (p *GeminiProvider) sendRequest(ctx context.Context, body map[string]any) (string, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s",
		p.config.BaseURL, p.config.Model, p.config.APIKey)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
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
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("%w: %v", ErrInvalidResponse, err)
	}
	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("%w: no content in response", ErrInvalidResponse)
	}

	return result.Candidates[0].Content.Parts[0].Text, nil
}
