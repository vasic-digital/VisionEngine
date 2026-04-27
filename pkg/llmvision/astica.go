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
	asticaDefaultEndpoint  = "https://vision.astica.ai/describe"
	asticaDefaultModelVer  = "2.5_full"
	asticaMaxImageSize     = 20 * 1024 * 1024 // 20 MB
	asticaDefaultVisionParams = "describe,objects,faces,text"
)

// AsticaProvider implements VisionProvider using the Astica.AI Vision API.
// Astica specializes in comprehensive image understanding: captioning,
// object detection, OCR, face detection, and content moderation.
type AsticaProvider struct {
	config     ProviderConfig
	modelVer   string
	httpClient *http.Client
}

// NewAsticaProvider creates a new Astica vision provider.
func NewAsticaProvider(config ProviderConfig) (*AsticaProvider, error) {
	if config.APIKey == "" {
		return nil, ErrNoAPIKey
	}
	if config.BaseURL == "" {
		config.BaseURL = asticaDefaultEndpoint
	}
	modelVer := config.Model
	if modelVer == "" {
		modelVer = asticaDefaultModelVer
	}
	if config.MaxImageSize == 0 {
		config.MaxImageSize = asticaMaxImageSize
	}
	timeout := 60 * time.Second
	if config.TimeoutSecs > 0 {
		timeout = time.Duration(config.TimeoutSecs) * time.Second
	}
	return &AsticaProvider{
		config:   config,
		modelVer: modelVer,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// Name returns "astica".
func (p *AsticaProvider) Name() string {
	return "astica"
}

// SupportsVision returns true.
func (p *AsticaProvider) SupportsVision() bool {
	return true
}

// MaxImageSize returns the max image size.
func (p *AsticaProvider) MaxImageSize() int {
	return p.config.MaxImageSize
}

// AnalyzeImage sends an image to the Astica.AI Vision API for analysis.
func (p *AsticaProvider) AnalyzeImage(ctx context.Context, image []byte, prompt string) (string, error) {
	if err := validateImage(image, p.config.MaxImageSize); err != nil {
		return "", err
	}
	if err := validatePrompt(prompt); err != nil {
		return "", err
	}

	encoded := base64.StdEncoding.EncodeToString(image)
	body := map[string]any{
		"tkn":          p.config.APIKey,
		"modelVersion": p.modelVer,
		"input":        fmt.Sprintf("data:image/png;base64,%s", encoded),
		"visionParams": asticaDefaultVisionParams,
		"gpt_prompt":   prompt,
	}

	return p.sendRequest(ctx, body)
}

// CompareImages sends two images to the Astica.AI Vision API for comparison.
// Astica does not natively support multi-image comparison, so both images
// are analyzed individually and the results are combined with the prompt
// requesting a comparison.
func (p *AsticaProvider) CompareImages(ctx context.Context, img1, img2 []byte, prompt string) (string, error) {
	if err := validateImage(img1, p.config.MaxImageSize); err != nil {
		return "", err
	}
	if err := validateImage(img2, p.config.MaxImageSize); err != nil {
		return "", err
	}
	if err := validatePrompt(prompt); err != nil {
		return "", err
	}

	// Analyze first image with the comparison prompt.
	result1, err := p.AnalyzeImage(ctx, img1, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to analyze first image: %w", err)
	}

	// Analyze second image with the comparison prompt.
	result2, err := p.AnalyzeImage(ctx, img2, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to analyze second image: %w", err)
	}

	return fmt.Sprintf("Image 1: %s\n\nImage 2: %s", result1, result2), nil
}

// asticaResponse models the JSON response from the Astica Vision API.
type asticaResponse struct {
	Status     string `json:"status"`
	CaptionGPT string `json:"caption_GPTS"`
	Caption    struct {
		Text       string  `json:"text"`
		Confidence float64 `json:"confidence"`
	} `json:"caption"`
}

func (p *AsticaProvider) sendRequest(ctx context.Context, body map[string]any) (string, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.config.BaseURL, bytes.NewReader(jsonBody))
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

	var result asticaResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("%w: %v", ErrInvalidResponse, err)
	}

	if result.Status != "success" {
		return "", fmt.Errorf("%w: API returned status %q", ErrInvalidResponse, result.Status)
	}

	// Prefer the detailed GPT-powered caption; fall back to the
	// standard caption if the GPT field is empty.
	content := result.CaptionGPT
	if content == "" {
		content = result.Caption.Text
	}
	if content == "" {
		return "", fmt.Errorf("%w: empty caption in response", ErrInvalidResponse)
	}

	return content, nil
}
