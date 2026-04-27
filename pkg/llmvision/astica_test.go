// Copyright 2026 Milos Vasic. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package llmvision

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Constructor Tests ---

func TestNewAsticaProvider_Success(t *testing.T) {
	p, err := NewAsticaProvider(ProviderConfig{APIKey: "test-key"})
	require.NoError(t, err)
	assert.Equal(t, "astica", p.Name())
	assert.True(t, p.SupportsVision())
	assert.Equal(t, asticaMaxImageSize, p.MaxImageSize())
}

func TestNewAsticaProvider_NoAPIKey(t *testing.T) {
	_, err := NewAsticaProvider(ProviderConfig{})
	assert.ErrorIs(t, err, ErrNoAPIKey)
}

func TestNewAsticaProvider_CustomConfig(t *testing.T) {
	p, err := NewAsticaProvider(ProviderConfig{
		APIKey:       "test-key",
		BaseURL:      "https://custom.astica.ai/describe",
		Model:        "2.1_full",
		MaxImageSize: 10 * 1024 * 1024,
		TimeoutSecs:  120,
	})
	require.NoError(t, err)
	assert.Equal(t, 10*1024*1024, p.MaxImageSize())
	assert.Equal(t, "2.1_full", p.modelVer)
	assert.NotNil(t, p.httpClient)
}

func TestNewAsticaProvider_Defaults(t *testing.T) {
	p, err := NewAsticaProvider(ProviderConfig{APIKey: "test-key"})
	require.NoError(t, err)
	assert.Equal(t, asticaDefaultEndpoint, p.config.BaseURL)
	assert.Equal(t, asticaDefaultModelVer, p.modelVer)
	assert.Equal(t, asticaMaxImageSize, p.config.MaxImageSize)
}

// --- Interface Compliance ---

func TestAsticaProvider_ImplementsInterface(t *testing.T) {
	var _ VisionProvider = (*AsticaProvider)(nil)
}

// --- Name and SupportsVision ---

func TestAsticaProvider_Name(t *testing.T) {
	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key"})
	assert.Equal(t, "astica", p.Name())
}

func TestAsticaProvider_SupportsVision(t *testing.T) {
	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key"})
	assert.True(t, p.SupportsVision())
}

// --- AnalyzeImage Tests ---

func TestAsticaProvider_AnalyzeImage_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var reqBody map[string]any
		json.NewDecoder(r.Body).Decode(&reqBody)
		assert.Equal(t, "test-key", reqBody["tkn"])
		assert.Equal(t, "2.5_full", reqBody["modelVersion"])
		assert.Equal(t, "describe,objects,faces,text", reqBody["visionParams"])
		assert.NotEmpty(t, reqBody["input"])
		assert.NotEmpty(t, reqBody["gpt_prompt"])

		resp := map[string]any{
			"status":       "success",
			"caption_GPTS": "I see a dashboard with navigation elements and charts",
			"caption": map[string]any{
				"text":       "A dashboard interface",
				"confidence": 0.95,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, err := NewAsticaProvider(ProviderConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})
	require.NoError(t, err)

	result, err := p.AnalyzeImage(context.Background(), []byte("fake-image"), "What do you see?")
	require.NoError(t, err)
	assert.Equal(t, "I see a dashboard with navigation elements and charts", result)
}

func TestAsticaProvider_AnalyzeImage_FallbackToCaption(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"status":       "success",
			"caption_GPTS": "",
			"caption": map[string]any{
				"text":       "A screenshot of an application",
				"confidence": 0.90,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	result, err := p.AnalyzeImage(context.Background(), []byte("img"), "Describe this")
	require.NoError(t, err)
	assert.Equal(t, "A screenshot of an application", result)
}

func TestAsticaProvider_AnalyzeImage_EmptyImage(t *testing.T) {
	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key"})
	_, err := p.AnalyzeImage(context.Background(), []byte{}, "prompt")
	assert.ErrorIs(t, err, ErrEmptyImage)
}

func TestAsticaProvider_AnalyzeImage_EmptyPrompt(t *testing.T) {
	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key"})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "")
	assert.ErrorIs(t, err, ErrEmptyPrompt)
}

func TestAsticaProvider_AnalyzeImage_ImageTooLarge(t *testing.T) {
	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key", MaxImageSize: 10})
	_, err := p.AnalyzeImage(context.Background(), make([]byte, 11), "prompt")
	assert.ErrorIs(t, err, ErrImageTooLarge)
}

func TestAsticaProvider_AnalyzeImage_RateLimited(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key", BaseURL: server.URL})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.ErrorIs(t, err, ErrRateLimited)
}

func TestAsticaProvider_AnalyzeImage_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "internal server error"}`))
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key", BaseURL: server.URL})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrProviderUnavailable)
}

func TestAsticaProvider_AnalyzeImage_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`not json`))
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key", BaseURL: server.URL})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.ErrorIs(t, err, ErrInvalidResponse)
}

func TestAsticaProvider_AnalyzeImage_FailedStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"status": "error",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key", BaseURL: server.URL})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.ErrorIs(t, err, ErrInvalidResponse)
}

func TestAsticaProvider_AnalyzeImage_EmptyContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"status":       "success",
			"caption_GPTS": "",
			"caption": map[string]any{
				"text":       "",
				"confidence": 0.0,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key", BaseURL: server.URL})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.ErrorIs(t, err, ErrInvalidResponse)
}

func TestAsticaProvider_AnalyzeImage_Cancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {}
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key", BaseURL: server.URL})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := p.AnalyzeImage(ctx, []byte("img"), "prompt")
	assert.Error(t, err)
}

// --- CompareImages Tests ---

func TestAsticaProvider_CompareImages_Success(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		caption := "First image description"
		if callCount == 2 {
			caption = "Second image description"
		}
		resp := map[string]any{
			"status":       "success",
			"caption_GPTS": caption,
			"caption": map[string]any{
				"text":       "fallback",
				"confidence": 0.90,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key", BaseURL: server.URL})
	result, err := p.CompareImages(context.Background(), []byte("img1"), []byte("img2"), "Compare these")
	require.NoError(t, err)
	assert.Contains(t, result, "First image description")
	assert.Contains(t, result, "Second image description")
	assert.Equal(t, 2, callCount)
}

func TestAsticaProvider_CompareImages_EmptyFirstImage(t *testing.T) {
	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key"})
	_, err := p.CompareImages(context.Background(), []byte{}, []byte("img"), "prompt")
	assert.ErrorIs(t, err, ErrEmptyImage)
}

func TestAsticaProvider_CompareImages_EmptySecondImage(t *testing.T) {
	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key"})
	_, err := p.CompareImages(context.Background(), []byte("img"), []byte{}, "prompt")
	assert.ErrorIs(t, err, ErrEmptyImage)
}

func TestAsticaProvider_CompareImages_EmptyPrompt(t *testing.T) {
	p, _ := NewAsticaProvider(ProviderConfig{APIKey: "key"})
	_, err := p.CompareImages(context.Background(), []byte("img1"), []byte("img2"), "")
	assert.ErrorIs(t, err, ErrEmptyPrompt)
}

// --- Custom Model Version ---

func TestAsticaProvider_CustomModelVersion(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		resp := map[string]any{
			"status":       "success",
			"caption_GPTS": "ok",
			"caption": map[string]any{
				"text":       "ok",
				"confidence": 0.95,
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{
		APIKey:  "key",
		BaseURL: server.URL,
		Model:   "2.1_full",
	})
	_, _ = p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.Equal(t, "2.1_full", receivedBody["modelVersion"])
}

// --- No Auth Header (token in body) ---

func TestAsticaProvider_TokenInBody(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Astica uses token in body, not Authorization header
		assert.Empty(t, r.Header.Get("Authorization"))
		json.NewDecoder(r.Body).Decode(&receivedBody)
		resp := map[string]any{
			"status":       "success",
			"caption_GPTS": "ok",
			"caption": map[string]any{
				"text":       "ok",
				"confidence": 0.95,
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewAsticaProvider(ProviderConfig{
		APIKey:  "my-secret-token",
		BaseURL: server.URL,
	})
	_, _ = p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.Equal(t, "my-secret-token", receivedBody["tkn"])
}

// --- Custom Timeout ---

func TestAsticaProvider_CustomTimeout(t *testing.T) {
	p, err := NewAsticaProvider(ProviderConfig{
		APIKey:      "key",
		TimeoutSecs: 300,
	})
	require.NoError(t, err)
	assert.NotNil(t, p.httpClient)
}
