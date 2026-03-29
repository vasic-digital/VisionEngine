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

func TestNewOllamaProvider_Defaults(t *testing.T) {
	p, err := NewOllamaProvider(ProviderConfig{})
	require.NoError(t, err)
	assert.Equal(t, "ollama", p.Name())
	assert.True(t, p.SupportsVision())
	assert.Equal(t, ollamaMaxImageSize, p.MaxImageSize())
}

func TestNewOllamaProvider_CustomConfig(t *testing.T) {
	p, err := NewOllamaProvider(ProviderConfig{
		BaseURL:      "http://192.168.1.100:11434",
		Model:        "llava:13b",
		MaxImageSize: 10 * 1024 * 1024,
		TimeoutSecs:  300,
	})
	require.NoError(t, err)
	assert.Equal(t, 10*1024*1024, p.MaxImageSize())
	assert.NotNil(t, p.httpClient)
}

func TestNewOllamaProvider_NoAPIKeyRequired(t *testing.T) {
	// Ollama is local inference — no API key needed.
	p, err := NewOllamaProvider(ProviderConfig{})
	require.NoError(t, err)
	assert.NotNil(t, p)
}

// --- Interface Compliance ---

func TestOllamaProvider_ImplementsInterface(t *testing.T) {
	var _ VisionProvider = (*OllamaProvider)(nil)
}

// --- Name and SupportsVision ---

func TestOllamaProvider_Name(t *testing.T) {
	p, _ := NewOllamaProvider(ProviderConfig{})
	assert.Equal(t, "ollama", p.Name())
}

func TestOllamaProvider_SupportsVision(t *testing.T) {
	p, _ := NewOllamaProvider(ProviderConfig{})
	assert.True(t, p.SupportsVision())
}

// --- AnalyzeImage Tests ---

func TestOllamaProvider_AnalyzeImage_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/chat", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var reqBody map[string]any
		json.NewDecoder(r.Body).Decode(&reqBody)
		assert.Equal(t, ollamaDefaultModel, reqBody["model"])
		assert.Equal(t, false, reqBody["stream"])

		messages := reqBody["messages"].([]any)
		assert.Len(t, messages, 1)
		msg := messages[0].(map[string]any)
		assert.Equal(t, "user", msg["role"])
		images := msg["images"].([]any)
		assert.Len(t, images, 1)

		resp := map[string]any{
			"message": map[string]string{
				"role":    "assistant",
				"content": "I see a dashboard with charts",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, err := NewOllamaProvider(ProviderConfig{BaseURL: server.URL})
	require.NoError(t, err)

	result, err := p.AnalyzeImage(context.Background(), []byte("fake-image"), "What do you see?")
	require.NoError(t, err)
	assert.Equal(t, "I see a dashboard with charts", result)
}

func TestOllamaProvider_AnalyzeImage_EmptyImage(t *testing.T) {
	p, _ := NewOllamaProvider(ProviderConfig{})
	_, err := p.AnalyzeImage(context.Background(), []byte{}, "prompt")
	assert.ErrorIs(t, err, ErrEmptyImage)
}

func TestOllamaProvider_AnalyzeImage_EmptyPrompt(t *testing.T) {
	p, _ := NewOllamaProvider(ProviderConfig{})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "")
	assert.ErrorIs(t, err, ErrEmptyPrompt)
}

func TestOllamaProvider_AnalyzeImage_ImageTooLarge(t *testing.T) {
	p, _ := NewOllamaProvider(ProviderConfig{MaxImageSize: 10})
	_, err := p.AnalyzeImage(context.Background(), make([]byte, 11), "prompt")
	assert.ErrorIs(t, err, ErrImageTooLarge)
}

func TestOllamaProvider_AnalyzeImage_RateLimited(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer server.Close()

	p, _ := NewOllamaProvider(ProviderConfig{BaseURL: server.URL})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.ErrorIs(t, err, ErrRateLimited)
}

func TestOllamaProvider_AnalyzeImage_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "model not found"}`))
	}))
	defer server.Close()

	p, _ := NewOllamaProvider(ProviderConfig{BaseURL: server.URL})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrProviderUnavailable)
}

func TestOllamaProvider_AnalyzeImage_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`not json`))
	}))
	defer server.Close()

	p, _ := NewOllamaProvider(ProviderConfig{BaseURL: server.URL})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.ErrorIs(t, err, ErrInvalidResponse)
}

func TestOllamaProvider_AnalyzeImage_EmptyContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"message": map[string]string{
				"role":    "assistant",
				"content": "",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewOllamaProvider(ProviderConfig{BaseURL: server.URL})
	_, err := p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.ErrorIs(t, err, ErrInvalidResponse)
}

func TestOllamaProvider_AnalyzeImage_Cancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {}
	}))
	defer server.Close()

	p, _ := NewOllamaProvider(ProviderConfig{BaseURL: server.URL})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := p.AnalyzeImage(ctx, []byte("img"), "prompt")
	assert.Error(t, err)
}

// --- CompareImages Tests ---

func TestOllamaProvider_CompareImages_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]any
		json.NewDecoder(r.Body).Decode(&reqBody)

		messages := reqBody["messages"].([]any)
		msg := messages[0].(map[string]any)
		images := msg["images"].([]any)
		assert.Len(t, images, 2)

		resp := map[string]any{
			"message": map[string]string{
				"role":    "assistant",
				"content": "The header color changed from blue to red",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewOllamaProvider(ProviderConfig{BaseURL: server.URL})
	result, err := p.CompareImages(context.Background(), []byte("img1"), []byte("img2"), "Compare these")
	require.NoError(t, err)
	assert.Contains(t, result, "header color changed")
}

func TestOllamaProvider_CompareImages_EmptyFirstImage(t *testing.T) {
	p, _ := NewOllamaProvider(ProviderConfig{})
	_, err := p.CompareImages(context.Background(), []byte{}, []byte("img"), "prompt")
	assert.ErrorIs(t, err, ErrEmptyImage)
}

func TestOllamaProvider_CompareImages_EmptySecondImage(t *testing.T) {
	p, _ := NewOllamaProvider(ProviderConfig{})
	_, err := p.CompareImages(context.Background(), []byte("img"), []byte{}, "prompt")
	assert.ErrorIs(t, err, ErrEmptyImage)
}

func TestOllamaProvider_CompareImages_EmptyPrompt(t *testing.T) {
	p, _ := NewOllamaProvider(ProviderConfig{})
	_, err := p.CompareImages(context.Background(), []byte("img1"), []byte("img2"), "")
	assert.ErrorIs(t, err, ErrEmptyPrompt)
}

// --- Custom Model ---

func TestOllamaProvider_CustomModel(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		resp := map[string]any{
			"message": map[string]string{
				"role":    "assistant",
				"content": "ok",
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewOllamaProvider(ProviderConfig{
		BaseURL: server.URL,
		Model:   "llava:34b",
	})
	_, _ = p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.Equal(t, "llava:34b", receivedBody["model"])
}

// --- Stream disabled ---

func TestOllamaProvider_StreamDisabled(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		resp := map[string]any{
			"message": map[string]string{
				"role":    "assistant",
				"content": "ok",
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewOllamaProvider(ProviderConfig{BaseURL: server.URL})
	_, _ = p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
	assert.Equal(t, false, receivedBody["stream"])
}

// --- Custom Timeout ---

func TestOllamaProvider_CustomTimeout(t *testing.T) {
	p, err := NewOllamaProvider(ProviderConfig{TimeoutSecs: 300})
	require.NoError(t, err)
	assert.NotNil(t, p.httpClient)
}

// --- No auth header ---

func TestOllamaProvider_NoAuthHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Empty(t, r.Header.Get("Authorization"))
		assert.Empty(t, r.Header.Get("x-api-key"))
		resp := map[string]any{
			"message": map[string]string{
				"role":    "assistant",
				"content": "ok",
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p, _ := NewOllamaProvider(ProviderConfig{BaseURL: server.URL})
	_, _ = p.AnalyzeImage(context.Background(), []byte("img"), "prompt")
}
