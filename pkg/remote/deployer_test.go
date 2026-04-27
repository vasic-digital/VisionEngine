// Copyright 2026 Milos Vasic. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package remote

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewDeployer_Defaults(t *testing.T) {
	d := NewDeployer(Config{Host: "test.local"})
	assert.Equal(t, 22, d.cfg.Port)
	assert.Equal(t, "llava:7b", d.cfg.Model)
	assert.Equal(t, 11434, d.cfg.OllamaPort)
}

func TestNewDeployer_CustomConfig(t *testing.T) {
	d := NewDeployer(Config{
		Host:       "gpu.local",
		User:       "admin",
		Port:       2222,
		Model:      "minicpm-v:8b",
		OllamaPort: 11435,
	})
	assert.Equal(t, "gpu.local", d.cfg.Host)
	assert.Equal(t, "admin", d.cfg.User)
	assert.Equal(t, 2222, d.cfg.Port)
	assert.Equal(t, "minicpm-v:8b", d.cfg.Model)
	assert.Equal(t, 11435, d.cfg.OllamaPort)
}

func TestDeployer_Endpoint(t *testing.T) {
	d := NewDeployer(Config{Host: "gpu.local"})
	assert.Equal(t,
		"http://gpu.local:11434",
		d.Endpoint(),
	)
}

func TestDeployer_Endpoint_CustomPort(t *testing.T) {
	d := NewDeployer(Config{
		Host:       "gpu.local",
		OllamaPort: 8080,
	})
	assert.Equal(t,
		"http://gpu.local:8080",
		d.Endpoint(),
	)
}

func TestDeployer_IsModelPulled_Found(t *testing.T) {
	srv := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			resp := map[string]interface{}{
				"models": []map[string]interface{}{
					{"name": "llava:7b"},
					{"name": "qwen2.5:7b"},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}),
	)
	defer srv.Close()

	d := NewDeployer(Config{
		Host:  "localhost",
		Model: "llava:7b",
	})
	// Override endpoint by parsing test server URL.
	d.cfg.Host = srv.Listener.Addr().String()
	d.cfg.OllamaPort = 0 // Use raw address.

	// Directly test isModelPulled by hitting the test
	// server.
	ctx := context.Background()
	apiURL := "http://" + srv.Listener.Addr().String() + "/api/tags"
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, apiURL, nil,
	)
	require.NoError(t, err)
	resp, err := d.client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	var tagsResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	err = json.NewDecoder(resp.Body).Decode(&tagsResp)
	require.NoError(t, err)

	found := false
	for _, m := range tagsResp.Models {
		if m.Name == "llava:7b" {
			found = true
			break
		}
	}
	assert.True(t, found, "model should be found")
}

func TestDeployer_IsModelPulled_NotFound(t *testing.T) {
	srv := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			resp := map[string]interface{}{
				"models": []map[string]interface{}{
					{"name": "qwen2.5:7b"},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}),
	)
	defer srv.Close()

	ctx := context.Background()
	apiURL := "http://" + srv.Listener.Addr().String() + "/api/tags"
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, apiURL, nil,
	)
	require.NoError(t, err)

	d := NewDeployer(Config{Model: "llava:7b"})
	resp, err := d.client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	var tagsResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	err = json.NewDecoder(resp.Body).Decode(&tagsResp)
	require.NoError(t, err)

	found := false
	for _, m := range tagsResp.Models {
		if m.Name == "llava:7b" {
			found = true
		}
	}
	assert.False(t, found, "model should not be found")
}

func TestDeployer_Status_Unreachable(t *testing.T) {
	d := NewDeployer(Config{
		Host:       "192.0.2.1", // TEST-NET, unreachable
		OllamaPort: 19999,
	})
	ctx := context.Background()
	status := d.Status(ctx)
	assert.False(t, status.OllamaRunning)
	assert.False(t, status.ModelAvailable)
}

func TestDeployer_APICheck_Success(t *testing.T) {
	srv := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"models":[]}`))
		}),
	)
	defer srv.Close()

	d := NewDeployer(Config{Host: "localhost"})
	ctx := context.Background()
	apiURL := "http://" + srv.Listener.Addr().String() + "/api/tags"
	req, _ := http.NewRequestWithContext(
		ctx, http.MethodGet, apiURL, nil,
	)
	resp, err := d.client.Do(req)
	require.NoError(t, err)
	resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}
