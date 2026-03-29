// Copyright 2026 Milos Vasic. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package remote provides automatic deployment and lifecycle
// management of Ollama vision models on remote hosts via SSH.
// It ensures Ollama is installed, the target vision model is
// pulled, and the Ollama API is accessible for HelixQA sessions.
//
// Usage:
//
//	d := remote.NewDeployer(remote.Config{
//	    Host:  "thinker.local",
//	    User:  "milosvasic",
//	    Model: "llava:7b",
//	})
//	endpoint, err := d.EnsureReady(ctx)
//	// endpoint = "http://thinker.local:11434"
package remote

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"time"
)

// Config holds SSH and Ollama configuration for remote
// deployment.
type Config struct {
	// Host is the hostname or IP of the remote machine.
	Host string
	// User is the SSH username.
	User string
	// Port is the SSH port (default 22).
	Port int
	// Model is the Ollama model to pull and serve
	// (default "llava:7b").
	Model string
	// OllamaPort is the Ollama API port (default 11434).
	OllamaPort int
}

// DeployStatus represents the state of a remote Ollama
// instance.
type DeployStatus struct {
	// OllamaInstalled reports whether Ollama is found.
	OllamaInstalled bool `json:"ollama_installed"`
	// OllamaRunning reports whether the API is reachable.
	OllamaRunning bool `json:"ollama_running"`
	// ModelAvailable reports whether the target model is
	// pulled.
	ModelAvailable bool `json:"model_available"`
	// OllamaVersion is the installed Ollama version.
	OllamaVersion string `json:"ollama_version"`
	// Endpoint is the full URL to the Ollama API.
	Endpoint string `json:"endpoint"`
	// Error holds any error message.
	Error string `json:"error,omitempty"`
}

// Deployer manages remote Ollama installations via SSH.
type Deployer struct {
	cfg    Config
	client *http.Client
}

// NewDeployer creates a Deployer with the given configuration.
func NewDeployer(cfg Config) *Deployer {
	if cfg.Port == 0 {
		cfg.Port = 22
	}
	if cfg.Model == "" {
		cfg.Model = "llava:7b"
	}
	if cfg.OllamaPort == 0 {
		cfg.OllamaPort = 11434
	}
	return &Deployer{
		cfg: cfg,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Endpoint returns the Ollama API URL for this remote host.
func (d *Deployer) Endpoint() string {
	return fmt.Sprintf(
		"http://%s:%d", d.cfg.Host, d.cfg.OllamaPort,
	)
}

// Status checks the current state of the remote Ollama
// installation without making changes.
func (d *Deployer) Status(
	ctx context.Context,
) DeployStatus {
	status := DeployStatus{
		Endpoint: d.Endpoint(),
	}

	// Check Ollama installed.
	ver, err := d.sshRun(ctx, "ollama --version")
	if err == nil && strings.Contains(ver, "ollama") {
		status.OllamaInstalled = true
		status.OllamaVersion = strings.TrimSpace(ver)
	}

	// Check API reachable.
	apiURL := d.Endpoint() + "/api/tags"
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, apiURL, nil,
	)
	if err == nil {
		resp, err := d.client.Do(req)
		if err == nil {
			resp.Body.Close()
			status.OllamaRunning = resp.StatusCode == http.StatusOK
		}
	}

	// Check model pulled.
	if status.OllamaRunning {
		status.ModelAvailable = d.isModelPulled(ctx)
	}

	return status
}

// EnsureReady performs the full deployment sequence:
// 1) Install Ollama if missing
// 2) Start Ollama service if not running
// 3) Pull the vision model if not available
//
// Returns the API endpoint URL on success.
func (d *Deployer) EnsureReady(
	ctx context.Context,
) (string, error) {
	fmt.Printf(
		"[vision-deploy] ensuring %s is ready on %s\n",
		d.cfg.Model, d.cfg.Host,
	)

	// Step 1: Check if Ollama is installed.
	status := d.Status(ctx)
	if !status.OllamaInstalled {
		fmt.Printf(
			"[vision-deploy] installing Ollama on %s\n",
			d.cfg.Host,
		)
		if err := d.installOllama(ctx); err != nil {
			return "", fmt.Errorf(
				"install ollama: %w", err,
			)
		}
	}

	// Step 2: Start Ollama if not running.
	if !status.OllamaRunning {
		fmt.Printf(
			"[vision-deploy] starting Ollama on %s\n",
			d.cfg.Host,
		)
		if err := d.startOllama(ctx); err != nil {
			return "", fmt.Errorf(
				"start ollama: %w", err,
			)
		}
		// Wait for API to become ready.
		if err := d.waitForAPI(ctx, 30); err != nil {
			return "", fmt.Errorf(
				"ollama API not ready: %w", err,
			)
		}
	}

	// Step 3: Pull model if not available.
	if !d.isModelPulled(ctx) {
		fmt.Printf(
			"[vision-deploy] pulling %s on %s\n",
			d.cfg.Model, d.cfg.Host,
		)
		if err := d.pullModel(ctx); err != nil {
			return "", fmt.Errorf(
				"pull model: %w", err,
			)
		}
	}

	fmt.Printf(
		"[vision-deploy] %s ready at %s\n",
		d.cfg.Model, d.Endpoint(),
	)
	return d.Endpoint(), nil
}

// sshRun executes a command on the remote host via SSH and
// returns the combined output.
func (d *Deployer) sshRun(
	ctx context.Context,
	command string,
) (string, error) {
	args := []string{
		"-o", "ConnectTimeout=10",
		"-o", "StrictHostKeyChecking=no",
		"-o", "BatchMode=yes",
	}
	if d.cfg.Port != 22 {
		args = append(args,
			"-p", fmt.Sprintf("%d", d.cfg.Port),
		)
	}
	target := d.cfg.Host
	if d.cfg.User != "" {
		target = d.cfg.User + "@" + d.cfg.Host
	}
	args = append(args, target, command)

	cmd := exec.CommandContext(ctx, "ssh", args...)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// installOllama runs the Ollama install script on the
// remote host.
func (d *Deployer) installOllama(
	ctx context.Context,
) error {
	// Ollama's official install script.
	_, err := d.sshRun(ctx,
		"curl -fsSL https://ollama.com/install.sh | sh",
	)
	return err
}

// startOllama starts the Ollama service on the remote host.
func (d *Deployer) startOllama(
	ctx context.Context,
) error {
	// Try systemd first, fall back to nohup.
	_, err := d.sshRun(ctx,
		"systemctl start ollama 2>/dev/null || "+
			"nohup ollama serve > /dev/null 2>&1 &",
	)
	return err
}

// waitForAPI polls the Ollama /api/tags endpoint until it
// responds or the retry limit is reached.
func (d *Deployer) waitForAPI(
	ctx context.Context,
	maxRetries int,
) error {
	apiURL := d.Endpoint() + "/api/tags"
	for i := 0; i < maxRetries; i++ {
		req, err := http.NewRequestWithContext(
			ctx, http.MethodGet, apiURL, nil,
		)
		if err != nil {
			return err
		}
		resp, err := d.client.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf(
		"ollama API on %s not ready after %d retries",
		d.cfg.Host, maxRetries,
	)
}

// isModelPulled checks if the target model is available on
// the remote Ollama instance by querying /api/tags.
func (d *Deployer) isModelPulled(
	ctx context.Context,
) bool {
	apiURL := d.Endpoint() + "/api/tags"
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, apiURL, nil,
	)
	if err != nil {
		return false
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	var tagsResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(
		&tagsResp,
	); err != nil {
		return false
	}

	// Match model name (with or without tag suffix).
	target := d.cfg.Model
	for _, m := range tagsResp.Models {
		if m.Name == target ||
			strings.HasPrefix(m.Name, target+":") ||
			strings.HasPrefix(target, m.Name) {
			return true
		}
	}
	return false
}

// pullModel pulls the target model on the remote host via
// the Ollama CLI.
func (d *Deployer) pullModel(
	ctx context.Context,
) error {
	pullCtx, cancel := context.WithTimeout(
		ctx, 30*time.Minute,
	)
	defer cancel()

	_, err := d.sshRun(pullCtx,
		fmt.Sprintf("ollama pull %s", d.cfg.Model),
	)
	return err
}
