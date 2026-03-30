// SPDX-FileCopyrightText: 2026 Milos Vasic
// SPDX-License-Identifier: Apache-2.0

package remote

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

// LlamaCppDeployer manages llama.cpp server instances on a
// remote GPU host via SSH. It handles starting/stopping
// llama-server processes and freeing GPU resources.
type LlamaCppDeployer struct {
	config LlamaCppConfig
}

// NewLlamaCppDeployer creates a deployer with the given
// llama.cpp configuration.
func NewLlamaCppDeployer(config LlamaCppConfig) *LlamaCppDeployer {
	return &LlamaCppDeployer{config: config}
}

// FreeGPU stops Ollama on the remote host to free GPU VRAM
// for llama-server instances. This is a no-op if Ollama is
// not running.
func (d *LlamaCppDeployer) FreeGPU(ctx context.Context) {
	_ = d.sshCmd(ctx, "systemctl", "--user", "stop", "ollama")
}

// StartInstance launches a llama-server process on the remote
// host at the specified port. The server runs in the
// background and listens for HTTP requests.
func (d *LlamaCppDeployer) StartInstance(
	ctx context.Context, port int,
) {
	args := []string{
		fmt.Sprintf("%s/build/bin/llama-server", d.config.RepoDir),
		"-m", d.config.ModelPath,
		"--mmproj", d.config.MMProjPath,
		"--port", fmt.Sprintf("%d", port),
		"-ngl", fmt.Sprintf("%d", d.config.GPULayers),
		"-c", fmt.Sprintf("%d", d.config.ContextSize),
	}
	cmd := strings.Join(args, " ")
	_ = d.sshCmd(ctx, "nohup", cmd, ">/dev/null", "2>&1", "&")
}

// RestoreOllama restarts Ollama on the remote host after
// llama-server instances have been stopped.
func (d *LlamaCppDeployer) RestoreOllama(ctx context.Context) {
	_ = d.sshCmd(ctx, "systemctl", "--user", "start", "ollama")
}

// sshCmd runs a command on the remote host via SSH.
func (d *LlamaCppDeployer) sshCmd(
	ctx context.Context, args ...string,
) error {
	if d.config.Host == "" {
		return fmt.Errorf("remote: deployer host is required")
	}
	target := d.config.Host
	if d.config.User != "" {
		target = d.config.User + "@" + d.config.Host
	}
	sshArgs := append(
		[]string{"-o", "StrictHostKeyChecking=no", target},
		args...,
	)
	cmd := exec.CommandContext(ctx, "ssh", sshArgs...)
	return cmd.Run()
}
