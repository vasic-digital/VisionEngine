// Copyright 2026 Milos Vasic. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

// LlamaCppConfig configures a llama.cpp server deployment.
type LlamaCppConfig struct {
	// Host is the remote machine hostname.
	Host string
	// User is the SSH user.
	User string
	// SSHPort is the SSH port (default 22).
	SSHPort int
	// RepoDir is where llama.cpp is cloned on the remote
	// host. Default: ~/llama.cpp
	RepoDir string
	// ModelPath is the path to the GGUF model file on the
	// remote host. If empty, auto-downloads a vision model.
	ModelPath string
	// MMProjPath is the path to the multimodal projector
	// GGUF file (required for vision models like LLaVA).
	MMProjPath string
	// ModelURL is the URL to download the model from if
	// ModelPath doesn't exist.
	ModelURL string
	// ModelName is a human-readable name for logging.
	ModelName string
	// BasePort is the first port for llama-server instances.
	BasePort int
	// GPULayers is the number of layers to offload to GPU.
	// Use -1 for all layers, 0 for CPU-only (default: auto).
	// When set to -2 (auto), the deployer probes GPU
	// availability and uses GPU if free, CPU otherwise.
	GPULayers int
	// ContextSize is the context window size (default 4096).
	ContextSize int
}

// LlamaCppDeployer manages llama.cpp installations and
// llama-server instances on a remote host via SSH.
type LlamaCppDeployer struct {
	cfg    LlamaCppConfig
	client *http.Client
}

// NewLlamaCppDeployer creates a deployer with the given config.
func NewLlamaCppDeployer(cfg LlamaCppConfig) *LlamaCppDeployer {
	if cfg.SSHPort == 0 {
		cfg.SSHPort = 22
	}
	if cfg.RepoDir == "" {
		cfg.RepoDir = "~/llama.cpp"
	}
	if cfg.BasePort == 0 {
		cfg.BasePort = 8090
	}
	if cfg.GPULayers == 0 {
		cfg.GPULayers = -1 // all layers on GPU
	}
	if cfg.ContextSize == 0 {
		cfg.ContextSize = 4096
	}
	if cfg.ModelName == "" {
		cfg.ModelName = "llava-v1.6-mistral-7b"
	}
	return &LlamaCppDeployer{
		cfg:    cfg,
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

// sshRun executes a command on the remote host via SSH.
func (d *LlamaCppDeployer) sshRun(
	ctx context.Context, command string,
) (string, error) {
	args := []string{
		"-o", "ConnectTimeout=10",
		"-o", "StrictHostKeyChecking=no",
		"-o", "BatchMode=yes",
	}
	if d.cfg.SSHPort != 22 {
		args = append(args,
			"-p", fmt.Sprintf("%d", d.cfg.SSHPort),
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

// FreeGPU attempts to stop Ollama (which typically consumes
// most of the GPU VRAM) so llama-server can use the GPU.
// Uses sudo where available. If unable to stop Ollama, logs
// a warning — llama-server will fall back to CPU mode.
func (d *LlamaCppDeployer) FreeGPU(
	ctx context.Context,
) {
	fmt.Println("[llamacpp] freeing GPU (stopping Ollama)...")
	// Try sudo systemctl (most reliable for root services).
	d.sshRun(ctx,
		"sudo systemctl stop ollama 2>/dev/null && "+
			"sudo systemctl mask ollama 2>/dev/null")
	// Also try non-sudo variants.
	d.sshRun(ctx, "systemctl --user stop ollama 2>/dev/null")
	d.sshRun(ctx, "pkill -f 'ollama serve' 2>/dev/null")
	d.sshRun(ctx, "sudo pkill -f 'ollama serve' 2>/dev/null")
	time.Sleep(3 * time.Second)

	// Check if GPU is now free.
	out, _ := d.sshRun(ctx,
		"nvidia-smi --query-gpu=memory.free "+
			"--format=csv,noheader,nounits 2>/dev/null",
	)
	var freeMB int
	fmt.Sscanf(strings.TrimSpace(out), "%d", &freeMB)
	fmt.Printf(
		"[llamacpp] GPU now has %dMB free\n", freeMB,
	)
}

// RestoreOllama restarts Ollama after QA session completes.
func (d *LlamaCppDeployer) RestoreOllama(
	ctx context.Context,
) {
	fmt.Println("[llamacpp] restoring Ollama service...")
	d.sshRun(ctx,
		"sudo systemctl unmask ollama 2>/dev/null && "+
			"sudo systemctl start ollama 2>/dev/null")
	// Fallback for non-sudo.
	d.sshRun(ctx, "systemctl --user start ollama 2>/dev/null")
	d.sshRun(ctx,
		"nohup ollama serve >/dev/null 2>&1 &")
}

// IsBuilt checks if llama-server binary exists on the remote.
func (d *LlamaCppDeployer) IsBuilt(
	ctx context.Context,
) bool {
	out, err := d.sshRun(ctx,
		fmt.Sprintf("test -x %s/build/bin/llama-server && echo yes",
			d.cfg.RepoDir),
	)
	return err == nil && strings.TrimSpace(out) == "yes"
}

// IsModelReady checks if the model file exists on the remote.
func (d *LlamaCppDeployer) IsModelReady(
	ctx context.Context,
) bool {
	if d.cfg.ModelPath == "" {
		return false
	}
	out, err := d.sshRun(ctx,
		fmt.Sprintf("test -f %s && echo yes", d.cfg.ModelPath),
	)
	return err == nil && strings.TrimSpace(out) == "yes"
}

// EnsureBuilt clones and builds llama.cpp if not already done.
func (d *LlamaCppDeployer) EnsureBuilt(
	ctx context.Context,
) error {
	if d.IsBuilt(ctx) {
		fmt.Printf(
			"[llamacpp] llama-server already built on %s\n",
			d.cfg.Host,
		)
		return nil
	}

	// Check if repo exists.
	out, _ := d.sshRun(ctx,
		fmt.Sprintf("test -d %s/.git && echo yes",
			d.cfg.RepoDir),
	)
	if strings.TrimSpace(out) != "yes" {
		fmt.Printf(
			"[llamacpp] cloning llama.cpp on %s\n",
			d.cfg.Host,
		)
		if _, err := d.sshRun(ctx,
			fmt.Sprintf(
				"git clone --depth 1 https://github.com/ggerganov/llama.cpp.git %s",
				d.cfg.RepoDir,
			),
		); err != nil {
			return fmt.Errorf("clone llama.cpp: %w", err)
		}
	}

	// Build with CUDA.
	fmt.Printf(
		"[llamacpp] building llama.cpp with CUDA on %s\n",
		d.cfg.Host,
	)
	buildCmd := fmt.Sprintf(
		"cd %s && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>&1 && cmake --build build --config Release -j$(nproc) 2>&1",
		d.cfg.RepoDir,
	)
	buildCtx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	if _, err := d.sshRun(buildCtx, buildCmd); err != nil {
		return fmt.Errorf("build llama.cpp: %w", err)
	}

	if !d.IsBuilt(ctx) {
		return fmt.Errorf(
			"llama-server binary not found after build",
		)
	}
	fmt.Printf("[llamacpp] build complete on %s\n", d.cfg.Host)
	return nil
}

// EnsureModel downloads the vision model if not present.
func (d *LlamaCppDeployer) EnsureModel(
	ctx context.Context,
) error {
	if d.IsModelReady(ctx) {
		fmt.Printf(
			"[llamacpp] model %s ready on %s\n",
			d.cfg.ModelName, d.cfg.Host,
		)
		return nil
	}

	if d.cfg.ModelURL == "" {
		return fmt.Errorf(
			"model not found at %s and no download URL",
			d.cfg.ModelPath,
		)
	}

	fmt.Printf(
		"[llamacpp] downloading %s on %s\n",
		d.cfg.ModelName, d.cfg.Host,
	)
	dlCmd := fmt.Sprintf(
		"mkdir -p $(dirname %s) && curl -L -o %s '%s'",
		d.cfg.ModelPath, d.cfg.ModelPath, d.cfg.ModelURL,
	)
	dlCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	if _, err := d.sshRun(dlCtx, dlCmd); err != nil {
		return fmt.Errorf("download model: %w", err)
	}
	return nil
}

// StartInstance starts a llama-server on the given port.
// Returns the API endpoint URL.
func (d *LlamaCppDeployer) StartInstance(
	ctx context.Context, port int,
) (string, error) {
	endpoint := fmt.Sprintf(
		"http://%s:%d", d.cfg.Host, port,
	)

	// Check if already running.
	if d.isInstanceRunning(ctx, port) {
		fmt.Printf(
			"[llamacpp] instance already running at %s\n",
			endpoint,
		)
		return endpoint, nil
	}

	mmproj := ""
	if d.cfg.MMProjPath != "" {
		mmproj = fmt.Sprintf("--mmproj %s ", d.cfg.MMProjPath)
	}

	// Auto-detect GPU availability: check if nvidia-smi
	// reports >2GB free VRAM. If not, use CPU-only (0 layers).
	gpuLayers := d.cfg.GPULayers
	if gpuLayers == -2 || gpuLayers == -1 {
		gpuLayers = d.probeGPU(ctx)
	}

	cmd := fmt.Sprintf(
		"nohup %s/build/bin/llama-server "+
			"--model %s "+
			"%s"+
			"--host 0.0.0.0 "+
			"--port %d "+
			"--n-gpu-layers %d "+
			"--ctx-size %d "+
			"--threads 8 "+
			"> /tmp/llama-server-%d.log 2>&1 &",
		d.cfg.RepoDir,
		d.cfg.ModelPath,
		mmproj,
		port,
		gpuLayers,
		d.cfg.ContextSize,
		port,
	)
	fmt.Printf(
		"[llamacpp] starting on port %d (gpu_layers=%d)\n",
		port, gpuLayers,
	)

	if _, err := d.sshRun(ctx, cmd); err != nil {
		return "", fmt.Errorf(
			"start llama-server on port %d: %w",
			port, err,
		)
	}

	// Wait for the server to be ready.
	for i := 0; i < 30; i++ {
		if d.isInstanceRunning(ctx, port) {
			fmt.Printf(
				"[llamacpp] instance started at %s\n",
				endpoint,
			)
			return endpoint, nil
		}
		time.Sleep(2 * time.Second)
	}
	return "", fmt.Errorf(
		"llama-server on port %d not ready after 60s",
		port,
	)
}

// StopInstance kills a llama-server on the given port.
func (d *LlamaCppDeployer) StopInstance(
	ctx context.Context, port int,
) {
	cmd := fmt.Sprintf(
		"pkill -f 'llama-server.*--port %d'", port,
	)
	_, _ = d.sshRun(ctx, cmd)
}

// StopAll kills all llama-server instances.
func (d *LlamaCppDeployer) StopAll(ctx context.Context) {
	_, _ = d.sshRun(ctx, "pkill -f llama-server")
}

// isInstanceRunning checks if a llama-server is responding
// on the given port.
func (d *LlamaCppDeployer) isInstanceRunning(
	ctx context.Context, port int,
) bool {
	url := fmt.Sprintf(
		"http://%s:%d/health", d.cfg.Host, port,
	)
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, url, nil,
	)
	if err != nil {
		return false
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()

	// llama-server /health returns 200 when ready.
	return resp.StatusCode == http.StatusOK
}

// probeGPU checks if the remote host has sufficient free GPU
// VRAM (>2GB) to run inference. Returns -1 for full GPU
// offload, 0 for CPU-only.
func (d *LlamaCppDeployer) probeGPU(
	ctx context.Context,
) int {
	out, err := d.sshRun(ctx,
		"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0",
	)
	if err != nil {
		fmt.Println("[llamacpp] no GPU detected, using CPU")
		return 0
	}
	var freeMB int
	fmt.Sscanf(strings.TrimSpace(out), "%d", &freeMB)
	if freeMB > 2000 {
		fmt.Printf(
			"[llamacpp] GPU has %dMB free, using GPU\n",
			freeMB,
		)
		return -1
	}
	fmt.Printf(
		"[llamacpp] GPU has only %dMB free, using CPU\n",
		freeMB,
	)
	return 0
}

// StartRPCServer starts the llama.cpp RPC server on this
// host. The RPC server exposes local compute (GPU/CPU) to
// a central llama-server via the GGML RPC protocol. Requires
// llama.cpp built with -DGGML_RPC=ON.
func (d *LlamaCppDeployer) StartRPCServer(
	ctx context.Context, port int,
) error {
	// Check if RPC server is already running on this port.
	out, _ := d.sshRun(ctx,
		fmt.Sprintf(
			"pgrep -f 'rpc-server.*--port %d' "+
				">/dev/null 2>&1 && echo running",
			port,
		),
	)
	if strings.TrimSpace(out) == "running" {
		fmt.Printf(
			"[llamacpp] RPC server already running "+
				"on %s:%d\n",
			d.cfg.Host, port,
		)
		return nil
	}

	cmd := fmt.Sprintf(
		"cd %s && nohup ./build/bin/rpc-server "+
			"--host 0.0.0.0 --port %d "+
			"> /tmp/rpc-server-%d.log 2>&1 &",
		d.cfg.RepoDir, port, port,
	)
	fmt.Printf(
		"[llamacpp] starting RPC server on %s:%d\n",
		d.cfg.Host, port,
	)
	_, err := d.sshRun(ctx, cmd)
	if err != nil {
		return fmt.Errorf(
			"start rpc-server on %s:%d: %w",
			d.cfg.Host, port, err,
		)
	}

	// Brief wait for the process to bind.
	time.Sleep(2 * time.Second)
	return nil
}

// StopRPCServer kills the RPC server on the given port.
func (d *LlamaCppDeployer) StopRPCServer(
	ctx context.Context, port int,
) {
	cmd := fmt.Sprintf(
		"pkill -f 'rpc-server.*--port %d'", port,
	)
	_, _ = d.sshRun(ctx, cmd)
}

// StartWithRPC starts llama-server configured to distribute
// inference across remote RPC workers. The rpcWorkers slice
// contains "host:port" addresses of running rpc-server
// instances. The model is loaded centrally and layers are
// distributed across workers.
func (d *LlamaCppDeployer) StartWithRPC(
	ctx context.Context,
	modelPath string,
	rpcWorkers []string,
	port int,
) error {
	endpoint := fmt.Sprintf(
		"http://%s:%d", d.cfg.Host, port,
	)

	if d.isInstanceRunning(ctx, port) {
		fmt.Printf(
			"[llamacpp] RPC-backed instance already "+
				"running at %s\n",
			endpoint,
		)
		return nil
	}

	rpcFlag := strings.Join(rpcWorkers, ",")
	cmd := fmt.Sprintf(
		"cd %s && nohup ./build/bin/llama-server "+
			"-m %s "+
			"--rpc %s "+
			"-ngl 99 "+
			"--host 0.0.0.0 "+
			"--port %d "+
			"--ctx-size %d "+
			"> /tmp/llama-server-rpc-%d.log 2>&1 &",
		d.cfg.RepoDir,
		modelPath,
		rpcFlag,
		port,
		d.cfg.ContextSize,
		port,
	)
	fmt.Printf(
		"[llamacpp] starting RPC-backed server on "+
			"port %d with workers: %s\n",
		port, rpcFlag,
	)

	if _, err := d.sshRun(ctx, cmd); err != nil {
		return fmt.Errorf(
			"start llama-server with RPC on port %d: %w",
			port, err,
		)
	}

	// Wait for the server to be ready.
	for i := 0; i < 30; i++ {
		if d.isInstanceRunning(ctx, port) {
			fmt.Printf(
				"[llamacpp] RPC-backed instance "+
					"started at %s\n",
				endpoint,
			)
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf(
		"llama-server (RPC) on port %d not ready "+
			"after 60s",
		port,
	)
}

// EnsureBuiltWithRPC clones and builds llama.cpp with RPC
// support enabled (-DGGML_RPC=ON). Falls back to a non-CUDA
// build if CUDA is not available.
func (d *LlamaCppDeployer) EnsureBuiltWithRPC(
	ctx context.Context,
) error {
	// Check if rpc-server binary already exists.
	out, err := d.sshRun(ctx,
		fmt.Sprintf(
			"test -x %s/build/bin/rpc-server && echo yes",
			d.cfg.RepoDir,
		),
	)
	if err == nil && strings.TrimSpace(out) == "yes" {
		fmt.Printf(
			"[llamacpp] rpc-server already built on %s\n",
			d.cfg.Host,
		)
		return nil
	}

	// Clone if needed.
	repoOut, _ := d.sshRun(ctx,
		fmt.Sprintf("test -d %s/.git && echo yes",
			d.cfg.RepoDir),
	)
	if strings.TrimSpace(repoOut) != "yes" {
		fmt.Printf(
			"[llamacpp] cloning llama.cpp on %s\n",
			d.cfg.Host,
		)
		if _, cloneErr := d.sshRun(ctx,
			fmt.Sprintf(
				"git clone --depth 1 "+
					"https://github.com/ggerganov/"+
					"llama.cpp.git %s",
				d.cfg.RepoDir,
			),
		); cloneErr != nil {
			return fmt.Errorf("clone llama.cpp: %w",
				cloneErr)
		}
	}

	// Build with RPC + CUDA, fallback to RPC-only.
	fmt.Printf(
		"[llamacpp] building llama.cpp with RPC on %s\n",
		d.cfg.Host,
	)
	buildCmd := fmt.Sprintf(
		"cd %s && "+
			"(cmake -B build -DGGML_RPC=ON "+
			"-DGGML_CUDA=ON "+
			"-DCMAKE_BUILD_TYPE=Release 2>&1 || "+
			"cmake -B build -DGGML_RPC=ON "+
			"-DCMAKE_BUILD_TYPE=Release 2>&1) && "+
			"cmake --build build --config Release "+
			"-j$(nproc) 2>&1",
		d.cfg.RepoDir,
	)
	buildCtx, cancel := context.WithTimeout(
		ctx, 10*time.Minute,
	)
	defer cancel()
	if _, buildErr := d.sshRun(
		buildCtx, buildCmd,
	); buildErr != nil {
		return fmt.Errorf(
			"build llama.cpp with RPC: %w", buildErr,
		)
	}

	fmt.Printf(
		"[llamacpp] RPC build complete on %s\n",
		d.cfg.Host,
	)
	return nil
}

// HealthCheck returns the health status of a running instance.
func (d *LlamaCppDeployer) HealthCheck(
	ctx context.Context, port int,
) (map[string]interface{}, error) {
	url := fmt.Sprintf(
		"http://%s:%d/health", d.cfg.Host, port,
	)
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, url, nil,
	)
	if err != nil {
		return nil, err
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(
		&result,
	); err != nil {
		return nil, err
	}
	return result, nil
}
