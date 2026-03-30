// Copyright 2026 Milos Vasic. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package remote provides automatic deployment and lifecycle
// management of Ollama vision models on remote hosts via SSH.

package remote

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// VisionSlot represents a dedicated vision inference slot
// assigned to one platform/device. Each slot queues requests
// sequentially so one slow inference doesn't block other
// platforms' vision calls.
type VisionSlot struct {
	// ID is the slot identifier (e.g. "androidtv-192.168.0.134:5555").
	ID string

	// Platform is the target platform (e.g. "androidtv", "web").
	Platform string

	// Device is the device identifier (e.g. ADB serial or URL).
	Device string

	// Endpoint is the Ollama API URL this slot uses.
	Endpoint string

	// Port is the Ollama port for this slot.
	Port int

	// mu serializes vision calls for this slot.
	mu sync.Mutex

	// stats tracks slot usage.
	stats SlotStats
}

// SlotStats tracks usage metrics for a vision slot.
type SlotStats struct {
	TotalCalls    int           `json:"total_calls"`
	TotalDuration time.Duration `json:"total_duration"`
	LastCallAt    time.Time     `json:"last_call_at,omitempty"`
	Errors        int           `json:"errors"`
}

// Lock acquires exclusive access to this slot. Each platform's
// curiosity phase should hold the lock during its vision call
// to prevent contention.
func (s *VisionSlot) Lock() {
	s.mu.Lock()
}

// Unlock releases the slot.
func (s *VisionSlot) Unlock() {
	s.mu.Unlock()
}

// RecordCall updates slot statistics after a vision call.
func (s *VisionSlot) RecordCall(dur time.Duration, err error) {
	s.stats.TotalCalls++
	s.stats.TotalDuration += dur
	s.stats.LastCallAt = time.Now()
	if err != nil {
		s.stats.Errors++
	}
}

// Stats returns a copy of the slot's usage statistics.
func (s *VisionSlot) Stats() SlotStats {
	return s.stats
}

// Backend selects the inference server type.
type Backend string

const (
	// BackendOllama uses Ollama (default).
	BackendOllama Backend = "ollama"
	// BackendLlamaCpp uses llama-server from llama.cpp.
	BackendLlamaCpp Backend = "llamacpp"
)

// HostConfig describes a single host in a multi-host vision
// pool. Each host can run Ollama or llama.cpp for vision
// inference.
type HostConfig struct {
	// Host is the hostname or IP address.
	Host string
	// User is the SSH user for deployment.
	User string
	// Port is the SSH port (default 22).
	Port int
	// Model is the vision model for this host.
	// Defaults to the pool-level model if empty.
	Model string
	// Backend selects "ollama" or "llamacpp".
	// Defaults to "ollama".
	Backend Backend
	// APIPort is the Ollama or llama-server port.
	// Defaults to 11434 for Ollama, 8090 for llama.cpp.
	APIPort int
}

// PoolConfig configures a VisionPool.
type PoolConfig struct {
	// Host is the remote server hostname.
	Host string

	// User is the SSH user for deployment.
	User string

	// BasePort is the starting port for instances.
	// Ollama default: 11434. LlamaCpp default: 8090.
	BasePort int

	// Model is the vision model to use.
	Model string

	// Shared indicates all slots use one instance (true) or
	// each slot gets a dedicated port (false). Shared mode
	// is the default for Ollama. LlamaCpp always uses
	// dedicated mode (one process per port).
	Shared bool

	// InferenceBackend selects ollama or llamacpp.
	// Default: ollama.
	InferenceBackend Backend

	// LlamaCpp holds llama.cpp-specific configuration.
	// Only used when InferenceBackend == BackendLlamaCpp.
	LlamaCpp *LlamaCppConfig

	// Hosts lists multiple hosts for distributed vision
	// inference. When non-empty, slots are distributed
	// round-robin across hosts. If one host fails during
	// EnsureReady, it is removed and remaining hosts are
	// used. Takes precedence over the single Host field.
	Hosts []HostConfig
}

// VisionPool manages a set of VisionSlots, one per
// platform/device being tested. It ensures the backend
// (Ollama or llama.cpp) is ready and assigns dedicated
// slots to each QA target.
type VisionPool struct {
	cfg          PoolConfig
	deployer     *Deployer
	llamaDeployer *LlamaCppDeployer
	slots        map[string]*VisionSlot
	mu           sync.Mutex
}

// NewVisionPool creates a pool backed by the given config.
func NewVisionPool(cfg PoolConfig) *VisionPool {
	if cfg.BasePort == 0 {
		if cfg.InferenceBackend == BackendLlamaCpp {
			cfg.BasePort = 8090
		} else {
			cfg.BasePort = 11434
		}
	}
	if cfg.Model == "" {
		cfg.Model = "llava:7b"
	}

	p := &VisionPool{
		cfg:   cfg,
		slots: make(map[string]*VisionSlot),
	}

	if cfg.InferenceBackend == BackendLlamaCpp &&
		cfg.LlamaCpp != nil {
		p.llamaDeployer = NewLlamaCppDeployer(*cfg.LlamaCpp)
	} else {
		p.deployer = NewDeployer(Config{
			Host:       cfg.Host,
			User:       cfg.User,
			Model:      cfg.Model,
			OllamaPort: cfg.BasePort,
		})
	}
	return p
}

// EnsureReady verifies the backend is running and the model
// is available. For llama.cpp, it builds the binary and
// downloads the model if needed.
func (p *VisionPool) EnsureReady(
	ctx context.Context,
) error {
	if p.llamaDeployer != nil {
		// llama.cpp backend: build + model.
		if err := p.llamaDeployer.EnsureBuilt(ctx); err != nil {
			return fmt.Errorf("vision pool: %w", err)
		}
		if err := p.llamaDeployer.EnsureModel(ctx); err != nil {
			return fmt.Errorf("vision pool: %w", err)
		}
		fmt.Printf(
			"[vision-pool] llama.cpp backend ready on %s\n",
			p.cfg.Host,
		)
		return nil
	}

	// Ollama backend.
	endpoint, err := p.deployer.EnsureReady(ctx)
	if err != nil {
		return fmt.Errorf("vision pool: %w", err)
	}
	fmt.Printf(
		"[vision-pool] ollama backend ready at %s\n",
		endpoint,
	)
	return nil
}

// SlotTarget describes a QA target that needs a vision slot.
type SlotTarget struct {
	Platform string // e.g. "androidtv", "web", "api"
	Device   string // e.g. "192.168.0.134:5555", "localhost:3000"
}

// AssignSlots creates dedicated VisionSlots for each target.
// In shared mode (Ollama), all slots share one endpoint.
// In dedicated mode (llama.cpp), each slot gets its own
// llama-server instance on a dedicated port.
func (p *VisionPool) AssignSlots(
	targets []SlotTarget,
) []*VisionSlot {
	p.mu.Lock()
	defer p.mu.Unlock()

	var result []*VisionSlot
	for i, t := range targets {
		id := fmt.Sprintf("%s-%s", t.Platform, t.Device)
		if id == fmt.Sprintf("%s-", t.Platform) {
			id = fmt.Sprintf("%s-%d", t.Platform, i)
		}

		port := p.cfg.BasePort
		if !p.cfg.Shared {
			port = p.cfg.BasePort + i
		}

		endpoint := fmt.Sprintf(
			"http://%s:%d", p.cfg.Host, port,
		)

		// For llama.cpp dedicated mode, start a server
		// instance on this port.
		if p.llamaDeployer != nil && !p.cfg.Shared {
			started, err := p.llamaDeployer.StartInstance(
				context.Background(), port,
			)
			if err != nil {
				fmt.Printf(
					"[vision-pool] WARNING: failed "+
						"to start llama-server on "+
						"port %d: %v\n",
					port, err,
				)
			} else {
				endpoint = started
			}
		}

		slot := &VisionSlot{
			ID:       id,
			Platform: t.Platform,
			Device:   t.Device,
			Endpoint: endpoint,
			Port:     port,
		}
		p.slots[id] = slot
		result = append(result, slot)

		fmt.Printf(
			"[vision-pool] slot %s -> %s\n",
			id, endpoint,
		)
	}
	return result
}

// GetSlot returns the slot for the given platform and device.
func (p *VisionPool) GetSlot(
	platform, device string,
) *VisionSlot {
	p.mu.Lock()
	defer p.mu.Unlock()

	id := fmt.Sprintf("%s-%s", platform, device)
	if slot, ok := p.slots[id]; ok {
		return slot
	}
	// Fallback: find by platform only.
	for _, slot := range p.slots {
		if slot.Platform == platform {
			return slot
		}
	}
	return nil
}

// AllSlots returns all assigned slots.
func (p *VisionPool) AllSlots() []*VisionSlot {
	p.mu.Lock()
	defer p.mu.Unlock()

	result := make([]*VisionSlot, 0, len(p.slots))
	for _, s := range p.slots {
		result = append(result, s)
	}
	return result
}

// Size returns the number of assigned slots.
func (p *VisionPool) Size() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.slots)
}

// PrintStats logs per-slot usage statistics.
func (p *VisionPool) PrintStats() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, s := range p.slots {
		st := s.Stats()
		avg := time.Duration(0)
		if st.TotalCalls > 0 {
			avg = st.TotalDuration / time.Duration(
				st.TotalCalls,
			)
		}
		fmt.Printf(
			"[vision-pool] %s: %d calls, "+
				"avg %v, %d errors\n",
			s.ID, st.TotalCalls, avg.Round(
				time.Millisecond,
			), st.Errors,
		)
	}
}

// Shutdown stops any dedicated instances started by the pool.
// In shared Ollama mode, this is a no-op. In llama.cpp
// dedicated mode, all llama-server processes are stopped.
func (p *VisionPool) Shutdown(ctx context.Context) {
	p.PrintStats()

	if p.cfg.Shared && p.llamaDeployer == nil {
		return
	}

	if p.llamaDeployer != nil {
		// Stop all llama-server instances we started.
		for _, slot := range p.AllSlots() {
			p.llamaDeployer.StopInstance(ctx, slot.Port)
		}
		fmt.Println("[vision-pool] all llama-server instances stopped")
		return
	}

	// Ollama dedicated mode.
	if p.deployer != nil {
		for _, slot := range p.AllSlots() {
			if slot.Port != p.cfg.BasePort {
				cmd := fmt.Sprintf(
					"pkill -f 'ollama.*%d'",
					slot.Port,
				)
				_, _ = p.deployer.sshRun(ctx, cmd)
			}
		}
	}
}

// hostEntry tracks a live host in a MultiHostPool along
// with its deployer and readiness state.
type hostEntry struct {
	cfg      HostConfig
	deployer *Deployer
	ready    bool
}

// MultiHostPool distributes vision inference slots across
// multiple hosts. It creates an Ollama deployer per host
// and assigns slots round-robin. If a host fails during
// EnsureReady, it is removed and the remaining hosts
// absorb its share of the work.
type MultiHostPool struct {
	hosts []*hostEntry
	model string
	slots map[string]*VisionSlot
	mu    sync.Mutex
}

// NewMultiHostPool creates a pool that distributes vision
// work across the given hosts. Each host gets its own
// Ollama deployer. The model parameter is used as a default
// for hosts that do not specify their own model.
func NewMultiHostPool(
	hosts []HostConfig, model string,
) *MultiHostPool {
	if model == "" {
		model = "llava:7b"
	}

	entries := make([]*hostEntry, 0, len(hosts))
	for _, h := range hosts {
		if h.Host == "" {
			continue
		}
		if h.Port == 0 {
			h.Port = 22
		}
		if h.APIPort == 0 {
			if h.Backend == BackendLlamaCpp {
				h.APIPort = 8090
			} else {
				h.APIPort = 11434
			}
		}
		m := h.Model
		if m == "" {
			m = model
		}
		deployer := NewDeployer(Config{
			Host:       h.Host,
			User:       h.User,
			Port:       h.Port,
			Model:      m,
			OllamaPort: h.APIPort,
		})
		entries = append(entries, &hostEntry{
			cfg:      h,
			deployer: deployer,
		})
	}

	return &MultiHostPool{
		hosts: entries,
		model: model,
		slots: make(map[string]*VisionSlot),
	}
}

// EnsureReady verifies each host's Ollama backend is
// running and the model is pulled. Hosts that fail are
// marked not-ready and excluded from slot assignment.
// Returns an error only if ALL hosts fail.
func (mp *MultiHostPool) EnsureReady(
	ctx context.Context,
) error {
	var readyCount int
	for _, h := range mp.hosts {
		endpoint, err := h.deployer.EnsureReady(ctx)
		if err != nil {
			fmt.Printf(
				"[multi-host] %s failed: %v "+
					"(excluding from pool)\n",
				h.cfg.Host, err,
			)
			h.ready = false
			continue
		}
		h.ready = true
		readyCount++
		fmt.Printf(
			"[multi-host] %s ready at %s\n",
			h.cfg.Host, endpoint,
		)
	}
	if readyCount == 0 {
		return fmt.Errorf(
			"multi-host pool: all %d hosts failed",
			len(mp.hosts),
		)
	}
	fmt.Printf(
		"[multi-host] %d/%d hosts ready\n",
		readyCount, len(mp.hosts),
	)
	return nil
}

// readyHosts returns only the hosts that passed
// EnsureReady.
func (mp *MultiHostPool) readyHosts() []*hostEntry {
	var ready []*hostEntry
	for _, h := range mp.hosts {
		if h.ready {
			ready = append(ready, h)
		}
	}
	return ready
}

// AssignSlots distributes targets round-robin across ready
// hosts. Each slot's endpoint points to the assigned host's
// Ollama API.
func (mp *MultiHostPool) AssignSlots(
	targets []SlotTarget,
) []*VisionSlot {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	ready := mp.readyHosts()
	if len(ready) == 0 {
		fmt.Println(
			"[multi-host] no ready hosts for " +
				"slot assignment",
		)
		return nil
	}

	var result []*VisionSlot
	for i, t := range targets {
		host := ready[i%len(ready)]

		id := fmt.Sprintf("%s-%s", t.Platform, t.Device)
		if id == fmt.Sprintf("%s-", t.Platform) {
			id = fmt.Sprintf("%s-%d", t.Platform, i)
		}

		endpoint := fmt.Sprintf(
			"http://%s:%d",
			host.cfg.Host, host.cfg.APIPort,
		)

		slot := &VisionSlot{
			ID:       id,
			Platform: t.Platform,
			Device:   t.Device,
			Endpoint: endpoint,
			Port:     host.cfg.APIPort,
		}
		mp.slots[id] = slot
		result = append(result, slot)

		fmt.Printf(
			"[multi-host] slot %s -> %s (%s)\n",
			id, endpoint, host.cfg.Host,
		)
	}
	return result
}

// GetSlot returns the slot for the given platform and
// device.
func (mp *MultiHostPool) GetSlot(
	platform, device string,
) *VisionSlot {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	id := fmt.Sprintf("%s-%s", platform, device)
	if slot, ok := mp.slots[id]; ok {
		return slot
	}
	for _, slot := range mp.slots {
		if slot.Platform == platform {
			return slot
		}
	}
	return nil
}

// AllSlots returns all assigned slots.
func (mp *MultiHostPool) AllSlots() []*VisionSlot {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	result := make([]*VisionSlot, 0, len(mp.slots))
	for _, s := range mp.slots {
		result = append(result, s)
	}
	return result
}

// Size returns the number of assigned slots.
func (mp *MultiHostPool) Size() int {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	return len(mp.slots)
}

// ReadyHostCount returns how many hosts are available for
// inference.
func (mp *MultiHostPool) ReadyHostCount() int {
	return len(mp.readyHosts())
}

// PrintStats logs per-slot usage statistics.
func (mp *MultiHostPool) PrintStats() {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	for _, s := range mp.slots {
		st := s.Stats()
		avg := time.Duration(0)
		if st.TotalCalls > 0 {
			avg = st.TotalDuration / time.Duration(
				st.TotalCalls,
			)
		}
		fmt.Printf(
			"[multi-host] %s: %d calls, "+
				"avg %v, %d errors\n",
			s.ID, st.TotalCalls, avg.Round(
				time.Millisecond,
			), st.Errors,
		)
	}
}

// Shutdown prints statistics. Ollama instances are left
// running (they are system services, not started by us
// in shared mode).
func (mp *MultiHostPool) Shutdown(
	ctx context.Context,
) {
	mp.PrintStats()
}
