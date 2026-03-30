// SPDX-FileCopyrightText: 2026 Milos Vasic
// SPDX-License-Identifier: Apache-2.0

package remote_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"digital.vasic.visionengine/pkg/remote"
)

func TestNewVisionPool_Defaults(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{})
	require.NotNil(t, pool)
	assert.Equal(t, 0, pool.Size())
}

func TestNewVisionPool_WithConfig(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{
		Host:             "thinker.local",
		User:             "admin",
		Model:            "llava:7b",
		InferenceBackend: remote.BackendOllama,
		BasePort:         9000,
	})
	require.NotNil(t, pool)
}

func TestVisionPool_EnsureReady_EmptyHost(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{})
	err := pool.EnsureReady(context.Background())
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "host is required")
}

func TestVisionPool_EnsureReady_LlamaCppMissingConfig(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{
		Host:             "thinker.local",
		InferenceBackend: remote.BackendLlamaCpp,
	})
	err := pool.EnsureReady(context.Background())
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "llama.cpp config required")
}

func TestVisionPool_EnsureReady_LlamaCppValid(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{
		Host:             "thinker.local",
		InferenceBackend: remote.BackendLlamaCpp,
		LlamaCpp: &remote.LlamaCppConfig{
			Host:      "thinker.local",
			ModelPath: "/models/llava.gguf",
		},
	})
	err := pool.EnsureReady(context.Background())
	assert.NoError(t, err)
}

func TestVisionPool_AssignSlots_Shared(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{
		Host:     "thinker.local",
		Shared:   true,
		BasePort: 8080,
	})

	targets := []remote.SlotTarget{
		{Platform: "android", Device: "device1"},
		{Platform: "android", Device: "device2"},
		{Platform: "web"},
	}
	pool.AssignSlots(targets)

	// All targets should share the same slot endpoint.
	s1 := pool.GetSlot("android", "device1")
	s2 := pool.GetSlot("android", "device2")
	s3 := pool.GetSlot("web", "")
	require.NotNil(t, s1)
	require.NotNil(t, s2)
	require.NotNil(t, s3)

	assert.Equal(t, s1.Endpoint, s2.Endpoint,
		"shared pool: all slots should have same endpoint")
	assert.Equal(t, s1.Endpoint, s3.Endpoint)
	assert.Equal(t, 8080, s1.Port)
}

func TestVisionPool_AssignSlots_Dedicated(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{
		Host:     "thinker.local",
		Shared:   false,
		BasePort: 9000,
	})

	targets := []remote.SlotTarget{
		{Platform: "android", Device: "device1"},
		{Platform: "android", Device: "device2"},
		{Platform: "web"},
	}
	pool.AssignSlots(targets)

	assert.Equal(t, 3, pool.Size())

	s1 := pool.GetSlot("android", "device1")
	s2 := pool.GetSlot("android", "device2")
	s3 := pool.GetSlot("web", "")
	require.NotNil(t, s1)
	require.NotNil(t, s2)
	require.NotNil(t, s3)

	assert.NotEqual(t, s1.Endpoint, s2.Endpoint,
		"dedicated pool: each slot should have different endpoint")
	assert.Equal(t, 9000, s1.Port)
	assert.Equal(t, 9001, s2.Port)
	assert.Equal(t, 9002, s3.Port)
}

func TestVisionPool_GetSlot_NotAssigned(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{
		Host:     "thinker.local",
		BasePort: 8080,
	})
	slot := pool.GetSlot("nonexistent", "")
	assert.Nil(t, slot)
}

func TestVisionPool_Shutdown(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{
		Host:     "thinker.local",
		BasePort: 8080,
	})
	pool.AssignSlots([]remote.SlotTarget{
		{Platform: "android"},
	})
	assert.Equal(t, 1, pool.Size())

	pool.Shutdown(context.Background())
	assert.Equal(t, 0, pool.Size())
}

func TestVisionSlot_LockUnlock(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{
		Host:     "thinker.local",
		BasePort: 8080,
	})
	pool.AssignSlots([]remote.SlotTarget{
		{Platform: "android", Device: "dev1"},
	})

	slot := pool.GetSlot("android", "dev1")
	require.NotNil(t, slot)

	// Lock/unlock should not deadlock.
	slot.Lock()
	slot.Unlock()
}

func TestVisionSlot_RecordCall(t *testing.T) {
	pool := remote.NewVisionPool(remote.PoolConfig{
		Host:     "thinker.local",
		BasePort: 8080,
	})
	pool.AssignSlots([]remote.SlotTarget{
		{Platform: "web"},
	})

	slot := pool.GetSlot("web", "")
	require.NotNil(t, slot)

	slot.RecordCall(100*time.Millisecond, nil)
	slot.RecordCall(200*time.Millisecond, assert.AnError)

	calls, totalTime, errors := slot.Stats()
	assert.Equal(t, 2, calls)
	assert.Equal(t, 300*time.Millisecond, totalTime)
	assert.Equal(t, 1, errors)
}

func TestNewLlamaCppDeployer(t *testing.T) {
	deployer := remote.NewLlamaCppDeployer(remote.LlamaCppConfig{
		Host:        "thinker.local",
		User:        "admin",
		ModelPath:   "/models/llava.gguf",
		MMProjPath:  "/models/mmproj.gguf",
		BasePort:    8080,
		GPULayers:   -1,
		ContextSize: 4096,
	})
	require.NotNil(t, deployer)
}

func TestBackendConstants(t *testing.T) {
	assert.Equal(t, "ollama", remote.BackendOllama)
	assert.Equal(t, "llama-cpp", remote.BackendLlamaCpp)
}

func TestSlotTarget_Fields(t *testing.T) {
	target := remote.SlotTarget{
		Platform: "android",
		Device:   "emulator-5554",
	}
	assert.Equal(t, "android", target.Platform)
	assert.Equal(t, "emulator-5554", target.Device)
}

func TestPoolConfig_AllFields(t *testing.T) {
	cfg := remote.PoolConfig{
		Host:             "gpu-host",
		User:             "user",
		Model:            "model",
		Shared:           true,
		InferenceBackend: remote.BackendLlamaCpp,
		BasePort:         9000,
		LlamaCpp: &remote.LlamaCppConfig{
			Host:        "gpu-host",
			User:        "user",
			RepoDir:     "~/llama.cpp",
			ModelPath:   "/models/model.gguf",
			MMProjPath:  "/models/proj.gguf",
			BasePort:    9000,
			GPULayers:   32,
			ContextSize: 2048,
		},
	}
	assert.Equal(t, "gpu-host", cfg.Host)
	assert.Equal(t, remote.BackendLlamaCpp, cfg.InferenceBackend)
	assert.NotNil(t, cfg.LlamaCpp)
	assert.Equal(t, 32, cfg.LlamaCpp.GPULayers)
}
