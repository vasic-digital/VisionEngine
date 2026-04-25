// Copyright 2026 Milos Vasic. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package graph

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAutomation_ModulePath(t *testing.T) {
	// Verify the Go module path is correct
	goMod, err := os.ReadFile(findGoMod(t))
	require.NoError(t, err)
	assert.Contains(t, string(goMod), "module digital.vasic.visionengine")
}

func TestAutomation_GoVersion(t *testing.T) {
	goMod, err := os.ReadFile(findGoMod(t))
	require.NoError(t, err)
	assert.Contains(t, string(goMod), "go 1.2") // matches 1.24.x
}

func TestAutomation_PackageCompiles(t *testing.T) {
	// Verify this package compiles (test is running, so it does)
	assert.True(t, true, "Package compiled successfully")
}

func TestAutomation_GoVet(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping go vet in short mode")  // SKIP-OK: #short-mode
	}
	cmd := exec.Command("go", "vet", "./...")
	cmd.Dir = findProjectRoot(t)
	output, err := cmd.CombinedOutput()
	assert.NoError(t, err, "go vet failed: %s", string(output))
}

func TestAutomation_GraphSnapshotSerializable(t *testing.T) {
	// Verify GraphSnapshot can round-trip through JSON
	original := GraphSnapshot{
		Screens: []ScreenNode{
			{ID: "test", Visited: true},
		},
		Transitions: []Transition{
			{From: "a", To: "b"},
		},
		Current:  "test",
		Coverage: 1.0,
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded GraphSnapshot
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.Current, decoded.Current)
	assert.Equal(t, original.Coverage, decoded.Coverage)
	assert.Len(t, decoded.Screens, 1)
	assert.Len(t, decoded.Transitions, 1)
}

func TestAutomation_ScreenNodeSerializable(t *testing.T) {
	node := ScreenNode{
		ID:      "test-screen",
		Visited: true,
	}

	data, err := json.Marshal(node)
	require.NoError(t, err)

	var decoded ScreenNode
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, node.ID, decoded.ID)
	assert.Equal(t, node.Visited, decoded.Visited)
}

func TestAutomation_TransitionSerializable(t *testing.T) {
	trans := Transition{
		From: "screen-a",
		To:   "screen-b",
	}

	data, err := json.Marshal(trans)
	require.NoError(t, err)

	var decoded Transition
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, trans.From, decoded.From)
	assert.Equal(t, trans.To, decoded.To)
}

func TestAutomation_InterfaceDefinition(t *testing.T) {
	// Verify NavigationGraph interface is properly implemented
	var g NavigationGraph = NewNavigationGraph()
	require.NotNil(t, g)

	// All methods should be callable without panic
	_ = g.CurrentScreen()
	_ = g.Screens()
	_ = g.Transitions()
	_ = g.UnvisitedScreens()
	_ = g.Coverage()
	_ = g.Export()
}

func TestAutomation_ErrorTypes(t *testing.T) {
	// Verify all error types are properly defined
	assert.NotNil(t, ErrScreenNotFound)
	assert.NotNil(t, ErrNoPath)
	assert.NotNil(t, ErrEmptyGraph)
	assert.NotNil(t, ErrSelfTransition)
	assert.NotNil(t, ErrDuplicateScreen)

	// Verify error messages are descriptive
	assert.Contains(t, ErrScreenNotFound.Error(), "screen")
	assert.Contains(t, ErrNoPath.Error(), "path")
	assert.Contains(t, ErrEmptyGraph.Error(), "empty")
}

func TestAutomation_RuntimeInfo(t *testing.T) {
	// Verify we're running on a supported platform
	assert.NotEmpty(t, runtime.GOOS)
	assert.NotEmpty(t, runtime.GOARCH)
	t.Logf("Running on %s/%s with Go %s", runtime.GOOS, runtime.GOARCH, runtime.Version())
}

func TestAutomation_ExportFunctionsExist(t *testing.T) {
	g := NewNavigationGraph()

	// Verify all export functions exist and return non-error values
	dot := ExportDOT(g)
	assert.NotEmpty(t, dot)

	jsonStr, err := ExportJSON(g)
	assert.NoError(t, err)
	assert.NotEmpty(t, jsonStr)

	mermaid := ExportMermaid(g)
	assert.NotEmpty(t, mermaid)
}

// Helper to find go.mod
func findGoMod(t *testing.T) string {
	t.Helper()
	root := findProjectRoot(t)
	return filepath.Join(root, "go.mod")
}

// Helper to find project root
func findProjectRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	require.NoError(t, err)

	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find project root (go.mod)")
		}
		dir = parent
	}
}
