# Video Course: VisionEngine

## Episode 1: Two-Layer Analysis Pipeline (12 min)
1. Architecture: GoCV (mechanical) + LLM Vision (intelligent)
2. Why two layers: cost, speed, accuracy tradeoffs
3. Data flow: screenshot → GoCV bounding boxes → LLM context → ScreenAnalysis
4. Demo: analyzing a sample UI screenshot

## Episode 2: GoCV Operations Deep Dive (15 min)
1. SSIM screenshot diffing and change masks
2. Edge detection (Canny) for UI element bounds
3. Contour detection for element bounding boxes
4. Color analysis: dominant colors, contrast ratios
5. Build tags: `//go:build vision` and stub pattern
6. Demo: detecting UI changes between screenshots

## Episode 3: LLM Vision Providers (12 min)
1. VisionProvider interface: AnalyzeImage, CompareImages
2. OpenAI GPT-4o adapter: base64 image encoding, prompt structure
3. Anthropic Claude adapter: messages API with image content
4. Gemini adapter: multimodal content parts
5. Qwen-VL adapter: vision-language model
6. FallbackProvider: score-ranked provider chain
7. Demo: comparing provider responses for same screenshot

## Episode 4: Navigation Graph Building (15 min)
1. NavigationGraph interface: screens as nodes, actions as edges
2. Adding screens with visual similarity hashing
3. BFS pathfinding: PathTo() shortest path
4. Coverage tracking: visited vs discovered screens
5. Export formats: DOT (Graphviz), JSON, Mermaid
6. Thread safety with sync.RWMutex
7. Demo: building a navigation graph from app exploration

## Episode 5: Video Frame Extraction (10 min)
1. VideoProcessor interface: frame extraction, scene changes
2. Key frame detection at screen transitions
3. Thumbnail generation for timeline
4. Integration with SessionRecorder for timestamp linking
5. Demo: extracting key frames from a test session recording
