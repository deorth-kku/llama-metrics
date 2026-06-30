package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// ---------- test fixtures ----------

const sampleModelsJSON = `{
  "data": [
    {"id": "Qwen3.6-35B-A3B-Claude-4.7-Opus.gguf", "status": {"value": "unloaded"}},
    {"id": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf", "status": {"value": "unloaded"}},
    {"id": "deepreinforce-ai_Ornith-1.0-35B-IQ4_XS", "status": {"value": "loaded"}},
    {"id": "default", "status": {"value": "unloaded"}},
    {"id": "gemma4-E4B", "status": {"value": "unloaded"}}
  ],
  "object": "list"
}`

const sampleMetrics = `# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 0
# HELP llamacpp:prompt_seconds_total Prompt process time
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 0
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 0
# HELP llamacpp:tokens_predicted_seconds_total Predict process time
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 0
# HELP llamacpp:n_decode_total Total number of llama_decode() calls
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 0
# HELP llamacpp:n_tokens_max Largest observed n_tokens.
# TYPE llamacpp:n_tokens_max counter
llamacpp:n_tokens_max 0
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 0
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 0
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 0
# HELP llamacpp:requests_deferred Number of requests deferred.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
# HELP llamacpp:n_busy_slots_per_decode Average number of busy slots per llama_decode() call
# TYPE llamacpp:n_busy_slots_per_decode gauge
llamacpp:n_busy_slots_per_decode 0
`

const allUnloadedModelsJSON = `{
  "data": [
    {"id": "model-a", "status": {"value": "unloaded"}},
    {"id": "model-b", "status": {"value": "unloaded"}}
  ],
  "object": "list"
}`

// ---------- helpers ----------

// metricsCounter wraps a handler and counts requests to a specific path.
type metricsCounter struct {
	handler          http.Handler
	metricsPathCalls int64
	modelsPathCalls  int64
}

func newMetricsCounter(handler http.Handler) *metricsCounter {
	mc := &metricsCounter{handler: handler}
	return mc
}

func (mc *metricsCounter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/metrics" {
		atomic.AddInt64(&mc.metricsPathCalls, 1)
	}
	if r.URL.Path == "/v1/models" {
		atomic.AddInt64(&mc.modelsPathCalls, 1)
	}
	mc.handler.ServeHTTP(w, r)
}

// ---------- Test 1: discoverLoadedModels ----------

func TestDiscoverLoadedModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sampleModelsJSON))
	}))
	defer srv.Close()

	loaded, err := discoverLoadedModels(context.Background(), new(http.Client), srv.URL)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(loaded) != 1 {
		t.Fatalf("expected 1 loaded model, got %d: %v", len(loaded), loaded)
	}
	if loaded[0] != "deepreinforce-ai_Ornith-1.0-35B-IQ4_XS" {
		t.Errorf("expected model ID 'deepreinforce-ai_Ornith-1.0-35B-IQ4_XS', got %q", loaded[0])
	}
}

// ---------- Test 2: addModelLabel ----------

func TestAddModelLabel(t *testing.T) {
	result := addModelLabel(sampleMetrics, "test-model")

	// Comment lines should pass through unchanged
	if !strings.Contains(result, "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.") {
		t.Error("HELP line should be unchanged")
	}
	if !strings.Contains(result, "# TYPE llamacpp:prompt_tokens_total counter") {
		t.Error("TYPE line should be unchanged")
	}

	// Data lines should have model label
	if !strings.Contains(result, `llamacpp:prompt_tokens_total{model="test-model"} 0`) {
		t.Error("data line should have model label")
	}
	if !strings.Contains(result, `llamacpp:requests_processing{model="test-model"} 0`) {
		t.Error("requests_processing line should have model label")
	}

	// Should NOT have unlabeled data lines
	lines := strings.Split(result, "\n")
	for _, line := range lines {
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.Contains(line, "llamacpp:") && !strings.Contains(line, `model="test-model"`) {
			t.Errorf("unlabeled data line found: %q", line)
		}
	}
}

// ---------- Test 3: No loaded models ----------

func TestAggregatedHandler_NoLoadedModels(t *testing.T) {
	// Dummy upstream: all unloaded, /metrics returns 404 for any model
	dummy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(allUnloadedModelsJSON))
			return
		}
		if r.URL.Path == "/metrics" {
			t.Error("/metrics should NOT be called when no models are loaded")
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer dummy.Close()

	r := newRouter(dummy.URL)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	body, _ := io.ReadAll(w.Body)
	if len(body) != 0 {
		t.Errorf("expected empty body, got %q", string(body))
	}
}

// ---------- Test 4: Single loaded model ----------

func TestAggregatedHandler_SingleLoadedModel(t *testing.T) {
	dummy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(sampleModelsJSON))
			return
		}
		if r.URL.Path == "/metrics" {
			model := r.URL.Query().Get("model")
			if model == "deepreinforce-ai_Ornith-1.0-35B-IQ4_XS" {
				w.WriteHeader(http.StatusOK)
				w.Write([]byte(sampleMetrics))
				return
			}
			t.Errorf("unexpected model parameter: %q", model)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer dummy.Close()

	r := newRouter(dummy.URL)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	body := w.Body.String()

	// Should contain labeled metrics
	if !strings.Contains(body, `llamacpp:prompt_tokens_total{model="deepreinforce-ai_Ornith-1.0-35B-IQ4_XS"} 0`) {
		t.Error("should contain labeled metric for loaded model")
	}

	// HELP/TYPE should be present
	if !strings.Contains(body, "# HELP llamacpp:prompt_tokens_total") {
		t.Error("HELP line should be present")
	}
	if !strings.Contains(body, "# TYPE llamacpp:prompt_tokens_total counter") {
		t.Error("TYPE line should be present")
	}

	// Should contain router_models_loaded gauge
	if !strings.Contains(body, "llamacpp_router_models_loaded 1") {
		t.Error("should contain router_models_loaded gauge")
	}

	// Should NOT contain any unloaded model metrics
	if strings.Contains(body, `model="Qwen3.6-35B-A3B-Claude-4.7-Opus.gguf"`) {
		t.Error("should NOT contain unloaded model metrics")
	}
}

// ---------- Test 5: Multiple loaded models ----------

func TestAggregatedHandler_MultipleLoadedModels(t *testing.T) {
	multiModelsJSON := `{
  "data": [
    {"id": "model-a", "status": {"value": "loaded"}},
    {"id": "model-b", "status": {"value": "loaded"}}
  ],
  "object": "list"
}`

	metricsA := `# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 100
llamacpp:requests_processing 5
`
	metricsB := `# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 200
llamacpp:requests_processing 3
`

	dummy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(multiModelsJSON))
			return
		}
		if r.URL.Path == "/metrics" {
			model := r.URL.Query().Get("model")
			switch model {
			case "model-a":
				w.WriteHeader(http.StatusOK)
				w.Write([]byte(metricsA))
			case "model-b":
				w.WriteHeader(http.StatusOK)
				w.Write([]byte(metricsB))
			default:
				w.WriteHeader(http.StatusNotFound)
			}
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer dummy.Close()

	r := newRouter(dummy.URL)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	body := w.Body.String()

	// Both models should be present
	if !strings.Contains(body, `llamacpp:prompt_tokens_total{model="model-a"} 100`) {
		t.Error("should contain model-a metric")
	}
	if !strings.Contains(body, `llamacpp:prompt_tokens_total{model="model-b"} 200`) {
		t.Error("should contain model-b metric")
	}
	if !strings.Contains(body, `llamacpp:requests_processing{model="model-a"} 5`) {
		t.Error("should contain model-a requests_processing")
	}
	if !strings.Contains(body, `llamacpp:requests_processing{model="model-b"} 3`) {
		t.Error("should contain model-b requests_processing")
	}

	// HELP/TYPE should appear only once (from first model)
	helpCount := strings.Count(body, "# HELP llamacpp:prompt_tokens_total")
	if helpCount != 1 {
		t.Errorf("HELP line should appear once, got %d", helpCount)
	}

	// router_models_loaded should be 2
	if !strings.Contains(body, "llamacpp_router_models_loaded 2") {
		t.Error("router_models_loaded should be 2")
	}
}

// ---------- Test 6: Upstream error for one model ----------

func TestAggregatedHandler_UpstreamError(t *testing.T) {
	errorModelsJSON := `{
  "data": [
    {"id": "model-ok", "status": {"value": "loaded"}},
    {"id": "model-broken", "status": {"value": "loaded"}}
  ],
  "object": "list"
}`

	metricsOK := `# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 42
`

	dummy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(errorModelsJSON))
			return
		}
		if r.URL.Path == "/metrics" {
			model := r.URL.Query().Get("model")
			if model == "model-ok" {
				w.WriteHeader(http.StatusOK)
				w.Write([]byte(metricsOK))
				return
			}
			// model-broken returns 500
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer dummy.Close()

	r := newRouter(dummy.URL)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	body := w.Body.String()

	// Should still contain the OK model's metrics
	if !strings.Contains(body, `llamacpp:prompt_tokens_total{model="model-ok"} 42`) {
		t.Error("should contain model-ok metric despite model-broken failure")
	}

	// Should NOT contain model-broken metrics
	if strings.Contains(body, `model="model-broken"`) {
		t.Error("should NOT contain model-broken metrics")
	}

	// router_models_loaded should still reflect all loaded models
	if !strings.Contains(body, "llamacpp_router_models_loaded 2") {
		t.Error("router_models_loaded should still be 2 (total loaded, not just successful)")
	}
}

// ---------- Test 7: Non-metrics path ----------

func TestNonMetricsPath(t *testing.T) {
	dummy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("dummy upstream should not be called for non-/metrics paths")
		w.WriteHeader(http.StatusOK)
	}))
	defer dummy.Close()

	r := newRouter(dummy.URL)

	req := httptest.NewRequest("GET", "/other", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

// ---------- Test 8: Content-Type header ----------

func TestContentTypeHeader(t *testing.T) {
	dummy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"data":[{"id":"m","status":{"value":"loaded"}}],"object":"list"}`))
			return
		}
		if r.URL.Path == "/metrics" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("# TYPE m counter\nm 1\n"))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer dummy.Close()

	r := newRouter(dummy.URL)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	ct := w.Header().Get("Content-Type")
	expected := "text/plain; version=0.0.4; charset=utf-8"
	if ct != expected {
		t.Errorf("expected Content-Type %q, got %q", expected, ct)
	}
}
