package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"
)

const (
	defaultListenAddr = ":8081"
	defaultUpstream   = "http://127.0.0.1:8080"
	defaultLogLevel   = "info"
)

// ---------- data types ----------

type modelStatus struct {
	Value string `json:"value"`
}

type modelInfo struct {
	ID     string      `json:"id"`
	Status modelStatus `json:"status"`
}

type modelsResponse struct {
	Data []modelInfo `json:"data"`
}

// ---------- model discovery ----------

func discoverLoadedModels(ctx context.Context, client *http.Client, upstream string) ([]string, error) {
	u := strings.TrimRight(upstream, "/") + "/v1/models"
	slog.Debug("fetching model list", "url", u)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET %s: %w", u, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: status %d", u, resp.StatusCode)
	}

	var mr modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&mr); err != nil {
		return nil, fmt.Errorf("parse /v1/models: %w", err)
	}

	total := len(mr.Data)
	var loaded []string
	for _, m := range mr.Data {
		if m.Status.Value == "loaded" {
			loaded = append(loaded, m.ID)
		}
	}

	slog.Debug("model list fetched",
		"total_models", total,
		"loaded_count", len(loaded),
		"loaded_models", loaded,
	)

	if len(loaded) == 0 {
		slog.Debug("no loaded models found")
	}

	return loaded, nil
}

// ---------- metrics fetch ----------

func fetchModelMetrics(ctx context.Context, client *http.Client, upstream, modelID string) (string, error) {
	u := strings.TrimRight(upstream, "/") + "/metrics?model=" + url.QueryEscape(modelID)
	slog.Debug("fetching model metrics", "url", u, "model", modelID)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("GET %s: %w", u, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GET %s: status %d", u, resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read body: %w", err)
	}

	lineCount := strings.Count(string(body), "\n") + 1
	slog.Debug("model metrics fetched", "model", modelID, "bytes", len(body), "lines", lineCount)

	return string(body), nil
}

// ---------- label injection ----------

func addModelLabel(raw string, modelID string) string {
	var sb strings.Builder
	scanner := bufio.NewScanner(strings.NewReader(raw))
	dataLines := 0
	commentLines := 0
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			sb.WriteByte('\n')
			continue
		}
		if strings.HasPrefix(line, "#") {
			sb.WriteString(line)
			sb.WriteByte('\n')
			commentLines++
			continue
		}
		metricName, value, ok := strings.Cut(line, " ")
		if !ok {
			sb.WriteString(line)
			sb.WriteByte('\n')
			dataLines++
			continue
		}
		sb.WriteString(metricName)
		sb.WriteString("{model=\"")
		sb.WriteString(modelID)
		sb.WriteString("\"} ")
		sb.WriteString(value)
		sb.WriteByte('\n')
		dataLines++
	}

	slog.Debug("labels injected",
		"model", modelID,
		"data_lines", dataLines,
		"comment_lines", commentLines,
	)

	return sb.String()
}

// ---------- handler ----------

type router struct {
	upstream string
	client   *http.Client
}

func newRouter(upstream string) *router {
	return &router{
		upstream: upstream,
		client:   new(http.Client),
	}
}

func (r *router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.URL.Path != "/metrics" {
		slog.Debug("ignoring non-metrics path", "path", req.URL.Path)
		http.NotFound(w, req)
		return
	}

	slog.Info("metrics request", "remote", req.RemoteAddr)
	ctx := req.Context()

	loaded, err := discoverLoadedModels(ctx, r.client, r.upstream)
	if err != nil {
		slog.Error("error discovering models", "error", err)
		http.Error(w, "model discovery failed", http.StatusInternalServerError)
		return
	}

	if len(loaded) == 0 {
		slog.Info("no loaded models, returning llamacpp:router_models_loaded")
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		writeLoaded(w, 0)
		return
	}

	slog.Info("fetching metrics for loaded models", "count", len(loaded), "models", loaded)

	// Fetch metrics for all loaded models concurrently.
	// Each goroutine gets its own request derived from ctx, so if the
	// downstream client cancels, all in-flight upstream requests abort.
	type result struct {
		modelID string
		body    string
		err     error
	}

	ch := make(chan result, len(loaded))
	for _, modelID := range loaded {
		go func(mid string) {
			raw, err := fetchModelMetrics(ctx, r.client, r.upstream, mid)
			ch <- result{modelID: mid, body: raw, err: err}
		}(modelID)
	}

	// Collect all results into a map for deterministic ordering.
	results := make(map[string]string, len(loaded))
	failed := 0
	for i := 0; i < len(loaded); i++ {
		res := <-ch
		if res.err != nil {
			slog.Warn("failed to fetch metrics for model", "model", res.modelID, "error", res.err)
			failed++
			continue
		}
		results[res.modelID] = addModelLabel(res.body, res.modelID)
	}

	if failed > 0 {
		slog.Warn("some models failed", "failed", failed, "success", len(loaded)-failed)
	}

	// Output in the order models appear in the loaded slice.
	// First model: HELP/TYPE + data interleaved.
	// Subsequent models: data only (HELP/TYPE already emitted).
	var output bytes.Buffer
	first := true
	totalDataLines := 0

	for _, modelID := range loaded {
		labeled, ok := results[modelID]
		if !ok {
			continue
		}

		scanner := bufio.NewScanner(strings.NewReader(labeled))
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}
			if strings.HasPrefix(line, "#") {
				if first {
					output.WriteString(line)
					output.WriteByte('\n')
				}
				continue
			}
			output.WriteString(line)
			output.WriteByte('\n')
			totalDataLines++
		}
		first = false
	}

	// Add router_models_loaded gauge
	writeLoaded(&output, len(loaded))

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	l, err := io.Copy(w, &output)

	slog.Info("response ready",
		"loaded_models", len(loaded),
		"data_lines", totalDataLines,
		"response_bytes", l,
		"failed_models", failed,
		"err", err,
	)
}

func writeLoaded(wt io.Writer, i int) {
	wt.Write([]byte("# HELP llamacpp:router_models_loaded Number of currently loaded models.\n"))
	wt.Write([]byte("# TYPE llamacpp:router_models_loaded gauge\n"))
	fmt.Fprintf(wt, "llamacpp:router_models_loaded %d\n", i)
}

// ---------- main ----------

func main() {
	listenAddr := flag.String("listen", envOr("LISTEN_ADDR", defaultListenAddr), "listen address")
	upstream := flag.String("upstream", envOr("UPSTREAM_URL", defaultUpstream), "upstream llama-server URL")
	logLevel := flag.String("log-level", envOr("LOG_LEVEL", defaultLogLevel), "log level: debug, info, warn, error")
	flag.Parse()

	level := parseLogLevel(*logLevel)
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})))

	r := newRouter(*upstream)

	slog.Info("llama-metrics starting",
		"listen", *listenAddr,
		"upstream", *upstream,
		"log_level", *logLevel,
	)
	if err := http.ListenAndServe(*listenAddr, r); err != nil {
		slog.Error("server error", "error", err)
		os.Exit(1)
	}
}

// ---------- helpers ----------

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func parseLogLevel(s string) slog.Level {
	var l slog.Level
	l.UnmarshalText([]byte(s))
	return l
}
