# llama-metrics

A lightweight Go proxy that aggregates per-model Prometheus metrics from llama.cpp router mode into a single endpoint with `model` labels.

## Problem

llama.cpp router mode's `/metrics` endpoint requires `?model={model_id}` query parameter — you can only scrape one model at a time. With dynamic model loading, this makes Prometheus monitoring cumbersome.

## Solution

`llama-metrics` sits between Prometheus and llama-server:

1. Calls `/v1/models` to discover which models are currently loaded
2. Fetches `/metrics?model={id}` for each loaded model
3. Adds `{model="..."}` label to every metric line
4. Exposes a single `/metrics` endpoint for Prometheus

```
Prometheus                    llama-metrics                    llama-server (router)
    |                               |                                  |
    +------- GET /metrics ---------->+                                  |
                                      |--- GET /v1/models ----------->+   |
                                      |<-- models list --------------+   |
                                      |--- GET /metrics?model=a ------>+   |
                                      |--- GET /metrics?model=b ------>+   |
                                      |<-- aggregated metrics ---------+   |
    |<------ labeled metrics --------------+                                  |
```

## Build & Run

```bash
go build -o llama-metrics .

# Default: listen on :8081, upstream http://127.0.0.1:8080
./llama-metrics

# Custom addresses
./llama-metrics -listen :9090 -upstream http://llama-server:8080
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LISTEN_ADDR` | `:8081` | Listen address |
| `UPSTREAM_URL` | `http://127.0.0.1:8080` | Upstream llama-server URL |

## Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'llama-router'
    static_configs:
      - targets: ['llama-metrics:8081']
```

No per-model configuration needed. All models appear as `model` labels on metrics from a single endpoint.

## Example Output

```prometheus
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total{model="model-a"} 100
llamacpp:prompt_tokens_total{model="model-b"} 200
# HELP llamacpp_router_models_loaded Number of currently loaded models.
# TYPE llamacpp_router_models_loaded gauge
llamacpp_router_models_loaded 2
```

## Useful PromQL Queries

```prometheus
# Throughput per model
sum(rate(llamacpp:tokens_predicted_total[5m])) by (model)

# Total active requests across all models
sum(llamacpp:requests_processing)

# Per-model active requests
sum(llamacpp:requests_processing) by (model)
```

## Features

- **Zero external dependencies** — pure Go stdlib
- **Concurrent fetch** — metrics for all loaded models fetched in parallel
- **Model discovery cache** — `/v1/models` results cached for 30s
- **Graceful error handling** — a failing model doesn't break the whole response
- **Empty response when idle** — no upstream requests when no models are loaded

## Systemd

Install as a system service:

```bash
sudo cp llama-metrics /usr/local/bin/
sudo cp llama-metrics.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-metrics
```

Edit `llama-metrics.service` to match your upstream URL, or override via environment:

```bash
sudo systemctl set-environment UPSTREAM_URL=http://llama-server:8080
sudo systemctl restart llama-metrics
```

## Testing

```bash
go test -v ./...
```

All tests use `httptest.NewServer` with dummy upstream responses — no external services required.
