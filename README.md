# Token Path Observability

Observability stack for LLM inference (AI Reliability)

Standard web metrics (CPU/RAM) fail for LLMs. This project provides a dedicated monitoring dashboard for the "token path"—the lifecycle of a prompt from request to final token generation.

## Features

- **Token Latency Metrics**: Tracks Time-to-First-Token (TTFT) and Inter-Token Latency (ITL) to diagnose user-perceived lag
- **Accelerator Awareness**: Monitors GPU VRAM vs. Compute utilization to identify memory-bound bottlenecks
- **Prometheus Exporters**: Native exporters for vLLM and TGI inference servers
- **Custom Grafana Dashboards**: Pre-built dashboards for LLM-specific metrics
- **Proactive Detection**: Enables detection of "tail latencies" in model serving

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOKEN PATH OBSERVABILITY STACK                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────────┐
│   LLM Inference │     │   GPU Metrics   │     │    Token Path Metrics       │
│    Servers      │     │   (nvidia-smi)  │     │                             │
│                 │     │                 │     │  • TTFT (Time to First      │
│  ┌───────────┐  │     │  • VRAM Usage   │     │    Token)                   │
│  │   vLLM    │  │     │  • Compute %    │     │  • ITL (Inter-Token         │
│  │  Server   │──┼─────┤  • Temperature  │     │    Latency)                 │
│  └───────────┘  │     │  • Power Draw   │     │  • Tokens/sec               │
│  ┌───────────┐  │     │  • Memory Band  │     │  • Queue Length             │
│  │    TGI    │  │     │    width        │     │  • Batch Size               │
│  │  Server   │──┤     └────────┬────────┘     └─────────────┬───────────────┘
│  └───────────┘  │              │                            │
└────────┬────────┘              │                            │
         │                       │                            │
         ▼                       ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROMETHEUS EXPORTERS                                 │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐     │
│  │  vLLM Exporter   │   │  TGI Exporter    │   │  GPU Exporter        │     │
│  │                  │   │                  │   │  (nvidia-smi wrap)   │     │
│  │ • /metrics       │   │ • /metrics       │   │  • /metrics          │     │
│  │ • Port 8000      │   │ • Port 8001      │   │  • Port 9400         │     │
│  └────────┬─────────┘   └────────┬─────────┘   └──────────┬───────────┘     │
└───────────┼──────────────────────┼────────────────────────┼─────────────────┘
            │                      │                        │
            └──────────────────────┼────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PROMETHEUS SERVER                                 │
│                                                                              │
│  • Scrapes exporters every 15s                                              │
│  • Stores time-series data                                                   │
│  • Recording rules for derived metrics                                       │
│  • Alertmanager integration for latency alerts                               │
│  • Port: 9090                                                               │
│                                                                              │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GRAFANA DASHBOARDS                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TOKEN PATH DASHBOARD                              │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │   TTFT Panel    │  │   ITL Panel     │  │  Throughput Panel   │  │   │
│  │  │                 │  │                 │  │                     │  │   │
│  │  │  P50/P90/P99    │  │  Distribution   │  │  Tokens/sec over    │  │   │
│  │  │  Percentiles    │  │  of inter-token │  │  time with batch   │  │   │
│  │  │  over time      │  │  latencies      │  │  size overlay      │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │              REQUEST LATENCY HEATMAP                             ││   │
│  │  │   Visual distribution of latencies across time buckets          ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GPU UTILIZATION DASHBOARD                         │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │   VRAM Usage    │  │  Compute %      │  │  Memory Bound       │  │   │
│  │  │                 │  │                 │  │  Detection          │  │   │
│  │  │  Used/Total     │  │  SM utilization │  │                     │  │   │
│  │  │  over time      │  │  over time      │  │  Alert when VRAM    │  │   │
│  │  │                 │  │                 │  │  > 90% & Compute    │  │   │
│  │  │                 │  │                 │  │  < 50%              │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │  Temperature    │  │  Power Draw     │  │  Memory Bandwidth   │  │   │
│  │  │                 │  │                 │  │                     │  │   │
│  │  │  Per GPU        │  │  Current/Max    │  │  Utilization %      │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Port: 3000                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

```

## Token Path Metrics Explained

### Time-to-First-Token (TTFT)
The time from when a request is received until the first token is generated. High TTFT indicates:
- Model loading delays
- Queue congestion
- KV-cache allocation overhead
- Cold start issues

### Inter-Token Latency (ITL)
The time between generating consecutive tokens. High ITL indicates:
- Memory bandwidth saturation
- Compute bottlenecks
- Batch scheduling inefficiencies

### GPU VRAM vs Compute Utilization
Identifies memory-bound vs compute-bound scenarios:
- **Memory-bound**: High VRAM, low compute → Need more memory bandwidth or smaller batch sizes
- **Compute-bound**: High compute, moderate VRAM → Optimal utilization
- **Idle**: Low both → Underutilized capacity

## Project Structure

```
token-path-observability/
├── exporters/
│   ├── vllm_exporter/          # Prometheus exporter for vLLM
│   │   ├── __init__.py
│   │   ├── exporter.py
│   │   └── metrics.py
│   ├── tgi_exporter/           # Prometheus exporter for TGI
│   │   ├── __init__.py
│   │   ├── exporter.py
│   │   └── metrics.py
│   └── gpu_exporter/           # NVIDIA GPU metrics exporter
│       ├── __init__.py
│       ├── exporter.py
│       └── metrics.py
├── dashboards/
│   ├── token_path.json         # Grafana dashboard for TTFT/ITL
│   └── gpu_utilization.json    # Grafana dashboard for GPU metrics
├── prometheus/
│   ├── prometheus.yml          # Prometheus configuration
│   └── rules/
│       └── llm_alerts.yml      # Alerting rules for LLM metrics
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml
│   │   └── dashboards/
│   │       └── dashboard.yml
│   └── grafana.ini
├── docker/
│   ├── Dockerfile.exporter     # Dockerfile for exporters
│   └── Dockerfile.gpu          # Dockerfile for GPU exporter
├── docker-compose.yml          # Full stack deployment
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# Clone the repository
git clone git@github.com:ritwikareddykancharla/token-path-observability.git
cd token-path-observability

# Start the observability stack
docker-compose up -d

# Access the dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_ENDPOINT` | vLLM server endpoint | `http://localhost:8000` |
| `TGI_ENDPOINT` | TGI server endpoint | `http://localhost:8080` |
| `PROMETHEUS_PORT` | Prometheus port | `9090` |
| `GRAFANA_PORT` | Grafana port | `3000` |
| `SCRAPE_INTERVAL` | Metrics scrape interval | `15s` |

### Alert Thresholds

| Alert | Condition | Severity |
|-------|-----------|----------|
| `HighTTFT` | P99 TTFT > 5s | warning |
| `CriticalTTFT` | P99 TTFT > 10s | critical |
| `HighITL` | P99 ITL > 100ms | warning |
| `MemoryBound` | VRAM > 90% & Compute < 50% | warning |
| `GPUMemoryExhaustion` | VRAM > 95% | critical |

## API Endpoints

### vLLM Exporter (`:8000/metrics`)
```
# HELP vllm_ttft_seconds Time to first token
# TYPE vllm_ttft_seconds histogram
vllm_ttft_seconds_bucket{le="0.1"} 100
vllm_ttft_seconds_bucket{le="0.5"} 250
...

# HELP vllm_itl_seconds Inter-token latency
# TYPE vllm_itl_seconds histogram
vllm_itl_seconds_bucket{le="0.01"} 500
...

# HELP vllm_tokens_generated_total Total tokens generated
# TYPE vllm_tokens_generated_total counter
vllm_tokens_generated_total{model="llama-2-70b"} 1000000
```

### GPU Exporter (`:9400/metrics`)
```
# HELP gpu_vram_used_bytes GPU memory used
# TYPE gpu_vram_used_bytes gauge
gpu_vram_used_bytes{gpu="0"} 70000000000

# HELP gpu_compute_utilization GPU compute utilization percentage
# TYPE gpu_compute_utilization gauge
gpu_compute_utilization{gpu="0"} 85.5
```

## Tech Stack

- **Python 3.11+**: Exporter implementations
- **Prometheus**: Time-series database and monitoring
- **Grafana**: Visualization and dashboards
- **Docker**: Containerized deployment
- **NVIDIA SMI**: GPU metrics collection

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check .

# Run type checking
mypy .
```

## License

MIT License - see [LICENSE](LICENSE) for details.
