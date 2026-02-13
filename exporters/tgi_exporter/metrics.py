from prometheus_client import Counter, Gauge, Histogram

TGI_TTFT_SECONDS = Histogram(
    "tgi_ttft_seconds",
    "Time to first token in seconds",
    ["model", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

TGI_ITL_SECONDS = Histogram(
    "tgi_itl_seconds",
    "Inter-token latency in seconds",
    ["model", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

TGI_TOKENS_GENERATED_TOTAL = Counter(
    "tgi_tokens_generated_total",
    "Total number of tokens generated",
    ["model", "endpoint"],
)

TGI_REQUESTS_TOTAL = Counter(
    "tgi_requests_total",
    "Total number of requests processed",
    ["model", "endpoint", "status"],
)

TGI_REQUESTS_IN_PROGRESS = Gauge(
    "tgi_requests_in_progress",
    "Number of requests currently being processed",
    ["model", "endpoint"],
)

TGI_QUEUE_LENGTH = Gauge(
    "tgi_queue_length",
    "Number of requests waiting in queue",
    ["model", "endpoint"],
)

TGI_BATCH_SIZE = Gauge(
    "tgi_batch_size",
    "Current batch size for inference",
    ["model", "endpoint"],
)

TGI_GPU_MEMORY_USED = Gauge(
    "tgi_gpu_memory_used_bytes",
    "GPU memory used by TGI in bytes",
    ["model", "endpoint", "gpu_id"],
)

TGI_GPU_MEMORY_TOTAL = Gauge(
    "tgi_gpu_memory_total_bytes",
    "Total GPU memory available in bytes",
    ["model", "endpoint", "gpu_id"],
)

TGI_DECODE_TOKENS = Counter(
    "tgi_decode_tokens_total",
    "Total decode tokens processed",
    ["model", "endpoint"],
)

TGI_PREFILL_TOKENS = Counter(
    "tgi_prefill_tokens_total",
    "Total prefill tokens processed",
    ["model", "endpoint"],
)

TGI_TIME_PER_TOKEN = Histogram(
    "tgi_time_per_token_seconds",
    "Time spent generating each token",
    ["model", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

TGI_VALIDATION_ERRORS = Counter(
    "tgi_validation_errors_total",
    "Total number of validation errors",
    ["model", "endpoint"],
)

TGI_INFERENCER_ERRORS = Counter(
    "tgi_inferencer_errors_total",
    "Total number of inferencer errors",
    ["model", "endpoint"],
)

METRICS = [
    TGI_TTFT_SECONDS,
    TGI_ITL_SECONDS,
    TGI_TOKENS_GENERATED_TOTAL,
    TGI_REQUESTS_TOTAL,
    TGI_REQUESTS_IN_PROGRESS,
    TGI_QUEUE_LENGTH,
    TGI_BATCH_SIZE,
    TGI_GPU_MEMORY_USED,
    TGI_GPU_MEMORY_TOTAL,
    TGI_DECODE_TOKENS,
    TGI_PREFILL_TOKENS,
    TGI_TIME_PER_TOKEN,
    TGI_VALIDATION_ERRORS,
    TGI_INFERENCER_ERRORS,
]
