from prometheus_client import Counter, Gauge, Histogram

VLLM_TTFT_SECONDS = Histogram(
    "vllm_ttft_seconds",
    "Time to first token in seconds",
    ["model", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

VLLM_ITL_SECONDS = Histogram(
    "vllm_itl_seconds",
    "Inter-token latency in seconds",
    ["model", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

VLLM_TOKENS_GENERATED_TOTAL = Counter(
    "vllm_tokens_generated_total",
    "Total number of tokens generated",
    ["model", "endpoint"],
)

VLLM_REQUESTS_TOTAL = Counter(
    "vllm_requests_total",
    "Total number of requests processed",
    ["model", "endpoint", "status"],
)

VLLM_REQUESTS_IN_PROGRESS = Gauge(
    "vllm_requests_in_progress",
    "Number of requests currently being processed",
    ["model", "endpoint"],
)

VLLM_QUEUE_LENGTH = Gauge(
    "vllm_queue_length",
    "Number of requests waiting in queue",
    ["model", "endpoint"],
)

VLLM_BATCH_SIZE = Gauge(
    "vllm_batch_size",
    "Current batch size for inference",
    ["model", "endpoint"],
)

VLLM_KV_CACHE_USAGE = Gauge(
    "vllm_kv_cache_usage_ratio",
    "KV cache usage ratio (0-1)",
    ["model", "endpoint"],
)

VLLM_GPU_MEMORY_USED = Gauge(
    "vllm_gpu_memory_used_bytes",
    "GPU memory used by vLLM in bytes",
    ["model", "endpoint", "gpu_id"],
)

VLLM_GPU_MEMORY_TOTAL = Gauge(
    "vllm_gpu_memory_total_bytes",
    "Total GPU memory available in bytes",
    ["model", "endpoint", "gpu_id"],
)

VLLM_NUM_PREEMPTED = Counter(
    "vllm_num_preempted_total",
    "Total number of preempted requests",
    ["model", "endpoint"],
)

VLLM_TIME_PER_TOKEN = Histogram(
    "vllm_time_per_token_seconds",
    "Time spent generating each token",
    ["model", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

VLLM_NUM_LIVE_GENERATIONS = Gauge(
    "vllm_num_live_generations",
    "Number of currently active generations",
    ["model", "endpoint"],
)

VLLM_SPECULATIVE_ACCEPTED = Counter(
    "vllm_speculative_accepted_total",
    "Number of tokens accepted via speculative decoding",
    ["model", "endpoint"],
)

VLLM_SPECULATIVE_REJECTED = Counter(
    "vllm_speculative_rejected_total",
    "Number of tokens rejected in speculative decoding",
    ["model", "endpoint"],
)

METRICS = [
    VLLM_TTFT_SECONDS,
    VLLM_ITL_SECONDS,
    VLLM_TOKENS_GENERATED_TOTAL,
    VLLM_REQUESTS_TOTAL,
    VLLM_REQUESTS_IN_PROGRESS,
    VLLM_QUEUE_LENGTH,
    VLLM_BATCH_SIZE,
    VLLM_KV_CACHE_USAGE,
    VLLM_GPU_MEMORY_USED,
    VLLM_GPU_MEMORY_TOTAL,
    VLLM_NUM_PREEMPTED,
    VLLM_TIME_PER_TOKEN,
    VLLM_NUM_LIVE_GENERATIONS,
    VLLM_SPECULATIVE_ACCEPTED,
    VLLM_SPECULATIVE_REJECTED,
]
