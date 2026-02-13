from prometheus_client import Gauge

GPU_VRAM_USED_BYTES = Gauge(
    "gpu_vram_used_bytes",
    "GPU memory used in bytes",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_VRAM_TOTAL_BYTES = Gauge(
    "gpu_vram_total_bytes",
    "Total GPU memory available in bytes",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_VRAM_FREE_BYTES = Gauge(
    "gpu_vram_free_bytes",
    "GPU memory free in bytes",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_VRAM_UTILIZATION = Gauge(
    "gpu_vram_utilization_ratio",
    "GPU memory utilization ratio (0-1)",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_COMPUTE_UTILIZATION = Gauge(
    "gpu_compute_utilization_ratio",
    "GPU compute utilization ratio (0-1)",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_TEMPERATURE_CELSIUS = Gauge(
    "gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_POWER_DRAW_WATTS = Gauge(
    "gpu_power_draw_watts",
    "GPU power draw in watts",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_POWER_LIMIT_WATTS = Gauge(
    "gpu_power_limit_watts",
    "GPU power limit in watts",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_POWER_UTILIZATION = Gauge(
    "gpu_power_utilization_ratio",
    "GPU power utilization ratio (0-1)",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_FAN_SPEED_PERCENT = Gauge(
    "gpu_fan_speed_percent",
    "GPU fan speed percentage",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_CLOCK_SM_MHZ = Gauge(
    "gpu_clock_sm_mhz",
    "GPU SM (streaming multiprocessor) clock in MHz",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_CLOCK_MEMORY_MHZ = Gauge(
    "gpu_clock_memory_mhz",
    "GPU memory clock in MHz",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_PCIE_TX_BYTES = Gauge(
    "gpu_pcie_tx_bytes",
    "GPU PCIe transmit bytes",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_PCIE_RX_BYTES = Gauge(
    "gpu_pcie_rx_bytes",
    "GPU PCIe receive bytes",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_MEMORY_BANDWIDTH_UTILIZATION = Gauge(
    "gpu_memory_bandwidth_utilization_ratio",
    "GPU memory bandwidth utilization ratio (0-1)",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_ENCODER_UTILIZATION = Gauge(
    "gpu_encoder_utilization_ratio",
    "GPU encoder utilization ratio (0-1)",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_DECODER_UTILIZATION = Gauge(
    "gpu_decoder_utilization_ratio",
    "GPU decoder utilization ratio (0-1)",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_PROCESS_COUNT = Gauge(
    "gpu_process_count",
    "Number of processes using the GPU",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

GPU_MEMORY_BOUND_FLAG = Gauge(
    "gpu_memory_bound_flag",
    "Flag indicating if GPU is memory bound (1) or not (0)",
    ["gpu_id", "gpu_name", "gpu_uuid"],
)

METRICS = [
    GPU_VRAM_USED_BYTES,
    GPU_VRAM_TOTAL_BYTES,
    GPU_VRAM_FREE_BYTES,
    GPU_VRAM_UTILIZATION,
    GPU_COMPUTE_UTILIZATION,
    GPU_TEMPERATURE_CELSIUS,
    GPU_POWER_DRAW_WATTS,
    GPU_POWER_LIMIT_WATTS,
    GPU_POWER_UTILIZATION,
    GPU_FAN_SPEED_PERCENT,
    GPU_CLOCK_SM_MHZ,
    GPU_CLOCK_MEMORY_MHZ,
    GPU_PCIE_TX_BYTES,
    GPU_PCIE_RX_BYTES,
    GPU_MEMORY_BANDWIDTH_UTILIZATION,
    GPU_ENCODER_UTILIZATION,
    GPU_DECODER_UTILIZATION,
    GPU_PROCESS_COUNT,
    GPU_MEMORY_BOUND_FLAG,
]
