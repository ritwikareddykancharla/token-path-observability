import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from prometheus_client import start_http_server

from exporters.config import settings
from exporters.vllm_exporter.metrics import (
    VLLM_BATCH_SIZE,
    VLLM_GPU_MEMORY_TOTAL,
    VLLM_GPU_MEMORY_USED,
    VLLM_ITL_SECONDS,
    VLLM_KV_CACHE_USAGE,
    VLLM_NUM_LIVE_GENERATIONS,
    VLLM_NUM_PREEMPTED,
    VLLM_QUEUE_LENGTH,
    VLLM_REQUESTS_IN_PROGRESS,
    VLLM_REQUESTS_TOTAL,
    VLLM_SPECULATIVE_ACCEPTED,
    VLLM_SPECULATIVE_REJECTED,
    VLLM_TIME_PER_TOKEN,
    VLLM_TOKENS_GENERATED_TOTAL,
    VLLM_TTFT_SECONDS,
)

logger = structlog.get_logger()


@dataclass
class VLLMMetrics:
    ttft: float
    itl: list[float]
    tokens_generated: int
    requests_total: int
    requests_in_progress: int
    queue_length: int
    batch_size: int
    kv_cache_usage: float
    gpu_memory_used: dict[int, int]
    gpu_memory_total: dict[int, int]
    num_preempted: int
    time_per_token: float
    num_live_generations: int
    speculative_accepted: int
    speculative_rejected: int


class VLLMExporter:
    def __init__(
        self,
        endpoint: str = settings.vllm_endpoint,
        port: int = settings.exporter_port_vllm,
        model: str = "unknown",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.port = port
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)
        self._running = False
        self._previous_metrics: dict[str, Any] = {}

    async def fetch_metrics(self) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self.endpoint}/metrics")
            response.raise_for_status()
            return self._parse_prometheus_metrics(response.text)
        except httpx.HTTPError as e:
            logger.error("Failed to fetch vLLM metrics", error=str(e))
            return {}

    async def fetch_health(self) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self.endpoint}/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("Failed to fetch vLLM health", error=str(e))
            return {}

    async def fetch_model_info(self) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self.endpoint}/v1/models")
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                self.model = data["data"][0].get("id", "unknown")
            return data
        except httpx.HTTPError as e:
            logger.error("Failed to fetch model info", error=str(e))
            return {}

    def _parse_prometheus_metrics(self, metrics_text: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for line in metrics_text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            try:
                if "{" in line:
                    metric_part, value = line.rsplit(" ", 1)
                    metric_name = metric_part.split("{")[0]
                    labels_part = metric_part.split("{")[1].rstrip("}")
                    labels = {}
                    for label in labels_part.split(","):
                        if "=" in label:
                            k, v = label.split("=", 1)
                            labels[k.strip()] = v.strip('"')
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append({"labels": labels, "value": float(value)})
                else:
                    metric_name, value = line.split(" ", 1)
                    metrics[metric_name] = float(value)
            except (ValueError, IndexError):
                continue
        return metrics

    def _extract_metric_value(
        self, metrics: dict[str, Any], metric_name: str, default: float = 0.0
    ) -> float:
        value = metrics.get(metric_name, default)
        if isinstance(value, list) and len(value) > 0:
            return value[0].get("value", default)
        return value if isinstance(value, (int, float)) else default

    def update_prometheus_metrics(self, metrics: dict[str, Any]) -> None:
        labels = {"model": self.model, "endpoint": self.endpoint}

        queue_length = self._extract_metric_value(metrics, "vllm:num_requests_waiting")
        VLLM_QUEUE_LENGTH.labels(**labels).set(queue_length)

        batch_size = self._extract_metric_value(metrics, "vllm:num_batched_tokens")
        VLLM_BATCH_SIZE.labels(**labels).set(batch_size)

        kv_cache = self._extract_metric_value(metrics, "vllm:gpu_cache_usage_perc")
        VLLM_KV_CACHE_USAGE.labels(**labels).set(kv_cache / 100 if kv_cache > 1 else kv_cache)

        num_live = self._extract_metric_value(metrics, "vllm:num_generations")
        VLLM_NUM_LIVE_GENERATIONS.labels(**labels).set(num_live)

        num_running = self._extract_metric_value(metrics, "vllm:num_requests_running")
        VLLM_REQUESTS_IN_PROGRESS.labels(**labels).set(num_running)

        if "vllm:time_to_first_token_seconds" in metrics:
            ttft_values = metrics["vllm:time_to_first_token_seconds"]
            if isinstance(ttft_values, list):
                for item in ttft_values:
                    VLLM_TTFT_SECONDS.labels(**labels).observe(item.get("value", 0))

        if "vllm:time_per_output_token_seconds" in metrics:
            itl_values = metrics["vllm:time_per_output_token_seconds"]
            if isinstance(itl_values, list):
                for item in itl_values:
                    VLLM_ITL_SECONDS.labels(**labels).observe(item.get("value", 0))
                    VLLM_TIME_PER_TOKEN.labels(**labels).observe(item.get("value", 0))

        total_tokens = self._extract_metric_value(metrics, "vllm:total_tokens")
        prev_tokens = self._previous_metrics.get("total_tokens", 0)
        if total_tokens > prev_tokens:
            VLLM_TOKENS_GENERATED_TOTAL.labels(**labels).inc(total_tokens - prev_tokens)
        self._previous_metrics["total_tokens"] = total_tokens

        total_requests = self._extract_metric_value(metrics, "vllm:num_requests_total")
        prev_requests = self._previous_metrics.get("total_requests", 0)
        if total_requests > prev_requests:
            VLLM_REQUESTS_TOTAL.labels(**labels, status="completed").inc(
                total_requests - prev_requests
            )
        self._previous_metrics["total_requests"] = total_requests

        preempted = self._extract_metric_value(metrics, "vllm:num_preemptions_total")
        prev_preempted = self._previous_metrics.get("preempted", 0)
        if preempted > prev_preempted:
            VLLM_NUM_PREEMPTED.labels(**labels).inc(preempted - prev_preempted)
        self._previous_metrics["preempted"] = preempted

        spec_accepted = self._extract_metric_value(
            metrics, "vllm:spec_decoding_accepted_tokens_total"
        )
        prev_accepted = self._previous_metrics.get("spec_accepted", 0)
        if spec_accepted > prev_accepted:
            VLLM_SPECULATIVE_ACCEPTED.labels(**labels).inc(spec_accepted - prev_accepted)
        self._previous_metrics["spec_accepted"] = spec_accepted

        spec_rejected = self._extract_metric_value(
            metrics, "vllm:spec_decoding_rejected_tokens_total"
        )
        prev_rejected = self._previous_metrics.get("spec_rejected", 0)
        if spec_rejected > prev_rejected:
            VLLM_SPECULATIVE_REJECTED.labels(**labels).inc(spec_rejected - prev_rejected)
        self._previous_metrics["spec_rejected"] = spec_rejected

        for gpu_id, gpu_metrics in self._extract_gpu_memory(metrics).items():
            gpu_labels = {**labels, "gpu_id": str(gpu_id)}
            VLLM_GPU_MEMORY_USED.labels(**gpu_labels).set(gpu_metrics["used"])
            VLLM_GPU_MEMORY_TOTAL.labels(**gpu_labels).set(gpu_metrics["total"])

    def _extract_gpu_memory(self, metrics: dict[str, Any]) -> dict[int, dict[str, int]]:
        gpu_memory: dict[int, dict[str, int]] = {}
        for metric_name in ["vllm:gpu_memory_used_bytes", "vllm:gpu_memory_total_bytes"]:
            if metric_name in metrics:
                values = metrics[metric_name]
                if isinstance(values, list):
                    for item in values:
                        gpu_id = int(item.get("labels", {}).get("gpu", 0))
                        if gpu_id not in gpu_memory:
                            gpu_memory[gpu_id] = {"used": 0, "total": 0}
                        if "used" in metric_name:
                            gpu_memory[gpu_id]["used"] = int(item.get("value", 0))
                        else:
                            gpu_memory[gpu_id]["total"] = int(item.get("value", 0))
        return gpu_memory

    async def collect_loop(self, interval: float = 15.0) -> None:
        self._running = True
        logger.info("Starting vLLM exporter collection loop", endpoint=self.endpoint)

        await self.fetch_model_info()

        while self._running:
            try:
                metrics = await self.fetch_metrics()
                if metrics:
                    self.update_prometheus_metrics(metrics)
                    logger.debug("Updated vLLM metrics", model=self.model)
            except Exception as e:
                logger.error("Error collecting vLLM metrics", error=str(e))

            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False
        logger.info("Stopping vLLM exporter")

    def run(self, interval: float = 15.0) -> None:
        logging.basicConfig(level=settings.log_level)
        start_http_server(self.port)
        logger.info(f"vLLM exporter started on port {self.port}")

        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.collect_loop(interval))
        except KeyboardInterrupt:
            self.stop()
        finally:
            loop.close()


def main() -> None:
    exporter = VLLMExporter()
    exporter.run()


if __name__ == "__main__":
    main()
