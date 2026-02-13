import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from prometheus_client import start_http_server

from exporters.config import settings
from exporters.tgi_exporter.metrics import (
    TGI_BATCH_SIZE,
    TGI_DECODE_TOKENS,
    TGI_GPU_MEMORY_TOTAL,
    TGI_GPU_MEMORY_USED,
    TGI_INFERENCER_ERRORS,
    TGI_ITL_SECONDS,
    TGI_PREFILL_TOKENS,
    TGI_QUEUE_LENGTH,
    TGI_REQUESTS_IN_PROGRESS,
    TGI_REQUESTS_TOTAL,
    TGI_TIME_PER_TOKEN,
    TGI_TOKENS_GENERATED_TOTAL,
    TGI_TTFT_SECONDS,
    TGI_VALIDATION_ERRORS,
)

logger = structlog.get_logger()


@dataclass
class TGIMetrics:
    ttft: float
    itl: list[float]
    tokens_generated: int
    requests_total: int
    requests_in_progress: int
    queue_length: int
    batch_size: int
    gpu_memory_used: dict[int, int]
    gpu_memory_total: dict[int, int]
    decode_tokens: int
    prefill_tokens: int
    validation_errors: int
    inferencer_errors: int


class TGIExporter:
    def __init__(
        self,
        endpoint: str = settings.tgi_endpoint,
        port: int = settings.exporter_port_tgi,
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
            logger.error("Failed to fetch TGI metrics", error=str(e))
            return {}

    async def fetch_health(self) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self.endpoint}/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("Failed to fetch TGI health", error=str(e))
            return {}

    async def fetch_model_info(self) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self.endpoint}/info")
            response.raise_for_status()
            data = response.json()
            if data.get("model_id"):
                self.model = data["model_id"]
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

    def _extract_metric_with_labels(
        self, metrics: dict[str, Any], metric_name: str, label_filter: dict[str, str] = None
    ) -> list[dict[str, Any]]:
        result = []
        values = metrics.get(metric_name, [])
        if not isinstance(values, list):
            return result

        for item in values:
            if label_filter:
                labels = item.get("labels", {})
                if all(labels.get(k) == v for k, v in label_filter.items()):
                    result.append(item)
            else:
                result.append(item)
        return result

    def update_prometheus_metrics(self, metrics: dict[str, Any]) -> None:
        labels = {"model": self.model, "endpoint": self.endpoint}

        queue_length = self._extract_metric_value(metrics, "tgi_queue_size")
        TGI_QUEUE_LENGTH.labels(**labels).set(queue_length)

        batch_size = self._extract_metric_value(metrics, "tgi_batch_size")
        TGI_BATCH_SIZE.labels(**labels).set(batch_size)

        requests_in_progress = self._extract_metric_value(metrics, "tgi_request_count")
        TGI_REQUESTS_IN_PROGRESS.labels(**labels).set(requests_in_progress)

        ttft_values = self._extract_metric_with_labels(metrics, "tgi_time_to_first_token")
        for item in ttft_values:
            TGI_TTFT_SECONDS.labels(**labels).observe(item.get("value", 0))

        itl_values = self._extract_metric_with_labels(metrics, "tgi_inter_token_latency")
        for item in itl_values:
            TGI_ITL_SECONDS.labels(**labels).observe(item.get("value", 0))
            TGI_TIME_PER_TOKEN.labels(**labels).observe(item.get("value", 0))

        decode_tokens = self._extract_metric_value(metrics, "tgi_decoder_tokens")
        prev_decode = self._previous_metrics.get("decode_tokens", 0)
        if decode_tokens > prev_decode:
            TGI_DECODE_TOKENS.labels(**labels).inc(decode_tokens - prev_decode)
            TGI_TOKENS_GENERATED_TOTAL.labels(**labels).inc(decode_tokens - prev_decode)
        self._previous_metrics["decode_tokens"] = decode_tokens

        prefill_tokens = self._extract_metric_value(metrics, "tgi_prefill_tokens")
        prev_prefill = self._previous_metrics.get("prefill_tokens", 0)
        if prefill_tokens > prev_prefill:
            TGI_PREFILL_TOKENS.labels(**labels).inc(prefill_tokens - prev_prefill)
        self._previous_metrics["prefill_tokens"] = prefill_tokens

        total_requests = self._extract_metric_value(metrics, "tgi_request_success")
        prev_requests = self._previous_metrics.get("total_requests", 0)
        if total_requests > prev_requests:
            TGI_REQUESTS_TOTAL.labels(**labels, status="success").inc(
                total_requests - prev_requests
            )
        self._previous_metrics["total_requests"] = total_requests

        failed_requests = self._extract_metric_value(metrics, "tgi_request_failure")
        prev_failed = self._previous_metrics.get("failed_requests", 0)
        if failed_requests > prev_failed:
            TGI_REQUESTS_TOTAL.labels(**labels, status="failed").inc(failed_requests - prev_failed)
        self._previous_metrics["failed_requests"] = failed_requests

        validation_errors = self._extract_metric_value(metrics, "tgi_validation_error")
        prev_validation = self._previous_metrics.get("validation_errors", 0)
        if validation_errors > prev_validation:
            TGI_VALIDATION_ERRORS.labels(**labels).inc(validation_errors - prev_validation)
        self._previous_metrics["validation_errors"] = validation_errors

        inferencer_errors = self._extract_metric_value(metrics, "tgi_inferencer_error")
        prev_inferencer = self._previous_metrics.get("inferencer_errors", 0)
        if inferencer_errors > prev_inferencer:
            TGI_INFERENCER_ERRORS.labels(**labels).inc(inferencer_errors - prev_inferencer)
        self._previous_metrics["inferencer_errors"] = inferencer_errors

        for gpu_id, gpu_metrics in self._extract_gpu_memory(metrics).items():
            gpu_labels = {**labels, "gpu_id": str(gpu_id)}
            TGI_GPU_MEMORY_USED.labels(**gpu_labels).set(gpu_metrics["used"])
            TGI_GPU_MEMORY_TOTAL.labels(**gpu_labels).set(gpu_metrics["total"])

    def _extract_gpu_memory(self, metrics: dict[str, Any]) -> dict[int, dict[str, int]]:
        gpu_memory: dict[int, dict[str, int]] = {}
        gpu_memory_pattern = re.compile(r"gpu_memory_(\d+)_used")
        for metric_name, values in metrics.items():
            match = gpu_memory_pattern.match(metric_name)
            if match:
                gpu_id = int(match.group(1))
                if gpu_id not in gpu_memory:
                    gpu_memory[gpu_id] = {"used": 0, "total": 0}
                if isinstance(values, list) and values:
                    gpu_memory[gpu_id]["used"] = int(values[0].get("value", 0))

        for metric_name, values in metrics.items():
            if "gpu_memory_total" in metric_name:
                match = re.search(r"(\d+)", metric_name)
                if match:
                    gpu_id = int(match.group(1))
                    if gpu_id not in gpu_memory:
                        gpu_memory[gpu_id] = {"used": 0, "total": 0}
                    if isinstance(values, list) and values:
                        gpu_memory[gpu_id]["total"] = int(values[0].get("value", 0))
        return gpu_memory

    async def collect_loop(self, interval: float = 15.0) -> None:
        self._running = True
        logger.info("Starting TGI exporter collection loop", endpoint=self.endpoint)

        await self.fetch_model_info()

        while self._running:
            try:
                metrics = await self.fetch_metrics()
                if metrics:
                    self.update_prometheus_metrics(metrics)
                    logger.debug("Updated TGI metrics", model=self.model)
            except Exception as e:
                logger.error("Error collecting TGI metrics", error=str(e))

            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False
        logger.info("Stopping TGI exporter")

    def run(self, interval: float = 15.0) -> None:
        logging.basicConfig(level=settings.log_level)
        start_http_server(self.port)
        logger.info(f"TGI exporter started on port {self.port}")

        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.collect_loop(interval))
        except KeyboardInterrupt:
            self.stop()
        finally:
            loop.close()


def main() -> None:
    exporter = TGIExporter()
    exporter.run()


if __name__ == "__main__":
    main()
