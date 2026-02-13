import asyncio
import logging
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import structlog
from prometheus_client import start_http_server

from exporters.config import settings
from exporters.gpu_exporter.metrics import (
    GPU_CLOCK_MEMORY_MHZ,
    GPU_CLOCK_SM_MHZ,
    GPU_COMPUTE_UTILIZATION,
    GPU_DECODER_UTILIZATION,
    GPU_ENCODER_UTILIZATION,
    GPU_FAN_SPEED_PERCENT,
    GPU_MEMORY_BANDWIDTH_UTILIZATION,
    GPU_MEMORY_BOUND_FLAG,
    GPU_PCIE_RX_BYTES,
    GPU_PCIE_TX_BYTES,
    GPU_POWER_DRAW_WATTS,
    GPU_POWER_LIMIT_WATTS,
    GPU_POWER_UTILIZATION,
    GPU_PROCESS_COUNT,
    GPU_TEMPERATURE_CELSIUS,
    GPU_VRAM_FREE_BYTES,
    GPU_VRAM_TOTAL_BYTES,
    GPU_VRAM_UTILIZATION,
    GPU_VRAM_USED_BYTES,
)

logger = structlog.get_logger()

MEMORY_BOUND_VRAM_THRESHOLD = 0.90
MEMORY_BOUND_COMPUTE_THRESHOLD = 0.50


@dataclass
class GPUMetrics:
    gpu_id: int
    gpu_name: str
    gpu_uuid: str
    vram_used: int
    vram_total: int
    vram_free: int
    vram_utilization: float
    compute_utilization: float
    temperature: int
    power_draw: float
    power_limit: float
    fan_speed: int
    clock_sm: int
    clock_memory: int
    pcie_tx: int
    pcie_rx: int
    memory_bandwidth_util: float
    encoder_util: float
    decoder_util: float
    process_count: int
    is_memory_bound: bool


class GPUExporter:
    def __init__(self, port: int = settings.exporter_port_gpu):
        self.port = port
        self._running = False
        self._nvidia_smi_path = "nvidia-smi"

    def _run_nvidia_smi(self, args: list[str]) -> str:
        try:
            result = subprocess.run(
                [self._nvidia_smi_path] + args,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error("nvidia-smi failed", stderr=result.stderr)
                return ""
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi timed out")
            return ""
        except FileNotFoundError:
            logger.error("nvidia-smi not found")
            return ""

    def _parse_xml_output(self, xml_output: str) -> ET.Element | None:
        try:
            return ET.fromstring(xml_output)
        except ET.ParseError as e:
            logger.error("Failed to parse nvidia-smi XML output", error=str(e))
            return None

    def _safe_float(self, value: str | None, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def _safe_int(self, value: str | None, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(float(value))
        except ValueError:
            return default

    def _parse_memory(self, memory_str: str | None) -> int:
        if memory_str is None:
            return 0
        memory_str = memory_str.strip().upper()
        if memory_str.endswith(" MIB"):
            return int(float(memory_str[:-4]) * 1024 * 1024)
        elif memory_str.endswith(" GIB"):
            return int(float(memory_str[:-4]) * 1024 * 1024 * 1024)
        elif memory_str.endswith(" B"):
            return int(float(memory_str[:-2]))
        return self._safe_int(memory_str)

    def collect_metrics(self) -> list[GPUMetrics]:
        xml_output = self._run_nvidia_smi(
            [
                "-q",
                "-x",
                "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,memory.free,temperature.gpu,power.draw,power.limit,fan.speed,clocks.current.sm,clocks.current.memory,pcie.tx_throughput,pcie.rx_throughput",
            ]
        )

        if not xml_output:
            return []

        root = self._parse_xml_output(xml_output)
        if root is None:
            return []

        metrics_list: list[GPUMetrics] = []

        for gpu in root.findall(".//gpu"):
            try:
                gpu_id = self._safe_int(gpu.findtext("gpu_id") or gpu.get("id", "0"))
                gpu_name = gpu.findtext("product_name") or "Unknown"
                gpu_uuid = gpu.findtext("uuid") or "Unknown"

                fb_memory = gpu.find("fb_memory_usage")
                if fb_memory is not None:
                    vram_used = self._parse_memory(fb_memory.findtext("used"))
                    vram_total = self._parse_memory(fb_memory.findtext("total"))
                    vram_free = self._parse_memory(fb_memory.findtext("free"))
                else:
                    vram_used = vram_total = vram_free = 0

                vram_utilization = vram_used / vram_total if vram_total > 0 else 0.0

                utilization = gpu.find("utilization")
                if utilization is not None:
                    compute_utilization = self._safe_float(utilization.findtext("gpu_util")) / 100
                    memory_bandwidth_util = (
                        self._safe_float(utilization.findtext("memory_util")) / 100
                    )
                else:
                    compute_utilization = memory_bandwidth_util = 0.0

                temperature = self._safe_int(gpu.findtext("temperature/gpu_temp"))

                power_readings = gpu.find("power_readings")
                if power_readings is not None:
                    power_draw = self._safe_float(
                        power_readings.findtext("power_draw"), default=0.0
                    )
                    power_limit = self._safe_float(
                        power_readings.findtext("default_power_limit")
                        or power_readings.findtext("power_limit"),
                        default=0.0,
                    )
                else:
                    power_draw = power_limit = 0.0

                fan_speed = self._safe_int(gpu.findtext("fan_speed"))

                clocks = gpu.find("clocks")
                if clocks is not None:
                    clock_sm = self._safe_int(clocks.findtext("sm_clock"))
                    clock_memory = self._safe_int(clocks.findtext("mem_clock"))
                else:
                    clock_sm = clock_memory = 0

                pcie = gpu.find("pci")
                if pcie is not None:
                    tx_throughput = pcie.find("tx_throughput")
                    rx_throughput = pcie.find("rx_throughput")
                    pcie_tx = (
                        self._safe_int(tx_throughput.findtext("value"))
                        if tx_throughput is not None
                        else 0
                    )
                    pcie_rx = (
                        self._safe_int(rx_throughput.findtext("value"))
                        if rx_throughput is not None
                        else 0
                    )
                else:
                    pcie_tx = pcie_rx = 0

                encoder_util = decoder_util = 0.0
                encoder_stats = gpu.find("encoder_stats")
                if encoder_stats is not None:
                    encoder_util = self._safe_float(encoder_stats.findtext("utilization")) / 100
                decoder_stats = gpu.find("decoder_stats")
                if decoder_stats is not None:
                    decoder_util = self._safe_float(decoder_stats.findtext("utilization")) / 100

                processes = gpu.find("processes")
                process_count = (
                    len(processes.findall("process_info")) if processes is not None else 0
                )

                is_memory_bound = (
                    vram_utilization > MEMORY_BOUND_VRAM_THRESHOLD
                    and compute_utilization < MEMORY_BOUND_COMPUTE_THRESHOLD
                )

                metrics = GPUMetrics(
                    gpu_id=gpu_id,
                    gpu_name=gpu_name,
                    gpu_uuid=gpu_uuid,
                    vram_used=vram_used,
                    vram_total=vram_total,
                    vram_free=vram_free,
                    vram_utilization=vram_utilization,
                    compute_utilization=compute_utilization,
                    temperature=temperature,
                    power_draw=power_draw,
                    power_limit=power_limit,
                    fan_speed=fan_speed,
                    clock_sm=clock_sm,
                    clock_memory=clock_memory,
                    pcie_tx=pcie_tx,
                    pcie_rx=pcie_rx,
                    memory_bandwidth_util=memory_bandwidth_util,
                    encoder_util=encoder_util,
                    decoder_util=decoder_util,
                    process_count=process_count,
                    is_memory_bound=is_memory_bound,
                )
                metrics_list.append(metrics)
            except Exception as e:
                logger.error("Error parsing GPU metrics", error=str(e))
                continue

        return metrics_list

    def update_prometheus_metrics(self, metrics_list: list[GPUMetrics]) -> None:
        for metrics in metrics_list:
            labels = {
                "gpu_id": str(metrics.gpu_id),
                "gpu_name": metrics.gpu_name,
                "gpu_uuid": metrics.gpu_uuid,
            }

            GPU_VRAM_USED_BYTES.labels(**labels).set(metrics.vram_used)
            GPU_VRAM_TOTAL_BYTES.labels(**labels).set(metrics.vram_total)
            GPU_VRAM_FREE_BYTES.labels(**labels).set(metrics.vram_free)
            GPU_VRAM_UTILIZATION.labels(**labels).set(metrics.vram_utilization)
            GPU_COMPUTE_UTILIZATION.labels(**labels).set(metrics.compute_utilization)
            GPU_TEMPERATURE_CELSIUS.labels(**labels).set(metrics.temperature)
            GPU_POWER_DRAW_WATTS.labels(**labels).set(metrics.power_draw)
            GPU_POWER_LIMIT_WATTS.labels(**labels).set(metrics.power_limit)

            power_utilization = (
                metrics.power_draw / metrics.power_limit if metrics.power_limit > 0 else 0.0
            )
            GPU_POWER_UTILIZATION.labels(**labels).set(power_utilization)

            GPU_FAN_SPEED_PERCENT.labels(**labels).set(metrics.fan_speed)
            GPU_CLOCK_SM_MHZ.labels(**labels).set(metrics.clock_sm)
            GPU_CLOCK_MEMORY_MHZ.labels(**labels).set(metrics.clock_memory)
            GPU_PCIE_TX_BYTES.labels(**labels).set(metrics.pcie_tx)
            GPU_PCIE_RX_BYTES.labels(**labels).set(metrics.pcie_rx)
            GPU_MEMORY_BANDWIDTH_UTILIZATION.labels(**labels).set(metrics.memory_bandwidth_util)
            GPU_ENCODER_UTILIZATION.labels(**labels).set(metrics.encoder_util)
            GPU_DECODER_UTILIZATION.labels(**labels).set(metrics.decoder_util)
            GPU_PROCESS_COUNT.labels(**labels).set(metrics.process_count)
            GPU_MEMORY_BOUND_FLAG.labels(**labels).set(1 if metrics.is_memory_bound else 0)

    async def collect_loop(self, interval: float = 15.0) -> None:
        self._running = True
        logger.info("Starting GPU exporter collection loop")

        while self._running:
            try:
                metrics_list = self.collect_metrics()
                if metrics_list:
                    self.update_prometheus_metrics(metrics_list)
                    logger.debug(
                        "Updated GPU metrics",
                        gpu_count=len(metrics_list),
                        gpus=[m.gpu_name for m in metrics_list],
                    )
            except Exception as e:
                logger.error("Error collecting GPU metrics", error=str(e))

            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False
        logger.info("Stopping GPU exporter")

    def run(self, interval: float = 15.0) -> None:
        logging.basicConfig(level=settings.log_level)
        start_http_server(self.port)
        logger.info(f"GPU exporter started on port {self.port}")

        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.collect_loop(interval))
        except KeyboardInterrupt:
            self.stop()
        finally:
            loop.close()


def main() -> None:
    exporter = GPUExporter()
    exporter.run()


if __name__ == "__main__":
    main()
