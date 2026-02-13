import pytest
from unittest.mock import patch, MagicMock

from exporters.gpu_exporter.exporter import GPUExporter, GPUMetrics


class TestGPUExporter:
    def test_init(self):
        exporter = GPUExporter(port=9400)
        assert exporter.port == 9400
        assert exporter._nvidia_smi_path == "nvidia-smi"

    def test_safe_float(self):
        exporter = GPUExporter()

        assert exporter._safe_float("42.5") == 42.5
        assert exporter._safe_float(None) == 0.0
        assert exporter._safe_float("invalid", default=10.0) == 10.0

    def test_safe_int(self):
        exporter = GPUExporter()

        assert exporter._safe_int("42") == 42
        assert exporter._safe_int("42.9") == 42
        assert exporter._safe_int(None) == 0
        assert exporter._safe_int("invalid", default=10) == 10

    def test_parse_memory_mib(self):
        exporter = GPUExporter()

        assert exporter._parse_memory("40960 MiB") == 40960 * 1024 * 1024

    def test_parse_memory_gib(self):
        exporter = GPUExporter()

        assert exporter._parse_memory("80 GiB") == 80 * 1024 * 1024 * 1024

    def test_parse_memory_bytes(self):
        exporter = GPUExporter()

        assert exporter._parse_memory("1024 B") == 1024

    def test_parse_memory_none(self):
        exporter = GPUExporter()

        assert exporter._parse_memory(None) == 0

    def test_run_nvidia_smi_success(self):
        exporter = GPUExporter()

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "test output"
            mock_run.return_value = mock_result

            result = exporter._run_nvidia_smi(["-q", "-x"])

            assert result == "test output"
            mock_run.assert_called_once()

    def test_run_nvidia_smi_failure(self):
        exporter = GPUExporter()

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "error"
            mock_run.return_value = mock_result

            result = exporter._run_nvidia_smi(["-q", "-x"])

            assert result == ""

    def test_run_nvidia_smi_timeout(self):
        exporter = GPUExporter()

        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 30)

            result = exporter._run_nvidia_smi(["-q", "-x"])

            assert result == ""

    def test_run_nvidia_smi_not_found(self):
        exporter = GPUExporter()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = exporter._run_nvidia_smi(["-q", "-x"])

            assert result == ""

    def test_parse_xml_output_success(self, mock_nvidia_smi_xml):
        exporter = GPUExporter()

        result = exporter._parse_xml_output(mock_nvidia_smi_xml)

        assert result is not None
        assert result.tag == "nvidia_smi_log"

    def test_parse_xml_output_invalid(self):
        exporter = GPUExporter()

        result = exporter._parse_xml_output("not valid xml")

        assert result is None

    def test_collect_metrics(self, mock_nvidia_smi_xml):
        exporter = GPUExporter()

        with patch.object(exporter, "_run_nvidia_smi", return_value=mock_nvidia_smi_xml):
            metrics_list = exporter.collect_metrics()

            assert len(metrics_list) == 1
            metrics = metrics_list[0]
            assert metrics.gpu_id == 0
            assert metrics.gpu_name == "NVIDIA A100-SXM4-80GB"
            assert metrics.vram_total > 0
            assert metrics.vram_used > 0
            assert metrics.compute_utilization == 0.75

    def test_collect_metrics_empty_output(self):
        exporter = GPUExporter()

        with patch.object(exporter, "_run_nvidia_smi", return_value=""):
            metrics_list = exporter.collect_metrics()

            assert len(metrics_list) == 0

    def test_update_prometheus_metrics(self):
        exporter = GPUExporter()

        metrics = GPUMetrics(
            gpu_id=0,
            gpu_name="NVIDIA A100-SXM4-80GB",
            gpu_uuid="GPU-12345678",
            vram_used=40960 * 1024 * 1024,
            vram_total=81920 * 1024 * 1024,
            vram_free=40960 * 1024 * 1024,
            vram_utilization=0.5,
            compute_utilization=0.75,
            temperature=65,
            power_draw=250.5,
            power_limit=400.0,
            fan_speed=50,
            clock_sm=1410,
            clock_memory=1215,
            pcie_tx=1000000000,
            pcie_rx=500000000,
            memory_bandwidth_util=0.6,
            encoder_util=0.0,
            decoder_util=0.0,
            process_count=1,
            is_memory_bound=False,
        )

        exporter.update_prometheus_metrics([metrics])

    def test_memory_bound_detection(self):
        exporter = GPUExporter()

        memory_bound = GPUMetrics(
            gpu_id=0,
            gpu_name="Test GPU",
            gpu_uuid="test-uuid",
            vram_used=80000 * 1024 * 1024,
            vram_total=81920 * 1024 * 1024,
            vram_free=1920 * 1024 * 1024,
            vram_utilization=0.976,
            compute_utilization=0.45,
            temperature=70,
            power_draw=200.0,
            power_limit=400.0,
            fan_speed=60,
            clock_sm=1200,
            clock_memory=1000,
            pcie_tx=0,
            pcie_rx=0,
            memory_bandwidth_util=0.8,
            encoder_util=0.0,
            decoder_util=0.0,
            process_count=1,
            is_memory_bound=True,
        )

        assert memory_bound.is_memory_bound is True

    def test_not_memory_bound(self):
        exporter = GPUExporter()

        not_bound = GPUMetrics(
            gpu_id=0,
            gpu_name="Test GPU",
            gpu_uuid="test-uuid",
            vram_used=40000 * 1024 * 1024,
            vram_total=81920 * 1024 * 1024,
            vram_free=41920 * 1024 * 1024,
            vram_utilization=0.488,
            compute_utilization=0.85,
            temperature=70,
            power_draw=350.0,
            power_limit=400.0,
            fan_speed=80,
            clock_sm=1410,
            clock_memory=1215,
            pcie_tx=0,
            pcie_rx=0,
            memory_bandwidth_util=0.7,
            encoder_util=0.0,
            decoder_util=0.0,
            process_count=1,
            is_memory_bound=False,
        )

        assert not_bound.is_memory_bound is False

    def test_stop(self):
        exporter = GPUExporter()
        exporter._running = True
        exporter.stop()
        assert exporter._running is False
