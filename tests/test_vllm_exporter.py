import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from exporters.vllm_exporter.exporter import VLLMExporter


class TestVLLMExporter:
    def test_init(self):
        exporter = VLLMExporter(endpoint="http://localhost:8000", port=8080, model="test-model")
        assert exporter.endpoint == "http://localhost:8000"
        assert exporter.port == 8080
        assert exporter.model == "test-model"

    def test_init_strips_trailing_slash(self):
        exporter = VLLMExporter(endpoint="http://localhost:8000/")
        assert exporter.endpoint == "http://localhost:8000"

    def test_parse_prometheus_metrics(self, mock_vllm_metrics):
        exporter = VLLMExporter()
        result = exporter._parse_prometheus_metrics(mock_vllm_metrics)

        assert "vllm:num_requests_running" in result
        assert result["vllm:num_requests_running"] == 5.0
        assert "vllm:num_requests_waiting" in result
        assert result["vllm:num_requests_waiting"] == 10.0
        assert "vllm:time_to_first_token_seconds_bucket" in result

    def test_parse_prometheus_metrics_empty_input(self):
        exporter = VLLMExporter()
        result = exporter._parse_prometheus_metrics("")
        assert result == {}

    def test_parse_prometheus_metrics_malformed_input(self):
        exporter = VLLMExporter()
        result = exporter._parse_prometheus_metrics("not valid prometheus\n# comment\n")
        assert result == {}

    def test_extract_metric_value_simple(self):
        exporter = VLLMExporter()
        metrics = {"simple_metric": 42.0}
        result = exporter._extract_metric_value(metrics, "simple_metric")
        assert result == 42.0

    def test_extract_metric_value_with_labels(self):
        exporter = VLLMExporter()
        metrics = {
            "labeled_metric": [
                {"labels": {"gpu": "0"}, "value": 100.0},
                {"labels": {"gpu": "1"}, "value": 200.0},
            ]
        }
        result = exporter._extract_metric_value(metrics, "labeled_metric")
        assert result == 100.0

    def test_extract_metric_value_default(self):
        exporter = VLLMExporter()
        metrics = {}
        result = exporter._extract_metric_value(metrics, "missing_metric", default=99.0)
        assert result == 99.0

    def test_update_prometheus_metrics(self, mock_vllm_metrics):
        exporter = VLLMExporter(model="test-model")
        metrics = exporter._parse_prometheus_metrics(mock_vllm_metrics)

        exporter.update_prometheus_metrics(metrics)

    @pytest.mark.asyncio
    async def test_fetch_metrics_success(self, mock_vllm_metrics):
        exporter = VLLMExporter()

        with patch.object(exporter.client, "get", new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_vllm_metrics
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await exporter.fetch_metrics()

            assert "vllm:num_requests_running" in result
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_metrics_http_error(self):
        exporter = VLLMExporter()

        with patch.object(exporter.client, "get", new_callable=AsyncMock) as mock_get:
            import httpx

            mock_get.side_effect = httpx.HTTPError("Connection failed")

            result = await exporter.fetch_metrics()

            assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_model_info_updates_model(self):
        exporter = VLLMExporter()

        with patch.object(exporter.client, "get", new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"id": "llama-2-70b"}]}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            await exporter.fetch_model_info()

            assert exporter.model == "llama-2-70b"

    def test_stop(self):
        exporter = VLLMExporter()
        exporter._running = True
        exporter.stop()
        assert exporter._running is False
