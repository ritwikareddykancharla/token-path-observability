import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from exporters.tgi_exporter.exporter import TGIExporter


class TestTGIExporter:
    def test_init(self):
        exporter = TGIExporter(endpoint="http://localhost:8080", port=9001, model="test-model")
        assert exporter.endpoint == "http://localhost:8080"
        assert exporter.port == 9001
        assert exporter.model == "test-model"

    def test_init_strips_trailing_slash(self):
        exporter = TGIExporter(endpoint="http://localhost:8080/")
        assert exporter.endpoint == "http://localhost:8080"

    def test_parse_prometheus_metrics(self, mock_tgi_metrics):
        exporter = TGIExporter()
        result = exporter._parse_prometheus_metrics(mock_tgi_metrics)

        assert "tgi_queue_size" in result
        assert result["tgi_queue_size"] == 5.0
        assert "tgi_batch_size" in result
        assert result["tgi_batch_size"] == 32.0

    def test_parse_prometheus_metrics_empty_input(self):
        exporter = TGIExporter()
        result = exporter._parse_prometheus_metrics("")
        assert result == {}

    def test_extract_metric_value_simple(self):
        exporter = TGIExporter()
        metrics = {"simple_metric": 42.0}
        result = exporter._extract_metric_value(metrics, "simple_metric")
        assert result == 42.0

    def test_extract_metric_with_labels_no_filter(self):
        exporter = TGIExporter()
        metrics = {
            "labeled_metric": [
                {"labels": {"status": "success"}, "value": 100.0},
                {"labels": {"status": "failed"}, "value": 10.0},
            ]
        }
        result = exporter._extract_metric_with_labels(metrics, "labeled_metric")
        assert len(result) == 2

    def test_extract_metric_with_labels_with_filter(self):
        exporter = TGIExporter()
        metrics = {
            "labeled_metric": [
                {"labels": {"status": "success"}, "value": 100.0},
                {"labels": {"status": "failed"}, "value": 10.0},
            ]
        }
        result = exporter._extract_metric_with_labels(
            metrics, "labeled_metric", {"status": "success"}
        )
        assert len(result) == 1
        assert result[0]["value"] == 100.0

    def test_update_prometheus_metrics(self, mock_tgi_metrics):
        exporter = TGIExporter(model="test-model")
        metrics = exporter._parse_prometheus_metrics(mock_tgi_metrics)

        exporter.update_prometheus_metrics(metrics)

    @pytest.mark.asyncio
    async def test_fetch_metrics_success(self, mock_tgi_metrics):
        exporter = TGIExporter()

        with patch.object(exporter.client, "get", new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_tgi_metrics
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await exporter.fetch_metrics()

            assert "tgi_queue_size" in result
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_metrics_http_error(self):
        exporter = TGIExporter()

        with patch.object(exporter.client, "get", new_callable=AsyncMock) as mock_get:
            import httpx

            mock_get.side_effect = httpx.HTTPError("Connection failed")

            result = await exporter.fetch_metrics()

            assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_model_info_updates_model(self):
        exporter = TGIExporter()

        with patch.object(exporter.client, "get", new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"model_id": "meta-llama/Llama-2-70b-hf"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            await exporter.fetch_model_info()

            assert exporter.model == "meta-llama/Llama-2-70b-hf"

    def test_stop(self):
        exporter = TGIExporter()
        exporter._running = True
        exporter.stop()
        assert exporter._running is False
