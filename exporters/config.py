from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vllm_endpoint: str = "http://localhost:8000"
    tgi_endpoint: str = "http://localhost:8080"
    prometheus_port: int = 9090
    grafana_port: int = 3000
    scrape_interval: str = "15s"
    exporter_port_vllm: int = 8000
    exporter_port_tgi: int = 8001
    exporter_port_gpu: int = 9400
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
