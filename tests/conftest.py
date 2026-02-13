import pytest


@pytest.fixture
def mock_vllm_metrics():
    return """# HELP vllm:num_requests_running Number of requests currently running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 5
# HELP vllm:num_requests_waiting Number of requests waiting in queue
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 10
# HELP vllm:num_batched_tokens Number of tokens batched
# TYPE vllm:num_batched_tokens gauge
vllm:num_batched_tokens 512
# HELP vllm:gpu_cache_usage_perc GPU cache usage percentage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 75.5
# HELP vllm:time_to_first_token_seconds Time to first token
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} 100
vllm:time_to_first_token_seconds_bucket{le="0.5"} 250
vllm:time_to_first_token_seconds_bucket{le="1.0"} 400
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 500
vllm:time_to_first_token_seconds_sum 250.5
vllm:time_to_first_token_seconds_count 500
# HELP vllm:time_per_output_token_seconds Time per output token
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.01"} 1000
vllm:time_per_output_token_seconds_bucket{le="0.05"} 2000
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 2500
vllm:time_per_output_token_seconds_sum 50.5
vllm:time_per_output_token_seconds_count 2500
# HELP vllm:total_tokens Total tokens
# TYPE vllm:total_tokens counter
vllm:total_tokens 100000
"""


@pytest.fixture
def mock_tgi_metrics():
    return """# HELP tgi_queue_size Queue size
# TYPE tgi_queue_size gauge
tgi_queue_size 5
# HELP tgi_batch_size Current batch size
# TYPE tgi_batch_size gauge
tgi_batch_size 32
# HELP tgi_request_count Active requests
# TYPE tgi_request_count gauge
tgi_request_count 4
# HELP tgi_decoder_tokens Decoder tokens
# TYPE tgi_decoder_tokens counter
tgi_decoder_tokens 50000
# HELP tgi_prefill_tokens Prefill tokens
# TYPE tgi_prefill_tokens counter
tgi_prefill_tokens 10000
# HELP tgi_request_success Successful requests
# TYPE tgi_request_success counter
tgi_request_success 1000
# HELP tgi_request_failure Failed requests
# TYPE tgi_request_failure counter
tgi_request_failure 5
"""


@pytest.fixture
def mock_nvidia_smi_xml():
    return """<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v10.dtd">
<nvidia_smi_log>
    <gpu id="00000000:00:04.0">
        <gpu_id>0</gpu_id>
        <product_name>NVIDIA A100-SXM4-80GB</product_name>
        <uuid>GPU-12345678-1234-1234-1234-123456789012</uuid>
        <fb_memory_usage>
            <total>81920 MiB</total>
            <used>40960 MiB</used>
            <free>40960 MiB</free>
        </fb_memory_usage>
        <utilization>
            <gpu_util>75 %</gpu_util>
            <memory_util>60 %</memory_util>
        </utilization>
        <temperature>
            <gpu_temp>65</gpu_temp>
        </temperature>
        <power_readings>
            <power_draw>250.50 W</power_draw>
            <power_limit>400.00 W</power_limit>
        </power_readings>
        <fan_speed>50 %</fan_speed>
        <clocks>
            <sm_clock>1410 MHz</sm_clock>
            <mem_clock>1215 MHz</mem_clock>
        </clocks>
        <processes>
            <process_info>
                <pid>12345</pid>
            </process_info>
        </processes>
    </gpu>
</nvidia_smi_log>
"""
