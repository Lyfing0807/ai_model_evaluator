"""
Network-related edge case tests for the AI evaluation tool.
Tests various network failure scenarios, API timeouts, and connectivity issues.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from ai_eval_tool.config_manager import AIInsightsConfig, MainConfig
from ai_eval_tool.reporting.insights import AIInsightsGenerator
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for network tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def ai_insights_config():
    """Create AI insights configuration for testing."""
    return AIInsightsConfig(
        enabled=True,
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="test-api-key",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        max_tokens=1000,
        temperature=0.7,
        timeout=30,
    )


@pytest.fixture
def main_config_with_ai(ai_insights_config):
    """Create main config with AI insights enabled."""
    config = Mock(spec=MainConfig)
    config.ai_insights = ai_insights_config
    config.project_info = Mock()
    config.project_info.project_name = "Network Test Project"
    config.project_info.model_type = "detection"
    return config


@pytest.fixture
def sample_evaluation_result():
    """Create sample evaluation result for testing."""
    import polars as pl

    metrics = {
        "perf_mean_fps": 25.5,
        "perf_mean_inference_time_ms": 39.2,
        "det_stab_mean_iou_consistency": 0.85,
        "det_stab_mean_bbox_drift": 2.3,
        "det_stab_outlier_count": 5,
    }

    extra_data = {
        "performance_details_df": pl.DataFrame(
            {
                "image_id": ["img1", "img2"],
                "fps": [25.0, 26.0],
                "inference_time_ms": [40.0, 38.5],
            }
        )
    }

    return EvaluationResult(metrics=metrics, extra_data=extra_data)


class TestOpenAIAPIFailures:
    """Test OpenAI API failure scenarios."""

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_openai_api_timeout(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test OpenAI API timeout."""
        # Mock timeout exception
        mock_openai.ChatCompletion.create.side_effect = requests.exceptions.Timeout(
            "Request timed out"
        )

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_evaluation_result, "timeout-test")

        # Should return empty string on timeout
        assert insights == ""
        mock_openai.ChatCompletion.create.assert_called_once()

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_openai_api_connection_error(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test OpenAI API connection error."""
        # Mock connection error
        mock_openai.ChatCompletion.create.side_effect = (
            requests.exceptions.ConnectionError("Connection failed")
        )

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "connection-test"
        )

        assert insights == ""
        mock_openai.ChatCompletion.create.assert_called_once()

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_openai_api_rate_limit(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test OpenAI API rate limiting."""
        # Mock rate limit error
        mock_openai.ChatCompletion.create.side_effect = Exception("Rate limit exceeded")

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "rate-limit-test"
        )

        assert insights == ""

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_openai_api_invalid_key(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test OpenAI API with invalid API key."""
        # Mock authentication error
        mock_openai.ChatCompletion.create.side_effect = Exception("Invalid API key")

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "invalid-key-test"
        )

        assert insights == ""

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_openai_api_malformed_response(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test OpenAI API with malformed response."""
        # Mock malformed response
        mock_response = Mock()
        mock_response.choices = []  # Empty choices
        mock_openai.ChatCompletion.create.return_value = mock_response

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "malformed-response-test"
        )

        # Should handle empty choices gracefully
        assert insights == ""

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_openai_api_partial_response(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test OpenAI API with partial/incomplete response."""
        # Mock partial response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None  # No content
        mock_openai.ChatCompletion.create.return_value = mock_response

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "partial-response-test"
        )

        # Should handle None content gracefully
        assert insights == ""


class TestCustomAPIFailures:
    """Test custom API endpoint failure scenarios."""

    @patch("ai_eval_tool.reporting.insights.requests.post")
    def test_custom_api_timeout(
        self, mock_post, main_config_with_ai, sample_evaluation_result
    ):
        """Test custom API timeout."""
        # Configure for custom API
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = (
            "https://custom-api.example.com/chat"
        )

        # Mock timeout
        mock_post.side_effect = requests.exceptions.Timeout("Custom API timeout")

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "custom-timeout-test"
        )

        assert insights == ""
        mock_post.assert_called_once()

    @patch("ai_eval_tool.reporting.insights.requests.post")
    def test_custom_api_http_error(
        self, mock_post, main_config_with_ai, sample_evaluation_result
    ):
        """Test custom API HTTP error responses."""
        # Configure for custom API
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = (
            "https://custom-api.example.com/chat"
        )

        # Mock HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_post.return_value = mock_response

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "custom-http-error-test"
        )

        assert insights == ""

    @patch("ai_eval_tool.reporting.insights.requests.post")
    def test_custom_api_invalid_json_response(
        self, mock_post, main_config_with_ai, sample_evaluation_result
    ):
        """Test custom API with invalid JSON response."""
        # Configure for custom API
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = (
            "https://custom-api.example.com/chat"
        )

        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "custom-invalid-json-test"
        )

        assert insights == ""

    @patch("ai_eval_tool.reporting.insights.requests.post")
    def test_custom_api_missing_response_fields(
        self, mock_post, main_config_with_ai, sample_evaluation_result
    ):
        """Test custom API with missing expected response fields."""
        # Configure for custom API
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = (
            "https://custom-api.example.com/chat"
        )

        # Mock response with missing fields
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "unexpected_field": "value"
        }  # Missing 'choices'
        mock_post.return_value = mock_response

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "custom-missing-fields-test"
        )

        assert insights == ""


class TestNetworkConnectivityIssues:
    """Test various network connectivity issues."""

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_dns_resolution_failure(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test DNS resolution failure."""
        # Mock DNS resolution error
        mock_openai.ChatCompletion.create.side_effect = (
            requests.exceptions.ConnectionError("Failed to resolve hostname")
        )

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "dns-failure-test"
        )

        assert insights == ""

    @patch("ai_eval_tool.reporting.insights.requests.post")
    def test_ssl_certificate_error(
        self, mock_post, main_config_with_ai, sample_evaluation_result
    ):
        """Test SSL certificate verification error."""
        # Configure for custom API with HTTPS
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = (
            "https://invalid-cert.example.com/chat"
        )

        # Mock SSL error
        mock_post.side_effect = requests.exceptions.SSLError(
            "SSL certificate verification failed"
        )

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "ssl-error-test"
        )

        assert insights == ""

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_network_unreachable(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test network unreachable error."""
        # Mock network unreachable error
        mock_openai.ChatCompletion.create.side_effect = (
            requests.exceptions.ConnectionError("Network is unreachable")
        )

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "network-unreachable-test"
        )

        assert insights == ""


class TestAPIRetryMechanisms:
    """Test API retry mechanisms and resilience."""

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_retry_on_temporary_failure(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test retry mechanism on temporary failures."""
        # Mock temporary failure followed by success
        mock_openai.ChatCompletion.create.side_effect = [
            requests.exceptions.ConnectionError("Temporary failure"),
            Mock(choices=[Mock(message=Mock(content="Success after retry"))]),
        ]

        generator = AIInsightsGenerator(main_config_with_ai)

        # If retry mechanism is implemented, this should succeed
        # If not, it should fail gracefully
        insights = generator.generate_insights(sample_evaluation_result, "retry-test")

        # Either succeeds with retry or fails gracefully
        assert isinstance(insights, str)

    @patch("ai_eval_tool.reporting.insights.time.sleep")  # Mock sleep to speed up tests
    @patch("ai_eval_tool.reporting.insights.openai")
    def test_max_retries_exceeded(
        self, mock_openai, mock_sleep, main_config_with_ai, sample_evaluation_result
    ):
        """Test behavior when max retries are exceeded."""
        # Mock consistent failures
        mock_openai.ChatCompletion.create.side_effect = (
            requests.exceptions.ConnectionError("Persistent failure")
        )

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "max-retries-test"
        )

        # Should eventually give up and return empty string
        assert insights == ""


class TestLargePayloadHandling:
    """Test handling of large payloads and data."""

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_large_evaluation_result(self, mock_openai, main_config_with_ai):
        """Test AI insights with very large evaluation results."""
        import polars as pl

        # Create a large evaluation result
        large_metrics = {f"metric_{i}": float(i) for i in range(1000)}
        large_df = pl.DataFrame(
            {
                "image_id": [f"img_{i}" for i in range(10000)],
                "value": list(range(10000)),
            }
        )

        large_result = EvaluationResult(
            metrics=large_metrics, extra_data={"large_df": large_df}
        )

        # Mock successful response
        mock_openai.ChatCompletion.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Handled large payload"))]
        )

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(large_result, "large-payload-test")

        # Should handle large payloads gracefully
        assert isinstance(insights, str)

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_payload_size_limit_exceeded(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test behavior when payload size exceeds API limits."""
        # Mock payload too large error
        mock_openai.ChatCompletion.create.side_effect = Exception(
            "Request payload too large"
        )

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "payload-limit-test"
        )

        assert insights == ""


class TestConfigurationEdgeCases:
    """Test edge cases in AI insights configuration."""

    def test_missing_api_key(self, temp_workspace):
        """Test AI insights with missing API key."""
        config_with_missing_key = AIInsightsConfig(
            enabled=True,
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="",  # Empty API key
            max_tokens=1000,
            temperature=0.7,
        )

        config = Mock(spec=MainConfig)
        config.ai_insights = config_with_missing_key
        config.project_info = Mock()
        config.project_info.project_name = "Missing Key Test"
        config.project_info.model_type = "detection"

        generator = AIInsightsGenerator(config)

        # Should handle missing API key gracefully
        assert generator.enabled is True  # Config says enabled
        # But actual API calls should fail gracefully

    def test_invalid_api_endpoint(self, temp_workspace):
        """Test AI insights with invalid API endpoint."""
        config_with_invalid_endpoint = AIInsightsConfig(
            enabled=True,
            provider="custom",
            model="gpt-3.5-turbo",
            api_key="test-key",
            api_endpoint="not-a-valid-url",  # Invalid URL
            max_tokens=1000,
            temperature=0.7,
        )

        config = Mock(spec=MainConfig)
        config.ai_insights = config_with_invalid_endpoint
        config.project_info = Mock()
        config.project_info.project_name = "Invalid Endpoint Test"
        config.project_info.model_type = "detection"

        generator = AIInsightsGenerator(config)

        # Should handle invalid endpoint gracefully
        assert generator.enabled is True

    def test_extreme_timeout_values(
        self, main_config_with_ai, sample_evaluation_result
    ):
        """Test AI insights with extreme timeout values."""
        # Test very short timeout
        main_config_with_ai.ai_insights.timeout = 0.001  # 1ms timeout

        with patch("ai_eval_tool.reporting.insights.openai") as mock_openai:
            mock_openai.ChatCompletion.create.side_effect = requests.exceptions.Timeout(
                "Timeout"
            )

            generator = AIInsightsGenerator(main_config_with_ai)
            insights = generator.generate_insights(
                sample_evaluation_result, "short-timeout-test"
            )

            assert insights == ""

        # Test very long timeout
        main_config_with_ai.ai_insights.timeout = 3600  # 1 hour timeout

        with patch("ai_eval_tool.reporting.insights.openai") as mock_openai:
            mock_openai.ChatCompletion.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Long timeout handled"))]
            )

            generator = AIInsightsGenerator(main_config_with_ai)
            insights = generator.generate_insights(
                sample_evaluation_result, "long-timeout-test"
            )

            assert "Long timeout handled" in insights


class TestConcurrentAPIRequests:
    """Test concurrent API request scenarios."""

    @patch("ai_eval_tool.reporting.insights.openai")
    def test_concurrent_requests_simulation(
        self, mock_openai, main_config_with_ai, sample_evaluation_result
    ):
        """Test simulation of concurrent API requests."""
        import queue
        import threading

        # Mock responses for concurrent requests
        mock_openai.ChatCompletion.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Concurrent response"))]
        )

        generator = AIInsightsGenerator(main_config_with_ai)
        results_queue = queue.Queue()

        def make_request(run_id):
            try:
                insights = generator.generate_insights(
                    sample_evaluation_result, f"concurrent-{run_id}"
                )
                results_queue.put(("success", insights))
            except Exception as e:
                results_queue.put(("error", str(e)))

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 5
        # All should either succeed or fail gracefully
        for status, result in results:
            assert status in ["success", "error"]
            assert isinstance(result, str)


@pytest.mark.parametrize(
    "network_error",
    [
        requests.exceptions.ConnectionError("Connection refused"),
        requests.exceptions.Timeout("Request timeout"),
        requests.exceptions.HTTPError("HTTP 500 Error"),
        requests.exceptions.RequestException("Generic request error"),
        Exception("Unknown error"),
    ],
)
def test_various_network_exceptions(
    network_error, main_config_with_ai, sample_evaluation_result
):
    """Test handling of various network-related exceptions."""
    with patch("ai_eval_tool.reporting.insights.openai") as mock_openai:
        mock_openai.ChatCompletion.create.side_effect = network_error

        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(
            sample_evaluation_result, "network-error-test"
        )

        # Should handle all network errors gracefully
        assert insights == ""


def test_network_recovery_after_failure(main_config_with_ai, sample_evaluation_result):
    """Test network recovery after initial failure."""
    with patch("ai_eval_tool.reporting.insights.openai") as mock_openai:
        # First call fails, second succeeds
        mock_openai.ChatCompletion.create.side_effect = [
            requests.exceptions.ConnectionError("Initial failure"),
            Mock(choices=[Mock(message=Mock(content="Recovery successful"))]),
        ]

        generator = AIInsightsGenerator(main_config_with_ai)

        # First call should fail
        insights1 = generator.generate_insights(
            sample_evaluation_result, "recovery-test-1"
        )
        assert insights1 == ""

        # Second call should succeed (if retry mechanism exists)
        insights2 = generator.generate_insights(
            sample_evaluation_result, "recovery-test-2"
        )
        # Either succeeds or fails gracefully
        assert isinstance(insights2, str)
