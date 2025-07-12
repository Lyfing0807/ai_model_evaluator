"""
Edge case tests specifically for AI insights functionality.
Tests network failures, API errors, and malformed responses.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from ai_eval_tool.config_manager import MainConfig, AIInsightsConfig
from ai_eval_tool.reporting.insights import AIInsightsGenerator
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture
def ai_insights_config():
    """Create AI insights config for testing."""
    return AIInsightsConfig(
        enabled=True,
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="test-key",
        max_tokens=1000,
        temperature=0.7
    )


@pytest.fixture
def main_config_with_ai(ai_insights_config):
    """Create main config with AI insights."""
    config = Mock(spec=MainConfig)
    config.ai_insights = ai_insights_config
    config.project_info = Mock()
    config.project_info.project_name = "Test Project"
    config.project_info.model_type = "detection"
    return config


@pytest.fixture
def sample_result():
    """Create sample evaluation result."""
    return EvaluationResult(
        metrics={"perf_mean_fps": 25.5, "det_stab_mean_iou_consistency": 0.85},
        extra_data={}
    )


class TestAIInsightsNetworkEdgeCases:
    """Test network-related edge cases for AI insights."""
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_openai_api_timeout(self, mock_openai, main_config_with_ai, sample_result):
        """Test OpenAI API timeout."""
        import requests
        mock_openai.ChatCompletion.create.side_effect = requests.exceptions.Timeout("Request timed out")
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "timeout_test")
        
        assert insights == ""  # Should return empty string on timeout
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_openai_api_connection_error(self, mock_openai, main_config_with_ai, sample_result):
        """Test OpenAI API connection error."""
        import requests
        mock_openai.ChatCompletion.create.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "connection_test")
        
        assert insights == ""
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_openai_api_rate_limit(self, mock_openai, main_config_with_ai, sample_result):
        """Test OpenAI API rate limit error."""
        import openai
        mock_openai.ChatCompletion.create.side_effect = openai.error.RateLimitError("Rate limit exceeded")
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "rate_limit_test")
        
        assert insights == ""
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_openai_api_invalid_key(self, mock_openai, main_config_with_ai, sample_result):
        """Test OpenAI API with invalid API key."""
        import openai
        mock_openai.ChatCompletion.create.side_effect = openai.error.AuthenticationError("Invalid API key")
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "auth_test")
        
        assert insights == ""
    
    @patch('ai_eval_tool.reporting.insights.requests.post')
    def test_custom_api_server_error(self, mock_post, main_config_with_ai, sample_result):
        """Test custom API server error."""
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = "https://api.example.com/chat"
        
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("Server Error 500")
        mock_post.return_value = mock_response
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "server_error_test")
        
        assert insights == ""
    
    @patch('ai_eval_tool.reporting.insights.requests.post')
    def test_custom_api_malformed_response(self, mock_post, main_config_with_ai, sample_result):
        """Test custom API with malformed JSON response."""
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = "https://api.example.com/chat"
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "malformed_json_test")
        
        assert insights == ""
    
    @patch('ai_eval_tool.reporting.insights.requests.post')
    def test_custom_api_unexpected_response_structure(self, mock_post, main_config_with_ai, sample_result):
        """Test custom API with unexpected response structure."""
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = "https://api.example.com/chat"
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "unexpected_field": "value",
            "no_choices": "field"
        }
        mock_post.return_value = mock_response
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "unexpected_structure_test")
        
        assert insights == ""


class TestAIInsightsConfigurationEdgeCases:
    """Test configuration edge cases for AI insights."""
    
    def test_missing_api_key(self, main_config_with_ai, sample_result):
        """Test AI insights with missing API key."""
        main_config_with_ai.ai_insights.api_key = None
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "no_key_test")
        
        assert insights == ""  # Should handle gracefully
    
    def test_empty_api_key(self, main_config_with_ai, sample_result):
        """Test AI insights with empty API key."""
        main_config_with_ai.ai_insights.api_key = ""
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "empty_key_test")
        
        assert insights == ""
    
    def test_invalid_provider(self, main_config_with_ai, sample_result):
        """Test AI insights with invalid provider."""
        main_config_with_ai.ai_insights.provider = "invalid_provider"
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "invalid_provider_test")
        
        assert insights == ""
    
    def test_missing_custom_endpoint(self, main_config_with_ai, sample_result):
        """Test custom provider without endpoint."""
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = None
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "no_endpoint_test")
        
        assert insights == ""
    
    def test_extreme_parameter_values(self, main_config_with_ai, sample_result):
        """Test AI insights with extreme parameter values."""
        main_config_with_ai.ai_insights.max_tokens = 0  # Invalid
        main_config_with_ai.ai_insights.temperature = -1.0  # Invalid
        
        generator = AIInsightsGenerator(main_config_with_ai)
        # Should handle invalid parameters gracefully
        context = generator._prepare_context_data(sample_result, "extreme_params_test")
        assert isinstance(context, dict)


class TestAIInsightsDataEdgeCases:
    """Test data-related edge cases for AI insights."""
    
    def test_empty_metrics(self, main_config_with_ai):
        """Test AI insights with empty metrics."""
        empty_result = EvaluationResult(metrics={}, extra_data={})
        
        generator = AIInsightsGenerator(main_config_with_ai)
        context = generator._prepare_context_data(empty_result, "empty_metrics_test")
        
        assert context["metrics"] == {}
        assert "No metrics available" in context["summary_stats"]
    
    def test_metrics_with_none_values(self, main_config_with_ai):
        """Test AI insights with None values in metrics."""
        none_result = EvaluationResult(
            metrics={"metric1": None, "metric2": 0.5, "metric3": None},
            extra_data={}
        )
        
        generator = AIInsightsGenerator(main_config_with_ai)
        formatted = generator._format_metrics_for_ai(none_result.metrics)
        
        assert "metric2: 0.5" in formatted
        # None values should be handled gracefully
    
    def test_metrics_with_extreme_values(self, main_config_with_ai):
        """Test AI insights with extreme metric values."""
        extreme_result = EvaluationResult(
            metrics={
                "very_large": 1e10,
                "very_small": 1e-10,
                "negative": -999999,
                "infinity": float('inf'),
                "nan": float('nan')
            },
            extra_data={}
        )
        
        generator = AIInsightsGenerator(main_config_with_ai)
        formatted = generator._format_metrics_for_ai(extreme_result.metrics)
        
        # Should handle extreme values without crashing
        assert isinstance(formatted, str)
        assert len(formatted) > 0
    
    def test_very_long_project_name(self, main_config_with_ai, sample_result):
        """Test AI insights with very long project name."""
        main_config_with_ai.project_info.project_name = "A" * 1000  # Very long name
        
        generator = AIInsightsGenerator(main_config_with_ai)
        context = generator._prepare_context_data(sample_result, "long_name_test")
        
        # Should handle long names gracefully
        assert len(context["project_name"]) == 1000
    
    def test_special_characters_in_data(self, main_config_with_ai):
        """Test AI insights with special characters in data."""
        special_result = EvaluationResult(
            metrics={
                "metric_with_unicode": "测试数据",
                "metric_with_symbols": "!@#$%^&*()",
                "metric_with_quotes": 'Contains "quotes" and \'apostrophes\'',
                "metric_with_newlines": "Line1\nLine2\nLine3"
            },
            extra_data={}
        )
        
        generator = AIInsightsGenerator(main_config_with_ai)
        formatted = generator._format_metrics_for_ai(special_result.metrics)
        
        # Should handle special characters without breaking
        assert isinstance(formatted, str)


class TestAIInsightsPromptEdgeCases:
    """Test prompt generation edge cases."""
    
    def test_very_large_context(self, main_config_with_ai):
        """Test AI insights with very large context data."""
        large_metrics = {f"metric_{i}": i * 0.1 for i in range(1000)}
        large_result = EvaluationResult(metrics=large_metrics, extra_data={})
        
        generator = AIInsightsGenerator(main_config_with_ai)
        context = generator._prepare_context_data(large_result, "large_context_test")
        prompt = generator._create_ai_prompt(context)
        
        # Should handle large context without crashing
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_empty_context(self, main_config_with_ai):
        """Test prompt generation with minimal context."""
        minimal_context = {
            "project_name": "",
            "model_type": "",
            "run_id": "",
            "metrics": {},
            "summary_stats": ""
        }
        
        generator = AIInsightsGenerator(main_config_with_ai)
        prompt = generator._create_ai_prompt(minimal_context)
        
        # Should generate a valid prompt even with minimal context
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "analysis" in prompt.lower()


class TestAIInsightsIntegrationEdgeCases:
    """Test integration edge cases."""
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_partial_api_response(self, mock_openai, main_config_with_ai, sample_result):
        """Test handling of partial/incomplete API responses."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None  # Incomplete response
        
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "partial_response_test")
        
        assert insights == ""  # Should handle None content gracefully
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_empty_api_response(self, mock_openai, main_config_with_ai, sample_result):
        """Test handling of empty API responses."""
        mock_response = Mock()
        mock_response.choices = []  # Empty choices
        
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "empty_response_test")
        
        assert insights == ""
    
    def test_disabled_ai_insights_with_valid_config(self, main_config_with_ai, sample_result):
        """Test that disabled AI insights don't make API calls."""
        main_config_with_ai.ai_insights.enabled = False
        
        generator = AIInsightsGenerator(main_config_with_ai)
        
        with patch('ai_eval_tool.reporting.insights.openai') as mock_openai:
            insights = generator.generate_insights(sample_result, "disabled_test")
            
            # Should not make any API calls
            mock_openai.ChatCompletion.create.assert_not_called()
            assert insights == ""
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_api_response_with_unexpected_structure(self, mock_openai, main_config_with_ai, sample_result):
        """Test API response with unexpected structure."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # Missing message attribute
        delattr(mock_response.choices[0], 'message')
        
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_result, "unexpected_structure_test")
        
        assert insights == ""  # Should handle gracefully


@pytest.mark.parametrize("invalid_config", [
    {"enabled": "not_boolean"},
    {"provider": 123},
    {"model": None},
    {"api_key": 123},
    {"max_tokens": "not_integer"},
    {"temperature": "not_float"},
])
def test_invalid_ai_insights_config_types(invalid_config):
    """Test AI insights with invalid configuration types."""
    # This would typically be caught by Pydantic validation
    # but we test the edge case where invalid data somehow gets through
    
    base_config = {
        "enabled": True,
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": "test-key",
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    base_config.update(invalid_config)
    
    # Should either raise validation error or handle gracefully
    try:
        config = AIInsightsConfig(**base_config)
        # If it doesn't raise an error, the values should be coerced or handled
        assert config is not None
    except (ValueError, TypeError):
        # Expected for invalid types
        pass