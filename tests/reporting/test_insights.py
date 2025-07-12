"""
Tests for AI insights functionality.
"""
import pytest
import polars as pl
from unittest.mock import Mock, patch
from pathlib import Path

from ai_eval_tool.config_manager import MainConfig, AIInsightsConfig
from ai_eval_tool.reporting.insights import AIInsightsGenerator
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture
def ai_insights_config():
    """Create a test configuration with AI insights enabled."""
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
    """Create a main config with AI insights enabled."""
    config = Mock(spec=MainConfig)
    config.ai_insights = ai_insights_config
    config.project_info = Mock()
    config.project_info.project_name = "Test Project"
    config.project_info.model_type = "detection"
    return config


@pytest.fixture
def sample_evaluation_result():
    """Create a sample evaluation result for testing."""
    metrics = {
        "perf_mean_fps": 25.5,
        "perf_mean_inference_time_ms": 39.2,
        "det_stab_mean_iou_consistency": 0.85,
        "det_stab_mean_bbox_drift": 2.3,
        "det_stab_outlier_count": 5
    }
    
    extra_data = {
        "performance_details_df": pl.DataFrame({
            "image_id": ["img1", "img2", "img3"],
            "fps": [25.0, 26.0, 25.5],
            "inference_time_ms": [40.0, 38.5, 39.2]
        }),
        "detection_stability_details_df": pl.DataFrame({
            "image_id": ["img1", "img2", "img3"],
            "iou_consistency": [0.9, 0.8, 0.85],
            "bbox_drift": [1.5, 3.1, 2.3]
        })
    }
    
    return EvaluationResult(metrics=metrics, extra_data=extra_data)


class TestAIInsightsGenerator:
    """Test cases for AI insights generation."""
    
    def test_init_with_enabled_config(self, main_config_with_ai):
        """Test initialization with AI insights enabled."""
        generator = AIInsightsGenerator(main_config_with_ai)
        assert generator.enabled is True
        assert generator.provider == "openai"
        assert generator.model == "gpt-3.5-turbo"
    
    def test_init_with_disabled_config(self):
        """Test initialization with AI insights disabled."""
        config = Mock(spec=MainConfig)
        config.ai_insights = AIInsightsConfig(enabled=False)
        
        generator = AIInsightsGenerator(config)
        assert generator.enabled is False
    
    def test_generate_insights_disabled(self):
        """Test that no insights are generated when disabled."""
        config = Mock(spec=MainConfig)
        config.ai_insights = AIInsightsConfig(enabled=False)
        
        generator = AIInsightsGenerator(config)
        result = Mock(spec=EvaluationResult)
        
        insights = generator.generate_insights(result, "test-run")
        assert insights == ""
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_generate_insights_openai_success(self, mock_openai, main_config_with_ai, sample_evaluation_result):
        """Test successful AI insights generation with OpenAI."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "AI generated insights about the model performance."
        
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_evaluation_result, "test-run-123")
        
        assert "AI generated insights" in insights
        mock_openai.ChatCompletion.create.assert_called_once()
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_generate_insights_openai_error(self, mock_openai, main_config_with_ai, sample_evaluation_result):
        """Test AI insights generation with OpenAI API error."""
        # Mock OpenAI error
        mock_openai.ChatCompletion.create.side_effect = Exception("API Error")
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_evaluation_result, "test-run-123")
        
        assert insights == ""  # Should return empty string on error
    
    def test_prepare_context_data(self, main_config_with_ai, sample_evaluation_result):
        """Test context data preparation for AI insights."""
        generator = AIInsightsGenerator(main_config_with_ai)
        context = generator._prepare_context_data(sample_evaluation_result, "test-run-123")
        
        assert "project_name" in context
        assert "model_type" in context
        assert "run_id" in context
        assert "metrics" in context
        assert "summary_stats" in context
        
        assert context["project_name"] == "Test Project"
        assert context["model_type"] == "detection"
        assert context["run_id"] == "test-run-123"
        assert context["metrics"]["perf_mean_fps"] == 25.5
    
    def test_format_metrics_for_ai(self, main_config_with_ai, sample_evaluation_result):
        """Test metrics formatting for AI consumption."""
        generator = AIInsightsGenerator(main_config_with_ai)
        formatted = generator._format_metrics_for_ai(sample_evaluation_result.metrics)
        
        assert "Performance Metrics:" in formatted
        assert "perf_mean_fps: 25.5" in formatted
        assert "Stability Metrics:" in formatted
        assert "det_stab_mean_iou_consistency: 0.85" in formatted
    
    def test_create_ai_prompt(self, main_config_with_ai):
        """Test AI prompt creation."""
        generator = AIInsightsGenerator(main_config_with_ai)
        context = {
            "project_name": "Test Project",
            "model_type": "detection",
            "run_id": "test-123",
            "metrics": {"perf_mean_fps": 25.5},
            "summary_stats": "Sample stats"
        }
        
        prompt = generator._create_ai_prompt(context)
        
        assert "Test Project" in prompt
        assert "detection" in prompt
        assert "test-123" in prompt
        assert "performance analysis" in prompt.lower()
    
    @patch('ai_eval_tool.reporting.insights.requests.post')
    def test_generate_insights_custom_api(self, mock_post, main_config_with_ai, sample_evaluation_result):
        """Test AI insights generation with custom API."""
        # Configure for custom API
        main_config_with_ai.ai_insights.provider = "custom"
        main_config_with_ai.ai_insights.api_endpoint = "https://api.example.com/chat"
        
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Custom API insights"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_evaluation_result, "test-run-123")
        
        assert "Custom API insights" in insights
        mock_post.assert_called_once()


class TestAIInsightsIntegration:
    """Integration tests for AI insights with the full pipeline."""
    
    @patch('ai_eval_tool.reporting.insights.openai')
    def test_end_to_end_insights_generation(self, mock_openai, main_config_with_ai, sample_evaluation_result):
        """Test end-to-end AI insights generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = """
        ## AI Analysis Summary
        
        The model shows good performance with an average FPS of 25.5, indicating efficient inference.
        The stability metrics reveal consistent behavior with IoU consistency of 0.85.
        
        ### Key Findings:
        - Performance is within acceptable range
        - Stability shows room for improvement
        - 5 outliers detected, requiring investigation
        
        ### Recommendations:
        - Monitor bbox drift patterns
        - Consider model fine-tuning for outlier cases
        """
        
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        generator = AIInsightsGenerator(main_config_with_ai)
        insights = generator.generate_insights(sample_evaluation_result, "integration-test-run")
        
        assert "AI Analysis Summary" in insights
        assert "Key Findings" in insights
        assert "Recommendations" in insights
        assert "25.5" in insights  # FPS value should be mentioned
        assert "0.85" in insights  # IoU consistency should be mentioned
    
    def test_insights_with_missing_data(self, main_config_with_ai):
        """Test AI insights generation with missing or incomplete data."""
        # Create evaluation result with minimal data
        minimal_result = EvaluationResult(
            metrics={"perf_mean_fps": 20.0},
            extra_data={}
        )
        
        generator = AIInsightsGenerator(main_config_with_ai)
        
        # Should not crash with minimal data
        context = generator._prepare_context_data(minimal_result, "minimal-test")
        assert context["metrics"]["perf_mean_fps"] == 20.0
        assert context["summary_stats"] == "No detailed statistics available."


@pytest.fixture
def ai_insights_test_config_path(tmp_path):
    """Create a temporary config file with AI insights enabled for testing."""
    config_content = """
project_info:
  project_name: "AI Insights Test"
  model_type: "detection"

data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"

evaluation_params:
  detection:
    iou_threshold: 0.5

ai_insights:
  enabled: true
  provider: "openai"
  model: "gpt-3.5-turbo"
  api_key: "test-key"
  max_tokens: 1000
  temperature: 0.7

report_settings:
  output_dir: "test_output"
"""
    
    config_path = tmp_path / "ai_insights_test_config.yaml"
    config_path.write_text(config_content)
    return config_path


def test_ai_insights_config_loading(ai_insights_test_config_path):
    """Test loading AI insights configuration from YAML."""
    from ai_eval_tool.config_manager import load_config
    
    config = load_config(ai_insights_test_config_path)
    
    assert config.ai_insights.enabled is True
    assert config.ai_insights.provider == "openai"
    assert config.ai_insights.model == "gpt-3.5-turbo"
    assert config.ai_insights.api_key == "test-key"
    assert config.ai_insights.max_tokens == 1000
    assert config.ai_insights.temperature == 0.7