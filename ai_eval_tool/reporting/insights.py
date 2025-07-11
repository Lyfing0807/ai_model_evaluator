"""
AI Insights Generation: Connects to LLMs to provide text interpretations of charts/data.
(Experimental Feature)
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import httpx # For making API calls to LLM services

from ..config_manager import AIInsightsConfig, MainConfig # For API key, endpoint, model
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class AIInsightsGenerator:
    """
    Handles the generation of AI-powered textual insights for charts or data summaries
    by querying an external Large Language Model (LLM).
    """
    def __init__(self, insights_config: AIInsightsConfig, project_config: MainConfig):
        self.config = insights_config
        self.project_config = project_config # To access API key if needed
        self.api_key = self._get_api_key() # Placeholder for actual API key retrieval

        self.cache_dir = Path(project_config.report_settings.output_dir) / ".insights_cache"
        if self.config.cache_responses:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AIInsightsGenerator initialized. Caching enabled: {self.config.cache_responses}")

    def _get_api_key(self) -> Optional[str]:
        """
        Retrieves the API key for the LLM service.
        Recommended: from environment variable.
        This is a placeholder; actual implementation depends on how API key is provided.
        """
        # Example: OPENAI_API_KEY is a common env var name
        # The actual env var name could be made configurable.
        import os
        api_key_env_var = "OPENAI_API_KEY" # This could be part of insights_config
        key = os.getenv(api_key_env_var)
        if not key:
            logger.warning(
                f"API Key environment variable '{api_key_env_var}' not set. "
                "AI Insights will likely fail if API calls are attempted."
            )
        return key

    def _build_prompt(self, chart_type: str, chart_data_summary: str, context: str) -> str:
        """
        Constructs a prompt for the LLM based on chart type, data, and context.
        """
        # This is a very generic prompt, needs to be tailored for specific charts/data.
        prompt = (
            f"You are an expert AI model performance analyst.\n"
            f"The following data pertains to a chart of type: '{chart_type}'.\n"
            f"Context: {context}\n"
            f"Data Summary: {chart_data_summary}\n\n"
            f"Please provide a brief (2-3 sentences) interpretation of this data, highlighting key insights or potential issues. "
            f"Focus on what this data implies about the AI model's performance or stability. "
            f"Be concise and data-driven. If the data is inconclusive, state that."
        )
        logger.debug(f"Generated LLM Prompt for {chart_type}:\n{prompt[:200]}...") # Log snippet
        return prompt

    def _make_llm_api_call(self, prompt: str) -> Optional[str]:
        """
        Makes the API call to the configured LLM endpoint.
        """
        if not self.api_key:
            logger.error("LLM API key not available. Cannot make API call.")
            return None
        if not self.config.api_endpoint:
            logger.error("LLM API endpoint not configured. Cannot make API call.")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Payload structure depends on the specific LLM API (e.g., OpenAI's chat completions)
        payload = {
            "model": self.config.model_name or "gpt-3.5-turbo", # Default if not specified
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150, # Adjust as needed
            "temperature": 0.5, # Adjust for creativity vs. factuality
        }

        logger.info(f"Sending request to LLM API: {self.config.api_endpoint} with model: {payload['model']}")
        try:
            with httpx.Client(timeout=30.0) as client: # Timeout of 30 seconds
                response = client.post(self.config.api_endpoint, headers=headers, json=payload)
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses

            response_data = response.json()
            # Extract text from response (this is specific to OpenAI's API structure)
            # Example for OpenAI chat completions:
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                insight_text = response_data["choices"][0].get("message", {}).get("content")
                if insight_text:
                    logger.info("Successfully received insight from LLM.")
                    logger.debug(f"LLM Raw Insight: {insight_text[:200]}...")
                    return insight_text.strip()
            logger.warning(f"LLM response format unexpected: {response_data}")
            return None
        except httpx.RequestError as e:
            logger.error(f"HTTP request to LLM API failed: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API returned an error: {e.response.status_code} - {e.response.text}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response from LLM API: {response.text}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM API call: {e}", exc_info=True)
        return None

    def _generate_cache_key(self, chart_type: str, chart_data_summary: str, context: str, prompt_template_version: str = "v1") -> str:
        """Generates a cache key based on inputs to ensure reproducibility."""
        hasher = hashlib.sha256()
        hasher.update(chart_type.encode())
        hasher.update(chart_data_summary.encode()) # Ensure chart_data_summary is a stable string representation
        hasher.update(context.encode())
        hasher.update(prompt_template_version.encode())
        if self.config.model_name: # Include model name in cache key
            hasher.update(self.config.model_name.encode())
        return hasher.hexdigest()

    def get_insight_for_chart(
        self,
        chart_type: str,
        chart_data: Any, # This could be a DataFrame, dict of series, etc.
        context_description: str = "General model evaluation context."
    ) -> Optional[str]:
        """
        Generates (or retrieves from cache) an AI-powered insight for a given chart.

        Args:
            chart_type: A string identifying the type of chart (e.g., "Latency Distribution").
            chart_data: The data used to generate the chart. Needs to be summarized into a string.
            context_description: A brief description of the evaluation context.

        Returns:
            A string containing the AI-generated insight, or None if generation fails.
        """
        if not self.config.enabled:
            logger.debug("AI Insights are disabled in configuration.")
            return None

        # Serialize chart_data to a stable string format for prompt and caching
        # This is crucial and might need custom logic per chart_type/data_type
        try:
            # Example serialization: if chart_data is a Polars DF's describe() output or small dict
            if hasattr(chart_data, 'to_dict'): # Polars describe output
                chart_data_summary = json.dumps(chart_data.to_dicts(), sort_keys=True)
            elif isinstance(chart_data, dict):
                chart_data_summary = json.dumps(chart_data, sort_keys=True)
            else: # Fallback, may not be ideal
                chart_data_summary = str(chart_data)
        except Exception as e:
            logger.error(f"Failed to serialize chart_data for AI insight: {e}")
            return None


        cache_key = self._generate_cache_key(chart_type, chart_data_summary, context_description)
        cache_file = self.cache_dir / f"{cache_key}.txt"

        if self.config.cache_responses and cache_file.exists():
            try:
                insight = cache_file.read_text(encoding="utf-8")
                logger.info(f"Retrieved insight for '{chart_type}' from cache.")
                return insight
            except Exception as e:
                logger.warning(f"Failed to read insight from cache file {cache_file}: {e}")

        prompt = self._build_prompt(chart_type, chart_data_summary, context_description)
        insight_text = self._make_llm_api_call(prompt)

        if insight_text and self.config.cache_responses:
            try:
                cache_file.write_text(insight_text, encoding="utf-8")
                logger.info(f"Saved insight for '{chart_type}' to cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to write insight to cache file {cache_file}: {e}")

        return insight_text


if __name__ == "__main__":
    # This basic test will likely fail without a valid API key and endpoint.
    # It primarily tests the structure and caching logic if API call is mocked/skipped.
    logger.info("--- Testing AIInsightsGenerator ---")

    # Dummy configs
    class DummyReportSettings:
        def __init__(self, path_str):
            self.output_dir = Path(path_str)

    class DummyProjectConfig:
        def __init__(self, output_dir_str):
            self.report_settings = DummyReportSettings(output_dir_str)
            # self.api_key_env_var = "YOUR_ENV_VAR_FOR_API_KEY" # Example

    test_output_dir = Path("./test_insights_output")
    test_output_dir.mkdir(exist_ok=True)

    insights_cfg = AIInsightsConfig(
        enabled=True, # Set to False to skip actual API calls in local test if no key
        model_name="gpt-3.5-turbo", # Or your preferred model
        api_endpoint="https://api.openai.com/v1/chat/completions", # Example
        cache_responses=True
    )
    project_cfg = DummyProjectConfig(str(test_output_dir))

    # Ensure OPENAI_API_KEY is set in your env for this test to attempt an API call
    # If not set, it will log a warning and API call will return None.
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. AI insight generation test will skip actual API call if enabled=True.")
        # insights_cfg.enabled = False # Optionally disable for test if key is missing

    ai_gen = AIInsightsGenerator(insights_config=insights_cfg, project_config=project_cfg)

    # Dummy chart data (e.g., summary statistics from a latency plot)
    dummy_latency_data_summary = {
        "mean_ms": 120.5, "median_ms": 115.0, "p95_ms": 180.0, "std_dev_ms": 25.3, "outlier_count": 5
    }

    insight = ai_gen.get_insight_for_chart(
        chart_type="Latency Distribution",
        chart_data=dummy_latency_data_summary,
        context_description="Evaluation of a real-time object detection model on edge device."
    )

    if insight:
        logger.info(f"Generated Insight:\n{insight}")
    else:
        logger.warning("Failed to generate insight (as expected if API key is missing or disabled).")

    # Test caching: Call again, should be faster and log cache hit if enabled and successful first time
    if insights_cfg.enabled and insights_cfg.cache_responses:
        logger.info("Attempting to retrieve insight from cache...")
        insight_cached = ai_gen.get_insight_for_chart(
            chart_type="Latency Distribution",
            chart_data=dummy_latency_data_summary, # Same data
            context_description="Evaluation of a real-time object detection model on edge device." # Same context
        )
        if insight_cached:
            logger.info(f"Cached Insight:\n{insight_cached}")
            if insight: # If first call was successful
                 assert insight_cached == insight
        else:
            logger.warning("Failed to retrieve insight from cache.")

    # Clean up test directory
    import shutil
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
    logger.info("AIInsightsGenerator test completed.")

```
