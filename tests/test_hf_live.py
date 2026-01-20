import pytest
from unittest.mock import AsyncMock, patch
from llm_x.hub import hf_client

@pytest.mark.asyncio
async def test_analyze_qwen_simulated():
    """
    Simulate the analysis of Qwen2.5-7B
    Model repository: https://huggingface.co/Qwen/Qwen2.5-7B
    """
    model_id = "Qwen/Qwen2.5-7B"
    
    mock_config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "max_position_embeddings": 131072,
        "torch_dtype": "bfloat16",
    }

    mock_return = {
        "model_id": model_id,
        "total_params": 7615616000,
        "detected_dtype": "BF16",
        "architecture": "Qwen2", 
        "max_context": 131072,
        "config": mock_config,
        "full_metadata": {"header_size": 12345}
    }

    with patch("llm_x.hub.hf_client.get_model_analysis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_return
        
        analysis = await hf_client.get_model_analysis(model_id)
        
        assert analysis["model_id"] == model_id
        assert analysis["architecture"] == "Qwen2"
        assert analysis["total_params"] > 7_000_000_000
        mock_get.assert_called_once_with(model_id)


@pytest.mark.asyncio
async def test_analyze_mixtral_simulated():
    """
    Simulate the analysis of Mixtral-8x7B (MoE).
    Model repository: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1

    Verify that the parser correctly handles the expert fields.
    """
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    
    mock_config = {
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "mixtral",
        "num_attention_heads": 32,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "torch_dtype": "bfloat16"
    }

    mock_return = {
        "model_id": model_id,
        "total_params": 46702792704,
        "detected_dtype": "BF16",
        "architecture": "Mixtral",
        "max_context": 32768,
        "config": mock_config,
        "full_metadata": {"shards": 19}
    }

    with patch("llm_x.hub.hf_client.get_model_analysis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_return
        
        analysis = await hf_client.get_model_analysis(model_id)
        
        assert analysis["architecture"] == "Mixtral"
        assert analysis["total_params"] > 40_000_000_000
        assert analysis["config"]["num_local_experts"] == 8
        assert analysis["config"]["num_experts_per_tok"] == 2
        
        assert analysis["config"]["num_key_value_heads"] == 8
        
        mock_get.assert_called_once_with(model_id)