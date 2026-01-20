import pytest
from llm_x.hub.hf_client import get_model_analysis


@pytest.mark.asyncio
async def test_analyze_qwen_live():
    """Test live analysis of a single-file dense model (Qwen2.5-7B)"""
    model_id = "Qwen/Qwen2.5-7B"
    
    # Get model metadata from Hugging Face
    analysis = await get_model_analysis(model_id)
    
    # Basic sanity checks
    assert analysis["model_id"] == model_id, "Model ID mismatch"
    assert analysis["total_params"] > 7_000_000_000, "Parameter count too low for 7B model"
    
    # Make sure we got the expected analysis fields
    assert "detected_dtype" in analysis, "Missing detected dtype"
    assert "architecture" in analysis, "Missing architecture field"


@pytest.mark.asyncio
async def test_analyze_mixtral_sharded_live():
    """
    Test live analysis of a heavily sharded MoE model (Mixtral-8x7B).
    Main goal: make sure the parser doesn't crash with 19+ shard files.
    """
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    
    # Get model metadata (this used to fail with many shards)
    analysis = await get_model_analysis(model_id)
    
    # If we reached this point → parsing didn't explode → good sign
    assert analysis["total_params"] > 40_000_000_000, "Parameter count too low for Mixtral 8x7B"
    
    # Very defensive architecture name check
    # Convert to string, clean it, lowercase → compare substrings
    arch = str(analysis["architecture"]).strip().lower()
    
    # Accept several reasonable variations people write / detect
    possible_names = ["mistral", "moe", "mixtral"]
    assert any(keyword in arch for keyword in possible_names), (
        f"Architecture name did not contain expected keywords. Got: {arch}"
    )