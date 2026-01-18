from typing import Dict, Any, Optional, List

def resolve_context_length(user_context: Optional[int], model_info: Dict[str, Any]) -> int:
    """
    Determines the final context length for estimation.
    Prioritizes user override, then model config value, defaulting to 8192.
    """
    return user_context or model_info.get("max_context") or 8192

def get_rope_warning(model_info: Dict[str, Any]) -> Optional[str]:
    """
    Checks for RoPE scaling in model configuration and returns a formatted 
    warning string if the model supports context extension.
    """
    rope = model_info.get("rope_info")
    if not rope:
        return None
    
    rope_type = rope.get("type", "unknown")
    factor = rope.get("factor", "?")
    original = model_info.get("original_max_context", "?")
    
    return (f"[yellow]NOTE: Model with RoPE scaling ({rope_type}). "
            f"config.json declares {model_info['max_context']} tokens, "
            f"supports extension (original ~{original}, factor {factor}).[/yellow]")

def get_quantization_estimates(params_count: int, static_vram_gb: float) -> List[Dict[str, Any]]:
    """
    Calculates estimated VRAM usage for various quantization bit-depths.
    'static_vram_gb' should include KV Cache and Activations to provide 
    a realistic total runtime memory footprint.
    """
    # Standard bit-rates for popular quantization methods (GPTQ, AWQ, EXL2, GGUF)
    quant_levels = [
        ("8-bit (FP8 / INT8)", 8.0),
        ("6-bit (EXL2 / HQQ)", 6.0),
        ("5-bit (EXL2 / HQQ)", 5.0),
        ("4-bit (GPTQ / AWQ)", 4.5),
        ("3-bit (GPTQ / EXL2)", 3.5),
        ("2-bit (EXL2 / HQQ)", 2.5),
    ]
    
    results = []
    for q_name, bits in quant_levels:
        # Theoretical model size: (Parameters * bits_per_param) / bits_in_byte
        model_size_gb = (params_count * bits / 8) / (1024**3)
        results.append({
            "name": q_name,
            "model_size_gb": model_size_gb,
            "total_vram_gb": model_size_gb + static_vram_gb
        })
    return results

def get_context_scaling_estimates(
    params_count: int, 
    kv_gb_per_token: float, 
    native_weights_gb: float, 
    static_overhead_gb: float
) -> List[Dict[str, Any]]:
    """
    Projects how VRAM scales across common context tiers.
    'static_overhead_gb' represents fixed costs like activations and CUDA kernels.
    """
    context_tiers = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
    
    # 4.5 bits is a safe average for 4-bit quantizations (including metadata/scales)
    q4_bits = 4.5
    q4_weights_gb = (params_count * q4_bits / 8) / (1024**3)
    
    results = []
    for ctx in context_tiers:
        temp_kv = kv_gb_per_token * ctx
        results.append({
            "context": ctx,
            "kv_cache_gb": temp_kv,
            "q4_vram_gb": q4_weights_gb + temp_kv + static_overhead_gb,
            "native_vram_gb": native_weights_gb + temp_kv + static_overhead_gb
        })
    return results

def prepare_vram_report_data(
    model_info: Dict[str, Any],
    kv_gb: float,
    activations_gb: float,
    breakdown: Dict[str, Any],
    final_context: int,
    batch_size: int,
    kv_dtype: str,
    kv_quant: str,
    include_activations: bool = True
) -> Dict[str, Any]:
    """
    Consolidates all calculated metrics and metadata into a single dictionary
    ready for the UI rendering layer (display_report).
    """
    # Raw weight size in Gigabytes based on the detected safetensors/config
    weights_gb = model_info["total_bytes"] / (1024 ** 3)

    return {
        "id": model_info["model_id"],
        "architecture": model_info["architecture"],
        "dtype": model_info["detected_dtype"].upper(),
        "params_n": model_info["total_params"],
        "bytes_n": model_info["total_bytes"],
        "weights_gb": weights_gb,
        "kv_gb": kv_gb,
        "activations_gb": activations_gb,
        "context_len": final_context,
        "max_context": model_info["max_context"],
        "original_max_context": model_info.get("original_max_context"),
        "rope_info": model_info.get("rope_info"),
        "full_metadata": model_info["full_metadata"],
        "config": model_info["config"],
        "batch_size": batch_size,
        "kv_dtype": kv_dtype.upper(),
        "kv_quant": kv_quant.upper(),
        "include_activations": include_activations,
        "breakdown": breakdown,
    }

def calculate_engine_overhead(weights_gb: float, params_bn: float) -> float:
    # Here it is assumed that the CUDA/ROCm context usually occupies between 0.6 and 0.9 GiB
    base_context = 0.8 

    # Calculation buffers (gemm) scale with the size of the layers
    # Therefore, 5% of the weight size is a very reliable engineering metric
    workspace_buffer = weights_gb * 0.05
    
    return base_context + workspace_buffer