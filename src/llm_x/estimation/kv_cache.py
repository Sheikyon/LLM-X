from typing import Dict, Any, Optional, Tuple
from llm_x.utils.types import get_bytes_per_element
from rich.console import Console
from .tensor_synthesizer import infer_architecture_from_metadata

console = Console()

def estimate_kv_cache(
    metadata: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    seq_len: int = 8192,
    batch_size: int = 1,
    kv_dtype: str = "BF16",
    kv_quant: str = "none",
) -> Tuple[float, Dict[str, Any], float]:
    """
    Calculates VRAM usage for KV Cache and optimized activations (2026 standard).
    """
    
    # Parameter synthesis / architecture inference
    inferred = infer_architecture_from_metadata(metadata)
    
    if "error" in inferred:
        raise ValueError(f"Failed to infer architecture: {inferred['error']}")

    params = {
        "hidden_size": (config or {}).get("hidden_size") or (config or {}).get("n_embd") or inferred["hidden_size"],
        "num_layers": (config or {}).get("num_hidden_layers") or (config or {}).get("n_layer") or inferred["num_layers"],
        "num_heads": (config or {}).get("num_attention_heads") or (config or {}).get("n_head") or inferred["num_heads"],
        "num_kv_heads": (config or {}).get("num_key_value_heads") or (config or {}).get("num_kv_head") or inferred["num_kv_heads"],
        "head_dim": (config or {}).get("head_dim") or inferred["head_dim"],
        "intermediate_size": (config or {}).get("intermediate_size") or inferred["intermediate_size"] or (inferred["hidden_size"] * 4),
        "num_experts": (config or {}).get("num_local_experts") or inferred.get("num_experts", 0)
    }

    required = ["hidden_size", "num_layers", "num_heads", "num_kv_heads", "head_dim"]
    if missing := [k for k in required if not params.get(k)]:
        raise ValueError(f"Missing critical parameters: {', '.join(missing)}")

    # KV Cache calculation (architecture dependent)
    is_mla = False
    kv_lora_rank = (config or {}).get("kv_lora_rank", 0)
    
    if kv_lora_rank > 0:
        is_mla = True
        qk_rope_dim = (config or {}).get("qk_rope_head_dim", 0)
        kv_dim_per_layer = kv_lora_rank + qk_rope_dim
    else:
        kv_dim_per_layer = 2 * params["num_kv_heads"] * params["head_dim"]

    # Precision / dtype size
    dtype_bytes = get_bytes_per_element(kv_dtype.upper())
    if kv_quant.lower() in ['int8', 'fp8', 'e4m3', 'e5m2']:
        dtype_bytes = 1

    # KV Cache VRAM usage
    total_kv_bytes = batch_size * seq_len * params["num_layers"] * kv_dim_per_layer * dtype_bytes
    kv_gb = total_kv_bytes / (1024 ** 3)
    
    # Modern architectures with fused kernels don't keep intermediate MLP states.
    # Main cost comes from residual stream + kernel output buffers.
    
    # Optimized activation factor (residual + fusion overhead)
    # Drastically reduced from ~2.0× (hidden+inter) to ~1.4× (hidden)
    fused_factor = 1.4 
    
    activations_bytes = batch_size * seq_len * params["hidden_size"] * dtype_bytes * fused_factor
    
    # For very long contexts we assume Activation Recomputation (Gradient Checkpointing)
    # This is now standard to handle 128k+ contexts without exploding VRAM
    if seq_len > 8192:
        # Recomputation reduces persistent activation memory to a fraction (usually logarithmic or per-block)
        # We approximate ~50% reduction in stored activations
        activations_bytes *= 0.5
        console.print(f"[dim yellow]Long context detected ({seq_len}):[/] Applying recomputation factor.")

    activations_gb = activations_bytes / (1024 ** 3)

    # Metadata / breakdown
    breakdown = {
        "method": inferred.get("arch_style", "Standard"),
        "is_mla": is_mla,
        "is_moe": params["num_experts"] > 1,
        "dtype_bytes": dtype_bytes,
        "optimized_activations": True
    }
    
    return kv_gb, breakdown, activations_gb