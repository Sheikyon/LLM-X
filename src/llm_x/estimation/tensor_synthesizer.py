import re
import math
from typing import Dict, Any, List, Tuple, Optional

def infer_architecture_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Black-box architecture analyzer.
    Uses GCD and linear equation solving to deduce head_dim and GQA splits
    in fused tensors.
    """
    
    # --- Aggressive regex mapping ---
    # We keep lists to collect all found evidence
    evidence = {
        "hidden_candidates": [],     # (value, source_confidence)
        "q_out": [], "k_out": [], "v_out": [],
        "qkv_fused_out": [],         # [dim]
        "mlp_gate": [],              # [dim] (up_proj / w1 / gate)
        "mlp_fused": [],             # [dim] (gate_up_proj)
        "norm_candidates": [],
        "vocab_size": None
    }
    
    layer_indices = set()
    expert_indices = set()

    # Compiled regex patterns for speed and precision
    # We prioritize explicit names over generic ones
    re_q = re.compile(r"(?:\.|_)(q_proj|query|w_q|wq)(?:\.|_|$)")
    re_k = re.compile(r"(?:\.|_)(k_proj|key|w_k|wk)(?:\.|_|$)")
    re_v = re.compile(r"(?:\.|_)(v_proj|value|w_v|wv)(?:\.|_|$)")
    re_qkv = re.compile(r"(?:\.|_)(query_key_value|qkv|wqkv)(?:\.|_|$)")
    
    # ChatGLM/Yi/Qwen sometimes use w1, w2... need to be careful
    re_mlp_gate = re.compile(r"(?:\.|_)(gate_proj|w1|up_proj|fc_in)(?:\.|_|$)") 
    re_mlp_fused = re.compile(r"(?:\.|_)(gate_up_proj)(?:\.|_|$)")
    
    re_norm = re.compile(r"(?:\.|_)(input_layernorm|ln_f|norm|final_layer_norm)(?:\.weight|$)")

    for key, info in metadata.items():
        if "shape" not in info: continue
        shape = info["shape"]
        if len(shape) == 0: continue
        
        # Noise filters
        k_low = key.lower()
        if any(x in k_low for x in ["bias", "mean", "var", "rope", "inv_freq", "rotary"]):
            continue
            
        # Layer detection
        # Looking for isolated digits or digits surrounded by separators
        digits = re.findall(r"(?:^|[\._])(\d+)(?:[\._]|$)", k_low)
        for d in digits:
            layer_indices.add(int(d))
            
        # Expert detection
        if "expert" in k_low:
             exp_digits = re.findall(r"expert(?:s|_)?[\._]?(\d+)", k_low)
             for d in exp_digits: expert_indices.add(int(d))

        # --- Dimension extraction ---
        
        # Embeddings & Norms (for hidden size)
        if len(shape) == 2 and any(x in k_low for x in ["embed_tokens", "wte", "word_embeddings"]):
            # shape [vocab, hidden]
            evidence["hidden_candidates"].append((shape[1], "high_embed"))
            evidence["vocab_size"] = shape[0]
            
        elif len(shape) == 1 and re_norm.search(k_low):
            evidence["hidden_candidates"].append((shape[0], "highest_norm"))
            evidence["norm_candidates"].append(shape[0])

        # This is critical to distinguish between separate Q and fused QKV
        if len(shape) == 2: # Linear weights [out, in]
            out_dim, in_dim = shape[0], shape[1]
            evidence["hidden_candidates"].append((in_dim, "medium_linear_in"))
            
            if re_qkv.search(k_low):
                evidence["qkv_fused_out"].append(out_dim)
            elif re_q.search(k_low):
                evidence["q_out"].append(out_dim)
            elif re_k.search(k_low):
                evidence["k_out"].append(out_dim)
            elif re_v.search(k_low):
                evidence["v_out"].append(out_dim)
            
            # 3. MLP
            elif re_mlp_fused.search(k_low):
                evidence["mlp_fused"].append(out_dim)
            elif re_mlp_gate.search(k_low):
                evidence["mlp_gate"].append(out_dim)

    # --- Hidden size determination ---
    # Consensus algorithm: most repeated or highest priority value (norm first)
    from collections import Counter
    if not evidence["hidden_candidates"]:
        return {"error": "CRITICAL: No hidden_size candidates found."}
    
    # Absolute priority to norm if it exists (embeddings can lie due to tying,
    # linear inputs can be misleading)
    norm_counts = Counter(evidence["norm_candidates"])
    if norm_counts:
        hidden_size = norm_counts.most_common(1)[0][0]
    else:
        # Fallback: majority vote on linear input dims
        all_vals = [x[0] for x in evidence["hidden_candidates"]]
        hidden_size = Counter(all_vals).most_common(1)[0][0]

    # --- Attention deduction (the mathematical core) ---
    
    head_dim = None
    num_heads = None
    num_kv_heads = None
    attn_mode = "Unknown"

    # Separate Q and K tensors (Llama, Mistral, Gemma, Qwen-Sep)
    if evidence["q_out"]:
        q_dim = evidence["q_out"][0]
        # If K exists, use GCD to find common head block
        if evidence["k_out"]:
            k_dim = evidence["k_out"][0]
            # GCD(Q, K) gives maximum common block.
            # Example: Q=4096, K=1024 (GQA) → GCD=1024
            # head_dim must be a divisor of 1024 (usually ≤ 256)
            common_block = math.gcd(q_dim, k_dim)
            
            # Look for largest reasonable divisor (≤ 512)
            candidates = [x for x in [256, 128, 64, 96, 192, 160, 112, 80] if common_block % x == 0]
            
            if candidates:
                head_dim = max(candidates)  # greedy: prefer largest
            else:
                head_dim = 128  # safe fallback
            
            num_heads = q_dim // head_dim
            num_kv_heads = k_dim // head_dim
            attn_mode = "Separate (GQA/MQA)" if num_heads != num_kv_heads else "Separate (MHA)"
            
        else:
            # Only Q found (rare — maybe implicit MHA without K detected)
            head_dim = 128
            num_heads = q_dim // 128
            num_kv_heads = num_heads

    # Fused QKV (Falcon, Bloom, Starcoder, many Chinese models)
    elif evidence["qkv_fused_out"]:
        fused_dim = evidence["qkv_fused_out"][0]
        attn_mode = "Fused QKV"
        
        # Equation: Fused = (N_q + N_k + N_v) * D
        # Assuming N_k == N_v (standard), Fused = (N_q + 2*N_k) * D
        
        # Try to find plausible D (head_dim)
        possible_dims = [128, 64, 256, 96, 80, 48, 160, 192, 112]
        valid_configs = []

        for d in possible_dims:
            if fused_dim % d == 0:
                total_heads = fused_dim // d
                # Now decompose: total_heads = N_q + 2*N_k
                # Strong constraint: N_q must be divisible by N_k (GQA)
                
                # Hypothesis 1: hidden_size exactly matches query projection
                if hidden_size % d == 0:
                    n_q_est = hidden_size // d
                    remaining_heads = total_heads - n_q_est
                    if remaining_heads > 0 and remaining_heads % 2 == 0:
                        n_k_est = remaining_heads // 2
                        # Check GQA ratio
                        if n_k_est > 0 and n_q_est % n_k_est == 0:
                            valid_configs.append((d, n_q_est, n_k_est))
        
        # Select best configuration
        if valid_configs:
            # Prefer standard head dims (128 > 64 > larger)
            valid_configs.sort(key=lambda x: (x[0] != 128, x[0] != 64, -x[0]))
            best = valid_configs[0]
            head_dim, num_heads, num_kv_heads = best
        else:
            # Desperate fallback: assume classic MHA (divide by 3)
            if fused_dim % 3 == 0:
                part = fused_dim // 3
                for d in [128, 64, 80, 96]:
                    if part % d == 0:
                        head_dim = d
                        num_heads = part // d
                        num_kv_heads = num_heads
                        break
    
    # --- Intermediate size estimation ---
    intermediate_size = 0
    if evidence["mlp_fused"]:
        # Gate + Up fused → total dim / 2
        intermediate_size = evidence["mlp_fused"][0] // 2
    elif evidence["mlp_gate"]:
        intermediate_size = evidence["mlp_gate"][0]
    
    # Final report
    
    return {
        "arch_type": attn_mode,
        "hidden_size": hidden_size,
        "num_layers": max(layer_indices) + 1 if layer_indices else 0,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "vocab_size": evidence["vocab_size"],
        "num_experts": max(expert_indices) + 1 if expert_indices else 0,
        # GQA ratio (for debugging)
        "gqa_ratio": (num_heads // num_kv_heads) if num_kv_heads else 0
    }