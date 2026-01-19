import json
import struct
import httpx
import os
import glob
from collections import Counter
from typing import Dict, Any, Optional
from llm_x.utils.types import get_bytes_per_element
from llm_x.estimation.tensor_synthesizer import infer_architecture_from_metadata

async def fetch_safetensors_header(
    client: httpx.AsyncClient,
    url: str,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Downloads only the JSON header from a remote .safetensors file."""
    try:
        range_header = {"Range": "bytes=0-7"}
        response = await client.get(url, headers={**range_header, **(headers or {})})
        response.raise_for_status()
        
        metadata_size = struct.unpack("<Q", response.content)[0]
        
        range_json = {"Range": f"bytes=8-{8 + metadata_size - 1}"}
        response = await client.get(url, headers={**range_json, **(headers or {})})
        response.raise_for_status()
        
        return json.loads(response.content)
    except Exception as e:
        raise ValueError(f"Error reading safetensors header from {url}: {e}")

def read_local_safetensors_header(file_path: str) -> Dict[str, Any]:
    """Reads only the JSON header from a local file."""
    with open(file_path, "rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) != 8:
             return {}
        
        metadata_size = struct.unpack("<Q", header_size_bytes)[0]
        metadata_bytes = f.read(metadata_size)
        return json.loads(metadata_bytes)


def parse_safetensors_metadata(shards_metadata: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Total Params, Total Disk Bytes, and Majority DType across shards."""
    total_params = 0
    total_bytes = 0
    dtype_counter = Counter()

    for metadata in shards_metadata:
        for key, info in metadata.items():
            if key == "__metadata__": continue
                
            if isinstance(info, dict) and "shape" in info and "dtype" in info:
                shape = info["shape"]
                dtype = info["dtype"]
                
                params = 1
                for dim in shape: params *= dim
                
                total_params += params
                bpe = get_bytes_per_element(dtype)
                total_bytes += params * bpe
                
                if len(shape) > 1: 
                    dtype_counter[dtype] += params

    detected_dtype = dtype_counter.most_common(1)[0][0] if dtype_counter else "BF16"

    return {
        "total_params": total_params,
        "total_bytes": total_bytes,
        "detected_dtype": detected_dtype
    }


async def analyze_local_model(path: str) -> Dict[str, Any]:
    base_path = os.path.abspath(path)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Path not found: {base_path}")

    with open(os.path.join(base_path, "config.json"), "r") as f:
        config = json.load(f)

    shards_metadata = []
    index_path = os.path.join(base_path, "model.safetensors.index.json")
    
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
            files_to_read = set(index.get("weight_map", {}).values())
            for fname in files_to_read:
                shards_metadata.append(read_local_safetensors_header(os.path.join(base_path, fname)))
    else:
        files = glob.glob(os.path.join(base_path, "*.safetensors"))
        for f_path in files:
            shards_metadata.append(read_local_safetensors_header(f_path))

    if not shards_metadata:
        raise ValueError("No safetensors found.")

    analysis = parse_safetensors_metadata(shards_metadata)
    # We use the first shard as a representative sample of the architecture
    arch_info = infer_architecture_from_metadata(shards_metadata[0])
    ctx_info = _extract_context_info(config)

    return {
        "model_id": f"Local: {os.path.basename(base_path)}",
        "config": config,
        "full_metadata": shards_metadata[0],
        "inferred_arch": arch_info,
        "architecture": ctx_info["arch"],
        **analysis,
        **ctx_info["ctx"]
    }

async def analyze_hub_model(client: httpx.AsyncClient, model_id: str) -> Dict[str, Any]:
    base_url = f"https://huggingface.co/{model_id}/resolve/main"
    resp = await client.get(f"{base_url}/config.json")
    resp.raise_for_status()
    config = resp.json()
    
    shards_metadata = []
    try:
        # Try to find the index first
        resp = await client.get(f"{base_url}/model.safetensors.index.json")
        if resp.status_code == 200:
            index = resp.json()
            weight_map = index.get("weight_map", {})
            # We only need one header for architecture inference, 
            # but we need all of them to count total parameters correctly.
            files_to_read = sorted(list(set(weight_map.values())))
            for fname in files_to_read:
                meta = await fetch_safetensors_header(client, f"{base_url}/{fname}")
                shards_metadata.append(meta)
    except Exception:
        pass
        
    if not shards_metadata:
        # Fallback to single file
        meta = await fetch_safetensors_header(client, f"{base_url}/model.safetensors")
        shards_metadata.append(meta)

    analysis = parse_safetensors_metadata(shards_metadata)
    arch_info = infer_architecture_from_metadata(shards_metadata[0])
    ctx_info = _extract_context_info(config)
    
    return {
        "model_id": model_id,
        "config": config,
        "full_metadata": shards_metadata[0],
        "inferred_arch": arch_info,
        "architecture": ctx_info["arch"],
        **analysis,
        **ctx_info["ctx"]
    }

def _extract_context_info(config: Dict[str, Any]) -> Dict[str, Any]:
    arch = "Unknown"
    if config.get("architectures"):
        arch = config["architectures"][0].replace("ForCausalLM", "").replace("Model", "")
    
    max_ctx = config.get("max_position_embeddings") or config.get("n_ctx") or 2048
    rope_info = config.get("rope_scaling")
    orig_ctx = None
    if isinstance(rope_info, dict):
        orig_ctx = rope_info.get("original_max_position_embeddings")

    return {
        "arch": arch,
        "ctx": {
            "max_context": max_ctx,
            "rope_info": rope_info,
            "original_max_context": orig_ctx
        }
    }