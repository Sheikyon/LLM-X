import httpx
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from huggingface_hub import model_info
from .safetensors_metadata import fetch_safetensors_header, parse_safetensors_metadata, _extract_context_info

TOKEN_PATH = Path.home() / ".llm_x_tokens.json"

def _load_token_data() -> Dict[str, Any]:
    """Helper to load the JSON token structure."""
    if TOKEN_PATH.exists():
        try:
            return json.loads(TOKEN_PATH.read_text())
        except Exception:
            pass
    return {"active_index": None, "tokens": []}

def _save_token_data(data: Dict[str, Any]) -> None:
    """Helper to save the JSON token structure."""
    TOKEN_PATH.write_text(json.dumps(data, indent=4))

def save_new_hf_token(token: str) -> str:
    """Adds a new token to the list if it doesn't exist."""
    data = _load_token_data()
    token = token.strip()
    
    if token in data["tokens"]:
        return "exists"
    
    data["tokens"].append(token)
    data["tokens"].sort()
    
    if data["active_index"] is None:
        data["active_index"] = data["tokens"].index(token)
        
    _save_token_data(data)
    return "saved"

def get_token_list() -> List[str]:
    """Returns the sorted list of tokens."""
    return _load_token_data()["tokens"]

def get_active_index() -> Optional[int]:
    """Returns the current active index."""
    return _load_token_data()["active_index"]

def set_active_token(index: int) -> bool:
    """Sets the active token by index (1-based for users)."""
    data = _load_token_data()
    actual_idx = index - 1
    if 0 <= actual_idx < len(data["tokens"]):
        data["active_index"] = actual_idx
        _save_token_data(data)
        return True
    return False

def delete_token(index: int) -> bool:
    """Deletes a token by index (1-based for users)."""
    data = _load_token_data()
    actual_idx = index - 1
    if 0 <= actual_idx < len(data["tokens"]):
        data["tokens"].pop(actual_idx)
        if not data["tokens"]:
            data["active_index"] = None
        else:
            data["active_index"] = 0
        _save_token_data(data)
        return True
    return False

def delete_all_tokens() -> None:
    """Clears the token file."""
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()

def load_hf_token() -> Optional[str]:
    """Loads the currently active token."""
    data = _load_token_data()
    idx = data["active_index"]
    if idx is not None and 0 <= idx < len(data["tokens"]):
        return data["tokens"][idx]
    return None

def get_auth_headers() -> Dict[str, str]:
    """Generates authentication headers using the active token."""
    token = load_hf_token()
    return {"Authorization": f"Bearer {token}"} if token else {}

def format_hf_error(e: httpx.HTTPStatusError, model_id: str) -> str:
    """Formats HTTP errors into user-friendly instructions."""
    code = e.response.status_code
    if code in [401, 403]:
        return (
            f"\n[bold red]Access Denied ({code}):[/bold red]\n"
            f"The repository [bold]'{model_id}'[/bold] is private or requires a license agreement.\n"
            "To add a token: [green]llm-x --set-new-token \"hf_...\"[/green]\n"
            "To select a token: [green]llm-x --set-token [number][/green]\n"
        )
    elif code == 404:
        return f"\n[bold red]Model Not Found (404):[/bold red] '{model_id}' on Hugging Face."
    return f"\n[bold red]HTTP Error {code}:[/bold red] {e}"

async def get_model_analysis(model_id: str, revision: str = "main") -> Dict[str, Any]:
    """
    Optimized model analysis using metadata fetching and architecture inference.
    """
    headers = get_auth_headers()
    async with httpx.AsyncClient(http2=True, follow_redirects=True, headers=headers) as client:
        # Fetch Config
        base_url = f"https://huggingface.co/{model_id}/resolve/{revision}"
        config_resp = await client.get(f"{base_url}/config.json")
        config_resp.raise_for_status()
        config = config_resp.json()
        
        ctx_info = _extract_context_info(config)
        
        # Try to get safetensors metadata
        shards_metadata = []
        try:
            # Check for index first
            index_url = f"{base_url}/model.safetensors.index.json"
            index_resp = await client.get(index_url)
            if index_resp.status_code == 200:
                index_data = index_resp.json()
                weight_map = index_data.get("weight_map", {})
                unique_files = list(set(weight_map.values()))
                for shard in unique_files:
                    header = await fetch_safetensors_header(client, f"{base_url}/{shard}", headers)
                    shards_metadata.append(header)
            else:
                # Try single file
                header = await fetch_safetensors_header(client, f"{base_url}/model.safetensors", headers)
                shards_metadata.append(header)
        except Exception:
            pass

        if not shards_metadata:
            raise RuntimeError(f"No valid SafeTensors metadata found for {model_id}.")

        analysis = parse_safetensors_metadata(shards_metadata)
        
        return {
            "model_id": model_id,
            "config": config,
            "full_metadata": shards_metadata[0],
            "architecture": ctx_info["arch"],
            **analysis,
            **ctx_info["ctx"]
        }