import httpx
import os
from rich.console import Console

from llm_x.hub.hf_client import get_auth_headers, format_hf_error
from llm_x.hub.safetensors_metadata import analyze_hub_model, analyze_local_model
from llm_x.estimation.kv_cache import estimate_kv_cache
from llm_x.estimation.memory import (
    prepare_vram_report_data,
    get_rope_warning,
    resolve_context_length
)
from llm_x.print import display_report

console = Console()

async def run_estimation_pipeline(
    model_id: str = None,
    source: str = None,
    context: int = None,
    batch_size: int = 1,
    kv_dtype: str = "BF16",
    kv_quant: str = "none"
):
    # Clear screen for a clean CLI experience
    os.system('cls' if os.name == 'nt' else 'clear')
    
    try:
        # Metadata Acquisition
        if source:
            console.print(f"[dim blue]Local Mode[/] Analyzing path: [bold #9370DB]{source}[/]")
            model_info = await analyze_local_model(source)
        else:
            headers = get_auth_headers()
            auth_status = "[dim green]Authenticated[/]" if "Authorization" in headers else "[dim yellow]Public only[/]"
            console.print(f"[{auth_status}] Analyzing Hub model: [bold #9370DB]{model_id}[/]")
            
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0, headers=headers) as client:
                model_info = await analyze_hub_model(client, model_id)

        # Determines final seq_len based on user input vs model config/RoPE limits
        final_context = resolve_context_length(context, model_info)
        
        # The estimate_kv_cache function uses the metadata for shape inference 
        # and config for architectural specifics (MLA/MoE/GQA)
        kv_gb, breakdown, activations_gb = estimate_kv_cache(
            metadata=model_info["full_metadata"],
            config=model_info["config"],
            seq_len=final_context,
            batch_size=batch_size,
            kv_dtype=kv_dtype,
            kv_quant=kv_quant
        )

        # Consolidate all data for the final display
        report_data = prepare_vram_report_data(
            model_info=model_info,
            kv_gb=kv_gb,
            activations_gb=activations_gb,
            breakdown=breakdown,
            final_context=final_context,
            batch_size=batch_size,
            kv_dtype=kv_dtype,
            kv_quant=kv_quant,
            include_activations=True
        )

        display_report(report_data)

    except httpx.HTTPStatusError as e:
        console.print(format_hf_error(e, model_id))
    except Exception as e:
        # Catch-all for unexpected pipeline failures (permission errors, missing keys, etc.)
        console.print(f"[bold red]PIPELINE ERROR:[/] {e}")