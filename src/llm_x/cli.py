import typer
import asyncio
import os
from typing import Optional
from rich.console import Console
from rich.table import Table

from llm_x.pipeline import run_estimation_pipeline
from llm_x.hub.hf_client import (
    save_new_hf_token,
    get_token_list,
    get_active_index,
    set_active_token,
    delete_token,
    delete_all_tokens
)

app = typer.Typer(
    add_completion=False, 
    help="LLM-X: High-accuracy Python library for inference memory estimation with real hardware detection."
)
console = Console()

def handle_token_commands(new: str, set_t: int, list_t: bool, del_t: int, del_all: bool):
    if new:
        res = save_new_hf_token(new)
        console.print("[bold green]Token saved and sorted.[/]" if res != "exists" else "[yellow]Token already exists.[/]")
    elif list_t:
        tokens = get_token_list()
        active_idx = get_active_index()
        if not tokens:
            return console.print("[yellow]No tokens stored.[/]")
        
        table = Table(title="Hugging Face Tokens")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Preview")
        
        for i, t in enumerate(tokens, 1):
            is_active = "[bold green]ACTIVE[/]" if (i-1) == active_idx else ""
            table.add_row(str(i), is_active, f"{t[:10]}...")
        console.print(table)
    elif set_t:
        if set_active_token(set_t):
            console.print(f"[bold green]Token #{set_t} is now active.[/]")
        else:
            console.print(f"[bold red]Error:[/] Index {set_t} out of range.")
    elif del_t:
        if delete_token(del_t):
            console.print(f"[bold green]Token #{del_t} deleted.[/]")
        else:
            console.print(f"[bold red]Error:[/] Index {del_t} out of range.")
    elif del_all:
        delete_all_tokens()
        console.print("[bold red]All tokens deleted.[/]")

@app.command()
def main(
    model_src: Optional[str] = typer.Option(
        None, "--model-src", "-m", 
        help="Hugging Face ID, full URL, local path, or '.' for the current directory"
    ),
    
    context: Optional[int] = typer.Option(None, "--context", "-c"),
    batch: int = typer.Option(1, "--batch", "-b"),
    kv_dtype: str = typer.Option("BF16", "--kv-dtype", "-q"),
    kv_quant: str = typer.Option("none", "--kv-quant"),
    
    set_new_token: Optional[str] = typer.Option(None, "--set-new-token"),
    set_token: Optional[int] = typer.Option(None, "--set-token"),
    token_list: bool = typer.Option(False, "--token-list"),
    del_token: Optional[int] = typer.Option(None, "--del-token"),
    del_all_tokens: bool = typer.Option(False, "--del-all-tokens"),
):
    if any([set_new_token, token_list, set_token, del_token, del_all_tokens]):
        handle_token_commands(set_new_token, set_token, token_list, del_token, del_all_tokens)
        return

    if not model_src:
        console.print("[bold red]Error:[/] You must provide a model source (--model-src o -m).")
        raise typer.Exit(code=1)

    source_path = None
    model_id = None

    clean_input = model_src.replace("https://huggingface.co/", "").strip("/")

    if os.path.isdir(model_src) or model_src == ".":
        source_path = os.path.abspath(model_src)
    else:
        model_id = clean_input

    try:
        asyncio.run(run_estimation_pipeline(
            model_id=model_id,
            source=source_path,
            context=context,
            batch_size=batch,
            kv_dtype=kv_dtype,
            kv_quant=kv_quant
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation aborted.[/]")
        raise typer.Exit(0)

if __name__ == "__main__":
    app()