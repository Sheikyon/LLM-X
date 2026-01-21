from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich import box

from llm_x.estimation.memory import (
    get_quantization_estimates, 
    get_context_scaling_estimates, 
    get_rope_warning,
    calculate_engine_overhead
)
from llm_x.utils.hardware import get_gpu_info, get_ram_info

console = Console(color_system="truecolor", force_terminal=True)

LLM_X_LOGO = """
    __     __     __   ___      _  __
   / /    / /    /  |/  /     | |/ /
  / /    / /    / /|_/ /______|  / 
 / /___/ /___/ /  / /______/   |  
/_____/_____/_/  /_/      /_/|_|  
"""

def display_report(data: dict):
    if not data:
        console.print("[bold red]N/A[/bold red] - An error occurred during metadata retrieval.")
        return

    # Theme Colors
    LAVENDER = "#9370DB"
    SOFT_BLUE = "#7B68EE"
    GHOST_WHITE = "#F8F8FF"
    DIM_PURPLE = "#C5B4E3"

    # Hardware Detection 
    total_gpu_vram, free_gpu_vram = get_gpu_info()
    total_ram, available_ram = get_ram_info()

    # Base Metrics & Dynamic Overhead
    weights_gb = data["weights_gb"]
    kv_gb = data["kv_gb"]
    activations_gb = data.get("activations_gb", 0.0)
    context_len = data["context_len"]
    params_n = data.get('params_n', 0)
    params_bn = params_n / 1e9

    dynamic_overhead = calculate_engine_overhead(weights_gb, params_bn)
    # Main reference value
    total_vram_needed = weights_gb + kv_gb + activations_gb + dynamic_overhead

    # Logo
    logo = Text(LLM_X_LOGO, style=f"bold {LAVENDER}")
    console.print(Align.center(logo))
    
    # Model ID Panel
    model_id = data.get('id', 'N/A')
    console.print(Align.center(Panel(
        f"[bold {GHOST_WHITE}]{model_id}[/]",
        border_style=SOFT_BLUE,
        box=box.ROUNDED,
        style=f"on {SOFT_BLUE}",
        expand=False
    )))
    console.print("")

    # Grid for Specifications and Primary Resource Estimate
    grid = Table.grid(expand=True, padding=1)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    # Spec Panel (Left)
    id_content = Table(show_header=False, box=None, padding=(0, 1))
    id_content.add_row(f"[{LAVENDER}]Architecture:[/]", f"[bold]{data.get('architecture', 'Inferred')}[/]")
    id_content.add_row(f"[{LAVENDER}]Native DType:[/]", f"{str(data.get('dtype', 'N/A')).upper()}")
    id_content.add_row(f"[{LAVENDER}]Parameters:[/]", f"[bold white]{params_bn:.2f}B[/]")
    
    max_ctx = data.get('max_context', 'N/A')
    formatted_max_ctx = f"{max_ctx:,}" if isinstance(max_ctx, (int, float)) else max_ctx
    id_content.add_row(f"[{LAVENDER}]Max Context (config):[/]", formatted_max_ctx)
    id_content.add_row(f"[{LAVENDER}]Status:[/]", "[green]Header Scan OK[/]")
    
    left_panel = Panel(
        id_content, 
        title=f"[bold {GHOST_WHITE}]MODEL SPECIFICATIONS[/]", 
        border_style=LAVENDER, 
        box=box.HORIZONTALS,
        title_align="left"
    )

    # Resource Panel (Right)
    vram_table = Table(show_header=False, box=None, padding=(0, 1))
    vram_table.add_row("Weights Usage:", f"{weights_gb:,.2f} GiB")
    vram_table.add_row(f"KV Cache ({context_len:,}):", f"{kv_gb:,.2f} GiB")
    if data.get("include_activations") and activations_gb > 0:
        vram_table.add_row("Activations (prefill):", f"{activations_gb:,.2f} GiB")
    
    vram_table.add_row("Engine & Workspace:", f"{dynamic_overhead:,.2f} GiB [dim](Measured)[/]")
    vram_table.add_section()
    
    # Visual alert
    vram_style = f"bold white on {LAVENDER}"
    if total_gpu_vram > 0 and total_vram_needed > free_gpu_vram:
        vram_style = "bold white on red"

    vram_table.add_row(
        f"[bold {LAVENDER}]APPROX. MAX VRAM USE:[/]", 
        f"[{vram_style}] {total_vram_needed:,.2f} GiB [/{vram_style}]"
    )

    right_panel = Panel(
        vram_table, 
        title=f"[bold {GHOST_WHITE}]REQUIRED RESOURCES[/]", 
        border_style=SOFT_BLUE, 
        box=box.HORIZONTALS,
        title_align="left"
    )

    grid.add_row(left_panel, right_panel)
    console.print(grid)
    
    # --- HARDWARE CHECK SECTION ---
    hw_info = Table(show_header=False, box=None, padding=(0, 2), expand=True)

    if total_gpu_vram > 0:
        gpu_pct_free = (total_vram_needed / free_gpu_vram) * 100 if free_gpu_vram > 0 else float('inf')
        gpu_pct_total = (total_vram_needed / total_gpu_vram) * 100
        
        if gpu_pct_free <= 100:
            gpu_desc = f"[green]{free_gpu_vram:,.2f} GiB / {total_gpu_vram:,.2f} GiB ({gpu_pct_free:,.1f}% of free, {gpu_pct_total:,.1f}% of total)[/]"
        else:
            excedente = gpu_pct_free - 100
            gpu_desc = f"[bold red]{free_gpu_vram:,.2f} GiB / {total_gpu_vram:,.2f} GiB (+{excedente:,.1f}% larger than free, {gpu_pct_total:,.1f}% of total)[/]"
    else:
        gpu_desc = "[dim]No NVIDIA GPU detected[/]"

    ram_pct_free = (total_vram_needed / available_ram) * 100 if available_ram > 0 else float('inf')
    ram_pct_total = (total_vram_needed / total_ram) * 100

    if ram_pct_free <= 100:
        ram_desc = f"[green]{available_ram:,.2f} GiB / {total_ram:,.1f} GiB ({ram_pct_free:,.1f}% of free, {ram_pct_total:,.1f}% of total)[/]"
    else:
        excedente_ram = ram_pct_free - 100
        ram_desc = f"[bold red]{available_ram:,.2f} GiB / {total_ram:,.1f} GiB (+{excedente_ram:,.1f}% larger than free, {ram_pct_total:,.1f}% of total)[/]"

    hw_info.add_row("[bold white]Available GPU VRAM:[/]", gpu_desc)
    hw_info.add_row("[bold white]Available System RAM:[/]", ram_desc)

    console.print(Panel(hw_info, title=f"[bold {GHOST_WHITE}]HARDWARE CHECK[/]", border_style="dim"))

    rope_msg = get_rope_warning(data)
    if rope_msg:
        console.print(Align.center(rope_msg))
        console.print("")

    # --- UNIFIED VRAM OPERATIONAL MATRIX (Sincronizada) ---
    static_overhead = activations_gb + dynamic_overhead

    if params_n > 0:
        kv_per_token = kv_gb / context_len if context_len > 0 else 0
        
        ctx_levels = [8192, 32768, 65536, 131072]
        if context_len not in ctx_levels:
            ctx_levels.append(context_len)
        ctx_levels.sort()

        matrix_table = Table(show_header=True, box=box.SIMPLE, padding=(0, 2), expand=True)
        matrix_table.add_column("Context Window", style=DIM_PURPLE)
        matrix_table.add_column("KV Cache", justify="right", style="dim")
        matrix_table.add_column("Native (BF16)", justify="right")
        matrix_table.add_column("INT8/FP8", justify="right")
        matrix_table.add_column("4-bit (GPTQ/AWQ)", justify="right", style="bold white")

        for ctx in ctx_levels:
            is_current = (ctx == context_len)
            style = "bold white" if is_current else ""
            ctx_label = f"{ctx:,}" + (" [cyan](Current)[/]" if is_current else "")
            
            kv_step = kv_per_token * ctx
            
            # Native synchronization: We use actual weights_gb, not parameter estimation
            t_native = weights_gb + kv_step + static_overhead
            
            # Estimates for quantization (SafeTensors standard)
            t_int8 = (params_bn * 1.0) + kv_step + static_overhead
            t_q4 = (params_bn * 0.75) + kv_step + static_overhead

            def color_stat(val):
                if total_gpu_vram > 0 and val > free_gpu_vram:
                    return f"[red]{val:,.2f} GiB[/]"
                return f"[green]{val:,.2f} GiB[/]"

            matrix_table.add_row(
                ctx_label, 
                f"{kv_step:,.2f} GiB", 
                color_stat(t_native), 
                color_stat(t_int8), 
                color_stat(t_q4), 
                style=style
            )

        console.print(Panel(
            matrix_table, 
            title=f"[bold {GHOST_WHITE}]VRAM OPERATIONAL MATRIX[/]", 
            border_style=LAVENDER,
            subtitle="[dim]Values: Weights + KV Cache + Optim. Activations + Engine. [red]Red[/] exceeds free VRAM.[/dim]"
        ))

    console.print(Align.right(f"[dim]llm-x v0.1.0 • Made by Sheikyon[/dim]"))