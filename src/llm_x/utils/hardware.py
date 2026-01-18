import shutil
import psutil
try:
    from pynvml import *
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

def get_gpu_info():
    """Returns (total_vram_gb, free_vram_gb) for the first GPU found."""
    if not NVML_AVAILABLE:
        return 0.0, 0.0
    
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.total / (1024**3), info.free / (1024**3)
    except:
        return 0.0, 0.0

def get_ram_info():
    """Returns (total_ram_gb, available_ram_gb)."""
    mem = psutil.virtual_memory()
    return mem.total / (1024**3), mem.available / (1024**3)