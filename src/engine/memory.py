"""
Hardware detection and memory management utilities.
"""

import os
import gc
import sys
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import torch


def clear_gpu_memory():
    """
    Clears Python garbage collection and CUDA cache.
    Safe to run on both CPU and CUDA devices.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_hardware_info() -> dict:
    """
    Automatically detects system CPU, RAM, GPU, and PyTorch environment.
    """
    cpu_cores = psutil.cpu_count(logical=False) if HAS_PSUTIL else os.cpu_count()
    logical_cpus = psutil.cpu_count(logical=True) if HAS_PSUTIL else os.cpu_count()
    total_ram = round(psutil.virtual_memory().total / (1024 ** 3), 2) if HAS_PSUTIL else "N/A"
    avail_ram = round(psutil.virtual_memory().available / (1024 ** 3), 2) if HAS_PSUTIL else "N/A"

    info = {
        "Platform": sys.platform,
        "Python Version": sys.version.split()[0],
        "PyTorch Version": torch.__version__,
        "CPU Cores": cpu_cores,
        "Logical CPUs": logical_cpus,
        "Total RAM (GB)": total_ram,
        "Available RAM (GB)": avail_ram,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "GPU Count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "GPU Name": "N/A",
        "Total VRAM (GB)": 0.0,
        "Allocated VRAM (GB)": 0.0,
        "Reserved VRAM (GB)": 0.0,
    }

    if torch.cuda.is_available():
        info["GPU Name"] = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["Total VRAM (GB)"] = round(vram_bytes / (1024 ** 3), 2)
        info["Allocated VRAM (GB)"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 3)
        info["Reserved VRAM (GB)"] = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 3)
    else:
        print("[WARNING] CUDA is NOT available! Execution will proceed on CPU, which is substantially slower.")

    return info


def print_hardware_summary():
    """Prints a clean summary of current execution hardware."""
    info = get_hardware_info()
    print("=" * 60)
    print("SYSTEM & HARDWARE SUMMARY")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k:22s}: {v}")
    print("=" * 60)
    return info
