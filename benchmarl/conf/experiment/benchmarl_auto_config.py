#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
from pprint import pprint

import psutil
import torch


def round_power2(x):
    return 2 ** round(math.log2(x))


def safe_int(x):
    try:
        if x in ["[N/A]", "N/A", "", None]:
            return None
        return int(float(str(x).strip()))
    except Exception:
        return None


def safe_float(x):
    try:
        if x in ["[N/A]", "N/A", "", None]:
            return None
        return float(str(x).strip())
    except Exception:
        return None


def get_system_info():
    ram = psutil.virtual_memory()

    info = {
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_freq_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        },
        "ram": {
            "total_gb": round(ram.total / 1024**3, 2),
            "available_gb": round(ram.available / 1024**3, 2),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": [],
        },
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["cuda"]["gpus"].append({
                "id": i,
                "name": p.name,
                "vram_gb": round(p.total_memory / 1024**3, 2),
                "sm_count": p.multi_processor_count,
            })

    return info


def get_nvidia_smi():

    # --------------------------------------------------------
    # NO CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        return {
            "available": False,
            "reason": "CUDA not available"
        }

    # --------------------------------------------------------
    # NVIDIA-SMI NOT INSTALLED
    # --------------------------------------------------------

    if subprocess.call(
        ["which", "nvidia-smi"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ) != 0:

        return {
            "available": False,
            "reason": "nvidia-smi not installed"
        }

    # --------------------------------------------------------
    # QUERY NVIDIA-SMI
    # --------------------------------------------------------

    try:

        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]).decode().strip()

        rows = []

        for line in out.splitlines():

            v = [x.strip() for x in line.split(",")]

            rows.append({

                "gpu_util_percent": safe_int(v[0]),
                "mem_util_percent": safe_int(v[1]),
                "memory_total_mb": safe_int(v[2]),
                "memory_used_mb": safe_int(v[3]),
                "power_w": safe_float(v[4]),
                "temp_c": safe_int(v[5]),

            })

        return {
            "available": True,
            "gpus": rows
        }

    except Exception as e:

        return {
            "available": False,
            "reason": str(e)
        }


def recommend_vmas_gpu(info, aggressiveness="max"):
    """
    VMAS is vectorized / GPU-friendly.
    Here we scale mainly with VRAM and GPU availability, not only CPU cores.
    """

    cpu_cores = info["cpu"]["physical_cores"]
    ram_gb = info["ram"]["total_gb"]
    has_cuda = info["cuda"]["available"]
    vram_gb = info["cuda"]["gpus"][0]["vram_gb"] if has_cuda else 0

    if not has_cuda:
        base_envs = max(8, cpu_cores)
    else:
        if vram_gb >= 96:
            base_envs = 256
        elif vram_gb >= 48:
            base_envs = 128
        elif vram_gb >= 24:
            base_envs = 64
        else:
            base_envs = 32

    if aggressiveness == "safe":
        n_envs = max(16, base_envs // 4)
    elif aggressiveness == "balanced":
        n_envs = max(32, base_envs // 2)
    else:
        n_envs = base_envs

    frames_per_batch = n_envs * 512

    if aggressiveness == "max":
        frames_per_batch = n_envs * 1024

    frames_per_batch = round_power2(frames_per_batch)

    minibatch_size = max(1024, frames_per_batch // 8)
    minibatch_size = round_power2(minibatch_size)

    if vram_gb >= 96 and aggressiveness == "max":
        minibatch_size = max(minibatch_size, 8192)

    evaluation_interval = frames_per_batch * 10
    checkpoint_interval = frames_per_batch * 10

    cfg = {
        "sampling_device": "cuda" if has_cuda else "cpu",
        "train_device": "cuda" if has_cuda else "cpu",
        "buffer_device": "cpu",

        "parallel_collection": False,

        "on_policy_n_envs_per_worker": n_envs,
        "on_policy_collected_frames_per_batch": frames_per_batch,
        "on_policy_minibatch_size": minibatch_size,
        "on_policy_n_minibatch_iters": 30 if aggressiveness == "max" else 20,

        "off_policy_n_envs_per_worker": n_envs,
        "off_policy_collected_frames_per_batch": frames_per_batch,
        "off_policy_train_batch_size": 8192 if ram_gb >= 96 else 4096,
        "off_policy_memory_size": 10_000_000 if ram_gb >= 96 else 5_000_000,

        "evaluation_interval": evaluation_interval,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_at_end": True,
    }

    return cfg


def recommend_cpu_bound(info, aggressiveness="balanced"):
    cpu_cores = info["cpu"]["physical_cores"]
    ram_gb = info["ram"]["total_gb"]
    has_cuda = info["cuda"]["available"]

    reserved = 2
    workers_capacity = max(1, cpu_cores - reserved)

    if aggressiveness == "safe":
        n_envs = max(4, workers_capacity // 2)
    elif aggressiveness == "balanced":
        n_envs = workers_capacity
    else:
        n_envs = workers_capacity * 2

    frames_per_batch = round_power2(max(4096, n_envs * 512))
    minibatch_size = round_power2(max(512, frames_per_batch // 8))

    return {
        "sampling_device": "cpu",
        "train_device": "cuda" if has_cuda else "cpu",
        "buffer_device": "cpu",

        "parallel_collection": True,

        "on_policy_n_envs_per_worker": n_envs,
        "on_policy_collected_frames_per_batch": frames_per_batch,
        "on_policy_minibatch_size": minibatch_size,
        "on_policy_n_minibatch_iters": 20,

        "off_policy_n_envs_per_worker": n_envs,
        "off_policy_collected_frames_per_batch": frames_per_batch,
        "off_policy_train_batch_size": 4096 if ram_gb >= 64 else 2048,
        "off_policy_memory_size": 5_000_000 if ram_gb >= 64 else 2_000_000,

        "evaluation_interval": frames_per_batch * 10,
        "checkpoint_interval": frames_per_batch * 10,
        "checkpoint_at_end": True,
    }


def generate_cli(cfg, algorithm, task):
    parts = [
        "python benchmarl/run.py",
        f"algorithm={algorithm}",
        f"task={task}",
    ]

    for k, v in cfg.items():
        if isinstance(v, bool):
            v = str(v).lower()
            parts.append(f"experiment.{k}={v}")
        elif isinstance(v, str):
            parts.append(f'experiment.{k}="{v}"')
        else:
            parts.append(f"experiment.{k}={v}")

    return " \\\n    ".join(parts)


def generate_yaml(cfg):
    lines = []
    for k, v in cfg.items():
        if isinstance(v, str):
            lines.append(f'{k}: "{v}"')
        elif isinstance(v, bool):
            lines.append(f"{k}: {str(v)}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="vmas/balance")
    parser.add_argument("--algorithm", default="mappo")
    parser.add_argument("--profile", choices=["vmas", "cpu"], default="vmas")
    parser.add_argument("--aggressiveness", choices=["safe", "balanced", "max"], default="max")
    args = parser.parse_args()

    info = get_system_info()
    smi = get_nvidia_smi()

    print("\n================ SYSTEM INFO ================\n")
    pprint(info)

    print("\n================ NVIDIA-SMI ================\n")
    pprint(smi)

    if args.profile == "vmas" or args.task.startswith("vmas/"):
        cfg = recommend_vmas_gpu(info, args.aggressiveness)
    else:
        cfg = recommend_cpu_bound(info, args.aggressiveness)

    print("\n================ RECOMMENDED CONFIG ================\n")
    pprint(cfg)

    print("\n================ YAML OVERRIDE ================\n")
    print(generate_yaml(cfg))

    print("\n================ BENCHMARL COMMAND ================\n")
    print(generate_cli(cfg, args.algorithm, args.task))


if __name__ == "__main__":
    main()