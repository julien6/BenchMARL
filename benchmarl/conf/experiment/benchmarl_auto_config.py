import os
import math
import psutil
import torch
import subprocess
from pprint import pprint

# ============================================================
# SYSTEM DETECTION
# ============================================================


def get_cpu_info():
    return {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": psutil.cpu_freq().max,
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "load_avg": os.getloadavg(),
    }



def get_ram_info():
    ram = psutil.virtual_memory()

    return {
        "total_gb": round(ram.total / 1024**3, 2),
        "available_gb": round(ram.available / 1024**3, 2),
        "used_percent": ram.percent,
    }


# ============================================================
# GPU DETECTION
# ============================================================


def get_gpu_info():

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "gpu_count": 0,
            "gpus": [],
        }

    gpus = []

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        gpus.append({
            "id": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "multi_processor_count": props.multi_processor_count,
            "max_threads_per_block": props.max_threads_per_block,
        })

    return {
        "cuda_available": True,
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpus,
    }


# ============================================================
# LIVE NVIDIA-SMI STATS
# ============================================================


def get_nvidia_smi():

    try:

        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]).decode().strip()

        stats = []

        for line in result.split("\n"):
            vals = [x.strip() for x in line.split(",")]

            stats.append({
                "gpu_util_percent": int(vals[0]),
                "mem_util_percent": int(vals[1]),
                "memory_total_mb": int(vals[2]),
                "memory_used_mb": int(vals[3]),
                "power_w": float(vals[4]),
                "temp_c": int(vals[5]),
            })

        return stats

    except Exception as e:
        return {
            "error": str(e)
        }


# ============================================================
# AUTO-TUNING
# ============================================================


def recommend_config(cpu_info, ram_info, gpu_info):

    physical_cores = cpu_info["physical_cores"]
    logical_cores = cpu_info["logical_cores"]
    total_ram = ram_info["total_gb"]

    config = {}

    # --------------------------------------------------------
    # DEVICE SELECTION
    # --------------------------------------------------------

    if gpu_info["cuda_available"]:

        config["train_device"] = "cuda"
        config["sampling_device"] = "cuda"

        # Replay buffers are often better on CPU
        config["buffer_device"] = "cpu"

    else:

        config["train_device"] = "cpu"
        config["sampling_device"] = "cpu"
        config["buffer_device"] = "cpu"

    # --------------------------------------------------------
    # ENV PARALLELISM
    # --------------------------------------------------------

    if physical_cores >= 64:
        n_envs_per_worker = 64
    elif physical_cores >= 32:
        n_envs_per_worker = 32
    elif physical_cores >= 16:
        n_envs_per_worker = 16
    elif physical_cores >= 8:
        n_envs_per_worker = 8
    else:
        n_envs_per_worker = 4

    config["on_policy_n_envs_per_worker"] = n_envs_per_worker
    config["off_policy_n_envs_per_worker"] = n_envs_per_worker

    # --------------------------------------------------------
    # FRAMES PER BATCH
    # --------------------------------------------------------

    # Rough heuristic:
    # more envs -> larger batch

    frames_per_batch = n_envs_per_worker * 512

    # Align power of two
    frames_per_batch = 2 ** math.ceil(math.log2(frames_per_batch))

    config["on_policy_collected_frames_per_batch"] = frames_per_batch
    config["off_policy_collected_frames_per_batch"] = frames_per_batch

    # --------------------------------------------------------
    # MINIBATCH SIZE
    # --------------------------------------------------------

    minibatch_size = max(512, frames_per_batch // 8)

    # Keep powers of two
    minibatch_size = 2 ** math.floor(math.log2(minibatch_size))

    config["on_policy_minibatch_size"] = minibatch_size

    # --------------------------------------------------------
    # PPO-LIKE ITERATIONS
    # --------------------------------------------------------

    if gpu_info["cuda_available"]:
        config["on_policy_n_minibatch_iters"] = 20
    else:
        config["on_policy_n_minibatch_iters"] = 10

    # --------------------------------------------------------
    # OFF-POLICY BATCH SIZE
    # --------------------------------------------------------

    if total_ram >= 128:
        train_batch_size = 8192
        memory_size = 10_000_000

    elif total_ram >= 64:
        train_batch_size = 4096
        memory_size = 5_000_000

    elif total_ram >= 32:
        train_batch_size = 2048
        memory_size = 2_000_000

    else:
        train_batch_size = 1024
        memory_size = 1_000_000

    config["off_policy_train_batch_size"] = train_batch_size
    config["off_policy_memory_size"] = memory_size

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    config["evaluation_interval"] = frames_per_batch * 20
    config["checkpoint_interval"] = frames_per_batch * 20
    config["checkpoint_at_end"] = True

    return config


# ============================================================
# COMMAND GENERATOR
# ============================================================


def generate_cli(config, algorithm="mappo", task="vmas/balance"):

    cmd = [
        "python benchmarl/run.py",
        f"algorithm={algorithm}",
        f"task={task}",
    ]

    for k, v in config.items():

        if isinstance(v, str):
            cmd.append(f'experiment.{k}="{v}"')
        else:
            cmd.append(f"experiment.{k}={v}")

    return " \\\n    ".join(cmd)


# ============================================================
# YAML GENERATOR
# ============================================================


def generate_yaml(config):

    lines = []

    for k, v in config.items():
        lines.append(f"{k}: {v}")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    print("\n================ CPU INFO ================\n")
    cpu_info = get_cpu_info()
    pprint(cpu_info)

    print("\n================ RAM INFO ================\n")
    ram_info = get_ram_info()
    pprint(ram_info)

    print("\n================ GPU INFO ================\n")
    gpu_info = get_gpu_info()
    pprint(gpu_info)

    print("\n================ NVIDIA-SMI ================\n")
    gpu_stats = get_nvidia_smi()
    pprint(gpu_stats)

    print("\n================ AUTO CONFIG ================\n")

    config = recommend_config(
        cpu_info,
        ram_info,
        gpu_info,
    )

    pprint(config)

    print("\n================ YAML OVERRIDE ================\n")

    yaml_text = generate_yaml(config)
    print(yaml_text)

    print("\n================ BENCHMARL COMMAND ================\n")

    cmd = generate_cli(config)
    print(cmd)