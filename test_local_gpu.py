import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

print("=" * 70)
print("  CareFlow AI - Local Fine-Tuned Model & GPU Diagnostic")
print("=" * 70)

# 1. Hardware & PyTorch CUDA Check
print("\n[1/4] Checking Physical Hardware & PyTorch...")
cuda_ok = False
try:
    import torch
    print(f"  - PyTorch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"  - CUDA Available in PyTorch: {cuda_ok}")
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  - Detected GPU: {device_name}")
        print(f"  - Total VRAM: {vram_gb:.2f} GB (MedGemma 4-bit requires ~3.2 GB)")
    else:
        print("  [!] WARNING: PyTorch currently installed is CPU-only (torch+cpu).")
        print("      To enable your NVIDIA RTX 4050 GPU, run:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("      pip install bitsandbytes")
except ImportError:
    print("  [!] PyTorch is not installed.")
    cuda_ok = False

# 2. Fine-Tuned LoRA Adapter Check
print("\n[2/4] Checking Fine-Tuned LoRA Adapter ('careflow-medgemma-final')...")
adapter_dir = BASE_DIR / "careflow-medgemma-final"
if adapter_dir.exists():
    import json
    cfg_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if cfg_path.exists() and weights_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        print(f"  - Adapter found at: {adapter_dir}")
        print(f"  - LoRA Weights size: {weights_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"  - LoRA Rank (r): {cfg.get('r')}")
        print(f"  - LoRA Alpha: {cfg.get('lora_alpha')}")
        print(f"  - Base Model ID: {cfg.get('base_model_name_or_path')}")
        print(f"  - Target Modules: {cfg.get('target_modules')}")
    else:
        print(f"  [!] Missing config or safetensors in {adapter_dir}")
else:
    print(f"  [!] Adapter directory not found at {adapter_dir}")

# 3. Environment Config Check
print("\n[3/4] Checking CareFlow Configuration...")
from backend.config import settings
print(f"  - MEDGEMMA_MODE: {settings.MEDGEMMA_MODE}")
print(f"  - MEDGEMMA_BASE_MODEL_ID: {settings.MEDGEMMA_BASE_MODEL_ID}")
print(f"  - MEDGEMMA_ADAPTER_PATH: {settings.MEDGEMMA_ADAPTER_PATH}")
print(f"  - MEDGEMMA_LOCAL_QUANTIZATION: {settings.MEDGEMMA_LOCAL_QUANTIZATION}")

# 4. Service Initialisation Check
print("\n[4/4] Initializing MedGemmaService...")
from backend.services.medgemma import MedGemmaService
svc = MedGemmaService()
status = svc.get_status()
print(f"  - Service Mode: {status['mode']}")
print(f"  - Has Adapter: {status['has_adapter']}")
print(f"  - Ready: {status['is_ready']}")

print("\n" + "=" * 70)
if cuda_ok:
    print("  Status: GPU READY FOR FULL POWER LOCAL INFERENCE!")
else:
    print("  Status: Hardware detected, awaiting PyTorch CUDA wheel.")
print("=" * 70)
