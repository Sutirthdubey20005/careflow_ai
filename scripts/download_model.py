import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

token = os.getenv("HF_TOKEN", "")
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

target_file = models_dir / "medgemma-4b-it-Q4_K_M.gguf"
temp_file = models_dir / "medgemma-4b-it-Q4_K_M.gguf.part"

url = "https://huggingface.co/unsloth/medgemma-4b-it-GGUF/resolve/main/medgemma-4b-it-Q4_K_M.gguf"

if target_file.exists() and target_file.stat().st_size > 2400000000:
    print(f"Model already fully downloaded: {target_file} ({target_file.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
    sys.exit(0)

existing_bytes = temp_file.stat().st_size if temp_file.exists() else 0
headers = {"Authorization": f"Bearer {token}"}
if existing_bytes > 0:
    headers["Range"] = f"bytes={existing_bytes}-"
    print(f"Resuming download from byte {existing_bytes} ({existing_bytes / 1024 / 1024:.2f} MB)...", flush=True)
else:
    print(f"Starting fresh download to {temp_file}...", flush=True)

resp = requests.get(url, headers=headers, stream=True, timeout=60)
if resp.status_code not in (200, 206):
    print(f"Download failed with status {resp.status_code}: {resp.text[:200]}", flush=True)
    sys.exit(1)

content_range = resp.headers.get("Content-Range", "")
if "bytes" in content_range:
    total_bytes = int(content_range.split("/")[-1])
else:
    total_bytes = int(resp.headers.get("Content-Length", 0)) + existing_bytes

downloaded = existing_bytes
start_time = time.time()
last_report = start_time

mode = "ab" if existing_bytes > 0 else "wb"
with open(temp_file, mode) as f:
    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_report >= 5:
                speed_mb = (downloaded - existing_bytes) / (1024 * 1024 * (now - start_time + 0.001))
                pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                print(f"Downloaded: {downloaded / 1024 / 1024:.1f}/{total_bytes / 1024 / 1024:.1f} MB ({pct:.1f}%) at {speed_mb:.2f} MB/s", flush=True)
                last_report = now

print(f"Download complete: {downloaded} bytes. Renaming to {target_file.name}...", flush=True)
if target_file.exists():
    target_file.unlink()
temp_file.rename(target_file)
print("Finished successfully!", flush=True)
