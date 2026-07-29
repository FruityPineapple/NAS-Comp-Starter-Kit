"""
colab_setup.py — Run this FIRST in Google Colab to set everything up.

Usage in Colab:
    1. Upload this file + your submission.zip to Colab
    2. Run: !python colab_setup.py
    3. Then run: !cd NAS-Comp && make submission=submission all
"""

import os
import subprocess
import sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"[WARN] Command exited with code {result.returncode}")
    return result.returncode

def main():
    print("=" * 60)
    print("NAS Unseen-Data 2026 — Colab Setup")
    print("=" * 60)

    # Step 1: Clone the starter kit (or use existing)
    if not os.path.exists("NAS-Comp"):
        print("\n[1/5] Cloning starter kit...")
        run("git clone https://github.com/Towers-D/NAS-Comp-Starter-Kit.git NAS-Comp")
    else:
        print("\n[1/5] NAS-Comp directory already exists, skipping clone")

    os.chdir("NAS-Comp")

    # Step 2: Unzip submission into the submission/ folder
    print("\n[2/5] Setting up submission files...")
    if os.path.exists("/content/submission.zip"):
        run("rm -rf submission")
        run("mkdir -p submission")
        run("unzip -o /content/submission.zip -d submission/")
        print("  Extracted submission.zip into submission/")
    elif os.path.exists("submission/nas.py"):
        print("  submission/ already contains files, skipping")
    else:
        print("  [ERROR] No submission found!")
        print("  Upload submission.zip to /content/ or place files in submission/")
        sys.exit(1)

    # Step 3: Install dependencies
    print("\n[3/5] Installing dependencies...")
    # torch and torchvision are pre-installed on Colab
    run("pip install -q scikit-learn numpy")

    # Step 4: Download datasets
    print("\n[4/5] Downloading datasets...")
    run("mkdir -p datasets")

    # Check if datasets already present
    if os.path.exists("datasets/dataset_1/train_x.npy"):
        print("  Datasets already present, skipping download")
    else:
        # Try the download script first
        if os.path.exists("download_datasets.py"):
            run("python download_datasets.py")
        else:
            print("  [WARN] No download script found.")
            print("  Please download datasets manually and place in datasets/")

    # Step 5: Create predictions directory
    print("\n[5/5] Creating output directories...")
    run("mkdir -p predictions")
    run("mkdir -p package/predictions")

    # Verify setup
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)

    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  GPU: None (CPU only)")
        print("  [WARN] Enable GPU: Runtime > Change runtime type > T4 GPU")

    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

    # List submission files
    sub_files = os.listdir("submission")
    print(f"  Submission: {len(sub_files)} files")
    for f in sorted(sub_files):
        print(f"    - {f}")

    # List datasets
    if os.path.exists("datasets"):
        datasets = [d for d in os.listdir("datasets") if os.path.isdir(f"datasets/{d}")]
        print(f"  Datasets: {len(datasets)} found")
    else:
        print("  Datasets: 0 (need to download!)")

    print("\n  Next steps:")
    print("    !cd NAS-Comp && make submission=submission all")
    print("  Or to test a single dataset:")
    print("    !cd NAS-Comp && make submission=submission build")
    print("    !cd NAS-Comp/package && python main.py")


if __name__ == "__main__":
    main()
