#!/bin/bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=23:59:00
#$ -N v301
#$ -o log/output_v301.log
#$ -e log/error_v301.log

echo "Job started at: $(date)"
df -h

# --- GPU モジュール ---
module load cuda/12.1.0
module load cudnn/9.0.0
module load nccl/2.20.5
echo "Modules loaded (CUDA 12.1 / cuDNN 9 / NCCL 2.20)"

# --- 仮想環境構築 ---
VENV_DIR=venv_midas
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

# ---- pip 自体を 23 系にしておく（22.3.1 固定でも可）----
python -m pip install --upgrade pip==23.3.1

# ---- Core DL stack (H100 対応) ----
pip install --upgrade-strategy only-if-needed \
  torch==2.3.0+cu121 \
  torchvision==0.18.0+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121

# ---- 研究コードが要求する追加パッケージ（バージョン固定） ----
pip install --upgrade-strategy only-if-needed \
  numpy==1.23.4 \
  opencv-python==4.6.0.66 \
  imutils==0.5.4 \
  timm==0.6.12 \
  einops==0.6.0 \
  matplotlib==3.6.2 \
  open_clip_torch==2.32.0 \
  kornia==0.7.0 \
  seaborn==0.13.0 \


# ---- cupy (CUDA 12 対応) ----
pip install --upgrade-strategy only-if-needed cupy-cuda12x

# clip
pip install git+https://github.com/openai/CLIP.git

# ---- バージョン確認 ----
python - <<'PY'
import sys, torch, torchvision, timm, kornia, cupy
print("Python     ", sys.version.split()[0])
print("PyTorch    ", torch.__version__)
print("TorchVision", torchvision.__version__)
print("CUDA       ", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0))
print("timm       ", timm.__version__)
print("kornia     ", kornia.__version__)
print("cupy       ", cupy.__version__)
PY

# ---- 学習実行 ----
echo "Launching training..."
python -u main_v301.py

echo "Job finished at: $(date)"