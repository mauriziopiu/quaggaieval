# launch_training.py
# Run this instead of calling training/train.py directly.
# Usage: python launch_training.py

import os
import setproctitle

# ── Process identity (visible in nvidia-smi and ps) ───────────────────────
setproctitle.setproctitle('quaggaieval - finetuning')

# ── GPU selection ─────────────────────────────────────────────────────────
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# USE_LIBUV=0 works around a known PyTorch distributed init issue 
# on some Linux setups. 
# Include by default — harmless if not needed
os.environ['USE_LIBUV'] = '0' 

# ── These must be set before any torch import ─────────────────────────────
# Hand off to SAM2's training entry point by running it as __main__
import sys
sys.argv = [
    'training/train.py',
    '-c', 'configs/sam2.1_training/quaggai_finetune.yaml',
    '--use-cluster', '0',
    '--num-gpus', '1',
]

# Import and run the training script directly
import runpy
runpy.run_path('training/train.py', run_name='__main__')