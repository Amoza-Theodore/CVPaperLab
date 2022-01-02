import os
import random
import shutil
import numpy as np
import torch

# ------- PLATFORM SETTINGS -------- #
os.chdir("/content/drive/MyDrive/PAPER-RETRIEVAL")
DEBUG = False

# ------- HYPER PARAMETERS SETTINGS -------- #
TRAIN_ID = 1
TRAIN_BATCH_SIZE = 256  # [A100-MAE-1024]
VALID_BATCH_SIZE = TRAIN_BATCH_SIZE // 8
EPOCHS = 200
IMAGE_SIZE = 224  # [IMAGENET1K-224, CIFAR10-32]
LR = 0.1
MILESTONES = [EPOCHS]
NUM_CLASSES = 1000  # [IMAGENET1K-1000, CIFAR10-10]
WEIGHT_DECAY = 0.0

# ------- OPTION BOX -------- #
MODEL_TYPE = "RESNET18"  # ["RESNET18", "RESNET50", "ViT", "MAE"]
DATASET_TYPE = "IMAGENET1K"  # ["IMAGENET1K", "CIFAR10"]
OUTPUT_DIRNAME = f"{MODEL_TYPE}_{DATASET_TYPE}"
RETRAIN_MODE = "latest"  # ["latest", "best_accuracy"]

# #----------------------------# #
# -------- MESSAGE SETTINGS -------- #
if DEBUG: print("Warning: now in DEBUG mode!")
if not DEBUG: print(f"TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE}")
if not DEBUG: print(f"MODEL_TYPE:{MODEL_TYPE} | DATASET_TYPE:{DATASET_TYPE}")

# -------- FOLDER SETTINGS -------- #
DATA_DIR = "/content/drive/MyDrive/DATASETS/"
OUTPUT_DIR = os.path.join("./output", OUTPUT_DIRNAME)
TENSORBOARD_DIR = "./runs"
if not DEBUG:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# ------- SEED SETTINGS ------- #
seed_value = 114514 + (TRAIN_ID - 1)

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

# ------- DEVICE settings ------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------- BACKUP settings ------- #
if not DEBUG:
    shutil.copy("settings.py", OUTPUT_DIR)
    shutil.copy("models/settings.py", os.path.join(OUTPUT_DIR, "model_settings.py"))
