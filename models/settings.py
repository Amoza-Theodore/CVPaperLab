import settings

MODEL_TYPE = settings.MODEL_TYPE
NUM_CLASSES = settings.NUM_CLASSES
IMAGE_SIZE = settings.IMAGE_SIZE
DEVICE = settings.DEVICE

# -------- Model-ViT-Base -------- #
PATCH_SIZE = 16
DIM = 768
DEPTH = 12
HEADS = 12
MLP_DIM = 3072

# -------- Model-MAE -------- #
MASKING_RATIO = 0.75
ACC_MODE = True

# -------- Model-RESNET50 | RESNET18 -------- #
PRETRAINED = True
