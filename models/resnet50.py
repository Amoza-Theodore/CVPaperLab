from models.settings import *
from torchvision.models import resnet50

resnet50 = resnet50(pretrained=PRETRAINED, num_classes=NUM_CLASSES)
