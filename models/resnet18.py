from models.settings import *
from torchvision.models import resnet18

resnet18 = resnet18(pretrained=PRETRAINED, num_classes=NUM_CLASSES)
