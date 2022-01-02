from models.settings import *

if MODEL_TYPE == "MAE":
    from models.mae import mae
    net = mae
elif MODEL_TYPE == "ViT":
    from models.vit import vit
    net = vit
elif MODEL_TYPE == "RESNET50":
    from models.resnet50 import resnet50
    net = resnet50
elif MODEL_TYPE == "RESNET18":
    from models.resnet18 import resnet18
    net = resnet18
else:
    raise Exception("Wrong MODEL_TYPE!")

net = net.to(DEVICE)