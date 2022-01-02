import settings
import dataset
import os
import torch
from model import mae

# Model
net = mae

load_path = os.path.join(settings.OUTPUT_DIR, "model_weights.pth")
net.load_state_dict(torch.load(load_path, map_location=settings.DEVICE))

# Dataset
data_loader = iter(dataset.valid_loader)
inputs, labels = next(data_loader)
loss = net(inputs)
