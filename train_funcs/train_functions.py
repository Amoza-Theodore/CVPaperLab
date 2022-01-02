from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datasets
import models
from settings import *

# Dataset
train_loader = datasets.train_loader
valid_loader = datasets.valid_loader

# Model
net = models.net

# Tensorboard
writer = SummaryWriter(TENSORBOARD_DIR)

# Train-Configs
criterion = CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = MultiStepLR(optimizer, gamma=0.1, milestones=MILESTONES)


def train_epoch(epoch_id):
    BATCH_SIZE = TRAIN_BATCH_SIZE
    data_loader = train_loader
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    net.train()

    loss_list = []
    acc_list = []

    pbar.set_description_str(f"Loading data, please wait.")
    for index, data in pbar:
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs.data, dim=1)
        acc = (predicted == labels).sum() / BATCH_SIZE
        lr = optimizer.param_groups[0]["lr"]
        loss_value = loss.item()
        loss_list.append(loss_value)
        acc_list.append(acc)
        writer.add_scalar(tag="train-loss", scalar_value=loss_value, global_step=epoch_id * len(data_loader) + index)
        writer.add_scalar(tag="train-acc", scalar_value=acc, global_step=epoch_id * len(data_loader) + index)

        pbar.set_description_str(f"epoch:{epoch_id:4d} | train | "
                                 f"acc:{acc:4.4f} | loss:{loss_value:4.4f} | lr:{lr:.4f}")

        if index % (len(data_loader) - 1) == 0:
            acc_avg = sum(acc_list) / len(acc_list)
            avg_loss = sum(loss_list) / len(loss_list)
            pbar.set_description_str(f"epoch:{epoch_id:4d} | train | "
                                     f"acc:{acc_avg:4.4f} | loss:{avg_loss:4.4f} | lr:{lr:.4f}")

        if DEBUG:
            if (index + 1) % (len(data_loader) // 100) == 0:
                break

    avg_loss = sum(loss_list) / len(loss_list)
    return avg_loss, loss_list


def valid_epoch(epoch_id):
    BATCH_SIZE = VALID_BATCH_SIZE
    data_loader = valid_loader
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    net.eval()

    with torch.no_grad():
        loss_list = []
        acc_list = []

        pbar.set_description_str(f"Loading data, please wait.")
        for index, data in pbar:
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(inputs)
            predicted = torch.argmax(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            acc = (predicted == labels).sum() / BATCH_SIZE
            lr = optimizer.param_groups[0]["lr"]
            loss_value = loss.item()
            loss_list.append(loss_value)
            acc_list.append(acc)
            avg_loss = sum(loss_list) / len(loss_list)
            avg_acc = sum(acc_list) / len(acc_list)

            pbar.set_description_str(f"epoch:{epoch_id:4d} | valid | "
                                     f"acc:{avg_acc:4.4f} | loss:{avg_loss:4.4f} | lr:{lr:.4f}")

            if DEBUG:
                if (index + 1) % (len(data_loader) // 100) == 0:
                    break

        writer.add_scalar(tag="valid-loss", scalar_value=avg_loss,
                          global_step=epoch_id)
        writer.add_scalar(tag="valid-acc", scalar_value=avg_acc,
                          global_step=epoch_id)

        avg_loss = sum(loss_list) / len(loss_list)
        return avg_loss, loss_list


def save_model(mode):
    save_path = os.path.join(OUTPUT_DIR, mode + ".pth")
    torch.save(net.state_dict(), save_path)


def load_model(mode):
    load_path = os.path.join(OUTPUT_DIR, mode+".pth")
    if not os.path.exists(load_path):
        print("Failed to load model: not exists! Start a new training!")
        return
    net.load_state_dict(torch.load(load_path, map_location=DEVICE), strict=False)
    print(f"Successfully loading model: {load_path}")
