from tqdm import tqdm

import train_funcs.train_functions as train_functions
from models.settings import ACC_MODE
from settings import *

# Dataset
train_loader = train_functions.train_loader
valid_loader = train_functions.valid_loader

# Model
net = train_functions.net

# Tensorboard
writer = train_functions.writer

# Train-Configs
from train_funcs.train_functions import criterion, optimizer


def train_epoch(epoch_id):
    if ACC_MODE: return train_epoch_with_acc(epoch_id)
    data_loader = train_loader
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    net.train()

    loss_list = []

    pbar.set_description_str(f"Loading data, please wait.")
    for index, data in pbar:
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        loss = net(inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        loss_value = loss.item()
        loss_list.append(loss_value)
        writer.add_scalar(tag="train-loss", scalar_value=loss_value, global_step=epoch_id * len(data_loader) + index)

        pbar.set_description_str(f"epoch:{epoch_id:4d} | train | "
                                 f"loss:{loss_value:4.4f} | lr:{lr:.4f}")

        if (index + 1) % len(data_loader) == 0:
            avg_loss = sum(loss_list) / len(loss_list)
            pbar.set_description_str(f"epoch:{epoch_id:4d} | train | "
                                     f"loss:{avg_loss:4.4f} | lr:{lr:.4f}")

        if DEBUG:
            if (index + 1) % 5 == 0:
                break

    avg_loss = sum(loss_list) / len(loss_list)
    return avg_loss, loss_list


def valid_epoch(epoch_id):
    if ACC_MODE: return valid_epoch_with_acc(epoch_id)
    data_loader = valid_loader
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    net.eval()

    with torch.no_grad():
        loss_list = []

        pbar.set_description_str(f"Loading data, please wait.")
        for index, data in pbar:
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            loss = net(inputs)

            lr = optimizer.param_groups[0]["lr"]
            loss_value = loss.item()
            loss_list.append(loss_value)
            avg_loss = sum(loss_list) / len(loss_list)

            pbar.set_description_str(f"epoch:{epoch_id:4d} | valid | "
                                     f"loss:{avg_loss:4.4f} | lr:{lr:.4f}")

            if DEBUG:
                if (index + 1) % 20 == 0:
                    break

        writer.add_scalar(tag="valid-loss", scalar_value=avg_loss,
                          global_step=epoch_id)

        avg_loss = sum(loss_list) / len(loss_list)
        return avg_loss, loss_list


def train_epoch_with_acc(epoch_id):
    BATCH_SIZE = TRAIN_BATCH_SIZE
    data_loader = train_loader
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    net.train()

    loss_list = []
    acc_list = []

    lr = optimizer.param_groups[0]["lr"]
    pbar.set_description_str(f"epoch:{epoch_id:4d} | train | acc:{0:4.4f} | loss:{0:4.4f} | lr:{lr:.4f}")
    for index, data in pbar:
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        loss, outputs = net(inputs)
        predicted = torch.argmax(outputs.data, dim=1)
        loss = 0.1 * loss + criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (predicted == labels).sum() / BATCH_SIZE
        lr = optimizer.param_groups[0]["lr"]
        loss_value = loss.item()
        loss_list.append(loss_value)
        acc_list.append(acc)
        writer.add_scalar(tag="train-loss", scalar_value=loss_value, global_step=epoch_id * len(data_loader) + index)
        writer.add_scalar(tag="train-acc", scalar_value=acc, global_step=epoch_id * len(data_loader) + index)

        pbar.set_description_str(f"epoch:{epoch_id:4d} | train | "
                                 f"acc:{acc:4.4f} | loss:{loss_value:4.4f} | lr:{lr:.4f}")

        if (index + 1) % len(data_loader) == 0:
            acc_avg = sum(acc_list) / len(acc_list)
            avg_loss = sum(loss_list) / len(loss_list)
            pbar.set_description_str(f"epoch:{epoch_id:4d} | train | "
                                     f"acc:{acc_avg:4.4f} | loss:{avg_loss:4.4f} | lr:{lr:.4f}")

        if DEBUG:
            if (index + 1) % 5 == 0:
                break

    avg_loss = sum(loss_list) / len(loss_list)
    return avg_loss, loss_list


def valid_epoch_with_acc(epoch_id):
    BATCH_SIZE = VALID_BATCH_SIZE
    data_loader = valid_loader
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    net.eval()

    with torch.no_grad():
        loss_list = []
        acc_list = []

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_description_str(f"epoch:{epoch_id:4d} | valid | acc:{0:4.4f} | loss:{0:4.4f} | lr:{lr:.4f}")
        for index, data in pbar:
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            loss, outputs = net(inputs)
            predicted = torch.argmax(outputs.data, dim=1)

            loss = 0.1 * loss + criterion(outputs, labels)
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
                if (index + 1) % 20 == 0:
                    break

        writer.add_scalar(tag="valid-loss", scalar_value=avg_loss,
                          global_step=epoch_id)
        writer.add_scalar(tag="valid-acc", scalar_value=avg_acc,
                          global_step=epoch_id)

        avg_loss = sum(loss_list) / len(loss_list)
        return avg_loss, loss_list
