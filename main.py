from settings import *
from train_funcs import train_epoch, valid_epoch
from train_funcs import scheduler, save_model, load_model

if __name__ == '__main__':
    if RETRAIN_MODE: load_model(RETRAIN_MODE)
    min_loss = np.Inf
    for epoch_id in range(EPOCHS):
        train_loss, *_ = train_epoch(epoch_id)
        valid_loss, *_ = valid_epoch(epoch_id)
        scheduler.step()

        if DEBUG: continue
        save_model("latest")
        if valid_loss < min_loss:
            min_loss = valid_loss
            save_model("best_accuracy")
