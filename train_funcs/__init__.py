from settings import *
from train_funcs.train_functions import scheduler, save_model, load_model

if MODEL_TYPE == "MAE":
    import train_funcs.train_mae_functions as train_mae_functions
    train_epoch = train_mae_functions.train_epoch
    valid_epoch = train_mae_functions.valid_epoch
else:
    import train_funcs.train_functions as train_functions
    train_epoch = train_functions.train_epoch
    valid_epoch = train_functions.valid_epoch
