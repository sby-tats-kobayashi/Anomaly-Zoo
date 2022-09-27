# Data augmentation parameters
ROT_ANGLE = 5
IMG_SIZE = 256
W_SHIFT_RANGE = 0.05
H_SHIFT_RANGE = 0.05
BRIGHTNESS_RANGE = 0.2
VAL_SPLIT = 0.1  # ratio that divide whole data to train/valid data.

# Model hyper parameters
ENCODED_DIM = 128

# Learning Rate Finder parameters
START_LR = 1e-5
LR_MAX_EPOCHS = 10
LRF_DECREASE_FACTOR = 0.85

# Training hyper parameters
BATCH_SIZE = 16
EARLY_STOPPING = 12
REDUCE_ON_PLATEAU = 6


# Fine tuning parameters
FINETUNE_SPLIT = 0.1
STEP_MIN_AREA = 5
START_MIN_AREA = 5
STOP_MIN_AREA = 1005