import torch
class Config:
    # Data paths
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/validation'
    TEST_DIR = 'data/test'

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001

    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'