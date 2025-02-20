import torch
import torch.nn as nn
from config import Config
from data_loader import load_data
from resnet_model import ResNetModel
from densenet_model import DenseNetModel
from trainer import train_model
from evaluator import evaluate_model
from plotter import plot_comparison, plot_metrics_comparison, plot_confusion_matrix


def main():
    print("Starting the training process...")

    # Load data
    print("Loading datasets...")
    train_loader = load_data(Config.TRAIN_DIR, Config.BATCH_SIZE)
    val_loader = load_data(Config.VAL_DIR, Config.BATCH_SIZE)
    test_loader = load_data(Config.TEST_DIR, Config.BATCH_SIZE)

    # Initialize models
    print("Initializing models...")
    resnet_model = ResNetModel().to(Config.DEVICE)
    densenet_model = DenseNetModel().to(Config.DEVICE)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    resnet_optimizer = torch.optim.Adam(resnet_model.parameters(), lr=Config.LEARNING_RATE)
    densenet_optimizer = torch.optim.Adam(densenet_model.parameters(), lr=Config.LEARNING_RATE)

    # Train models
    print("\nTraining ResNet...")
    resnet_train_history, resnet_val_history = train_model(
        resnet_model, train_loader, val_loader, criterion, resnet_optimizer,
        Config.DEVICE, Config.NUM_EPOCHS
    )

    print("\nTraining DenseNet...")
    densenet_train_history, densenet_val_history = train_model(
        densenet_model, train_loader, val_loader, criterion, densenet_optimizer,
        Config.DEVICE, Config.NUM_EPOCHS
    )

    # Evaluate models
    print("\nEvaluating models...")
    resnet_metrics = evaluate_model(resnet_model, test_loader, Config.DEVICE)
    densenet_metrics = evaluate_model(densenet_model, test_loader, Config.DEVICE)

    # Plot results
    plot_comparison(resnet_val_history, densenet_val_history)
    plot_metrics_comparison(resnet_metrics, densenet_metrics)

    # Plot confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(resnet_metrics['confusion_matrix'], 'ResNet')
    plot_confusion_matrix(densenet_metrics['confusion_matrix'], 'DenseNet')

    # Print final metrics
    print("\nResNet Metrics:", {k: v for k, v in resnet_metrics.items() if k != 'confusion_matrix'})
    print("DenseNet Metrics:", {k: v for k, v in densenet_metrics.items() if k != 'confusion_matrix'})


if __name__ == "__main__":
    main()