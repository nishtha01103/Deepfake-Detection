import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_comparison(resnet_history, densenet_history):
    plt.figure(figsize=(12, 6))
    plt.plot(resnet_history, label='ResNet')
    plt.plot(densenet_history, label='DenseNet')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics_comparison(resnet_metrics, densenet_metrics):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = ['ResNet', 'DenseNet']

    data = {
        'ResNet': [resnet_metrics[metric] for metric in metrics],
        'DenseNet': [densenet_metrics[metric] for metric in metrics]
    }

    df = pd.DataFrame(data, index=metrics)

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Model Performance Comparison')
    plt.show()


def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake']
    )
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()