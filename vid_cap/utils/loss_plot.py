# -*- coding: utf-8 -*-
"""Helper functions to plot training and validation metrics."""
import matplotlib.pyplot as plt


def plot_traing_losses(train_loss: list[float], val_loss: list[float]) -> None:
    """Plot train anv validation losses.

    :param train_loss: Training loss.
    :param val_loss: Validation loss.
    """
    # Create a list for the number of epochs
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.plot(epochs, train_loss, "r", label="Training Loss")

    # Plot validation loss
    plt.plot(epochs, val_loss, "b", label="Validation Loss")

    # Add labels and title
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Show the plot
    plt.show()


def plot_bleu_scores(scores: list[float]) -> None:
    """Plot Bleu scores.

    :param scores: List of bleu scores.
    """
    # Create a list for the number of epochs
    epochs = range(1, len(scores) + 1)

    plt.figure(figsize=(10, 5))

    # Plot Bleu scores
    plt.plot(epochs, scores, "b", label="Bleu score")

    # Add labels and title
    plt.title("Bleu Score on validation")
    plt.xlabel("Epochs")
    plt.ylabel("Bleu score")
    plt.legend()

    # Show the plot
    plt.show()
