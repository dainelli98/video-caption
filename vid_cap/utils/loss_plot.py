# -*- coding: utf-8 -*-
"""Helper functions to plot training and validation metrics."""
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def plot_and_store_graphs(
    train_loss: list[float], val_loss: list[float], val_bleu: list[float], output_folder: Path
) -> None:
    """Plot and store graphs for training loss, validation loss, and validation BLEU scores.

    This function creates a two-row subplot where the first row is the training and
    validation loss per epoch, and the second row is the validation BLEU score per epoch.
    The plot is then saved as an interactive HTML file in the output directory.

    :param train_loss: A list of training loss values for each epoch.
    :param val_loss: A list of validation loss values for each epoch.
    :param val_bleu: A list of validation BLEU scores for each epoch.
    :param output_folder: The path to the folder where the plot will be saved.
    """
    # Create the output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True, parents=True)

    # Create subplot with 2 rows
    fig = make_subplots(rows=2, cols=1)

    # Add traces
    fig.add_trace(
        go.Scatter(y=train_loss, mode="lines", name="Train Loss", line={"color": "royalblue"}),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(y=val_loss, mode="lines", name="Validation Loss", line={"color": "firebrick"}),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(y=val_bleu, mode="lines", name="Validation BLEU", line={"color": "seagreen"}),
        row=2,
        col=1,
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="BLEU", row=2, col=1)

    # Update layout and title
    fig.update_layout(
        height=600,
        width=800,
        title_text="Train/Validation Loss and Validation BLEU per Epoch",
        title_x=0.5,
        paper_bgcolor="rgb(243, 243, 243)",
        plot_bgcolor="rgb(243, 243, 243)",
        font={"family": "Courier New, monospace", "size": 18, "color": "RebeccaPurple"},
    )

    # Save the plot as HTML
    fig.write_html(output_folder / "loss_bleu_plot.html")
