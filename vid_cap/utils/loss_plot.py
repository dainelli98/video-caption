import matplotlib.pyplot as plt
from datetime import datetime

def plot_traing_losses(train_loss, val_loss, file_suffix):

    # Create a list for the number of epochs
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.plot(epochs, train_loss, 'r', label='Training Loss')

    # Plot validation loss
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')

    # Add labels and title
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    now = datetime.now()
    plt.savefig(f'{now.strftime("%d%m%Y%H%M%S")}-{file_suffix}-LOSS.png')


def plot_bleu_scores(scores, file_suffix):
     # Create a list for the number of epochs
    epochs = range(1, len(scores) + 1)

    plt.figure(figsize=(10, 5))

    #Plot Bleu scores
    plt.plot(epochs, scores, 'b', label='Bleu score')

    # Add labels and title
    plt.title('Bleu Score on validation')
    plt.xlabel('Epochs')
    plt.ylabel('Bleu score')
    plt.legend()

    now = datetime.now()
    plt.savefig(f'{now.strftime("%d%m%Y%H%M%S")}-{file_suffix}-BLEU.png')