# -*- coding: utf-8 -*-
"""Train - decoder."""
import torch
from loguru import logger


class DecoderTrainer:
    def __init__(self, optimizer, loss_fn, training_loader, validation_loader, model) -> None:
        self.__optimizer = optimizer
        self.__loss_fn = loss_fn
        self.__training_loader = training_loader
        self.__validation_loader = validation_loader
        self.__model = model

    def train(self, num_epochs, tb_writer):
        for epoch_index in range(num_epochs):
            logger.info("Starting epoch {}/{}", epoch_index + 1, num_epochs)
            train_loss = self.train_one_epoch(epoch_index, tb_writer)
            logger.info(
                "End of epoch {}/{}. Train loss: {}", epoch_index + 1, num_epochs, train_loss
            )
            val_loss = self.validate_one_epoch(epoch_index, tb_writer)
            logger.info(
                "End of epoch {}/{}. Validation loss: {}", epoch_index + 1, num_epochs, val_loss
            )

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(self.__training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.__optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.__model(inputs)

            # Compute the loss and its gradients
            loss = self.__loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.__optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                logger.info("  batch {} loss: {}", i + 1, last_loss)
                tb_x = epoch_index * len(self.__training_loader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        return last_loss

    def validate_one_epoch(self, epoch_index, tb_writer):
        # Set model to evaluation mode
        self.__model.eval()

        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, data in enumerate(self.__validation_loader):
                inputs, labels = data
                outputs = self.__model(inputs)

                loss = self.__loss_fn(outputs, labels)
                running_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        # Switch model back to train mode
        self.__model.train()

        avg_loss = running_loss / len(self.__validation_loader)
        accuracy = correct_predictions / total_predictions

        logger.info("Validation loss: {}", avg_loss)
        logger.info("Validation accuracy: {}", accuracy)

        tb_writer.add_scalar("Loss/validation", avg_loss, epoch_index)
        tb_writer.add_scalar("Accuracy/validation", accuracy, epoch_index)

        return avg_loss, accuracy
