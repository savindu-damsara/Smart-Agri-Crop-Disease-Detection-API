from pathlib import Path
import csv

import torch
import torch.nn as nn


class Trainer:
    """
    Handles model training and validation.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpoint_dir,
        log_file,
        gradient_clip=1.0,
    ):

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.device = device

        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.log_file = Path(
            log_file
        )

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.gradient_clip = gradient_clip

        self.best_val_loss = float("inf")

        self.best_val_accuracy = 0.0

        self._create_log_file()


    # ========================================================
    # Create CSV Log
    # ========================================================

    def _create_log_file(self):

        if not self.log_file.exists():

            with open(
                self.log_file,
                "w",
                newline="",
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy",
                    "learning_rate",
                ])


    # ========================================================
    # Training
    # ========================================================

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in self.train_loader:

            images = images.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )


            # Reset gradients
            self.optimizer.zero_grad(
                set_to_none=True
            )


            # Forward pass
            outputs = self.model(
                images
            )


            # Calculate loss
            loss = self.criterion(
                outputs,
                labels
            )


            # Backpropagation
            loss.backward()


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )


            # Update parameters
            self.optimizer.step()


            # Statistics

            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


        epoch_loss = (
            running_loss / total
        )

        epoch_accuracy = (
            correct / total
        )


        return (
            epoch_loss,
            epoch_accuracy
        )


    # ========================================================
    # Validation
    # ========================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in self.val_loader:

            images = images.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )


            outputs = self.model(
                images
            )


            loss = self.criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


        epoch_loss = (
            running_loss / total
        )

        epoch_accuracy = (
            correct / total
        )


        return (
            epoch_loss,
            epoch_accuracy
        )


    # ========================================================
    # Save Checkpoint
    # ========================================================

    def save_checkpoint(
        self,
        epoch,
        val_loss,
        val_accuracy,
    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                (
                    self.scheduler.state_dict()
                    if self.scheduler
                    else None
                ),

            "val_loss": val_loss,

            "val_accuracy": val_accuracy,
        }


        checkpoint_path = (
            self.checkpoint_dir
            / "best_model.pt"
        )


        torch.save(
            checkpoint,
            checkpoint_path
        )


        return checkpoint_path


    # ========================================================
    # Complete Training
    # ========================================================

    def fit(
        self,
        epochs,
        patience=5,
    ):

        epochs_without_improvement = 0


        for epoch in range(
            1,
            epochs + 1
        ):

            print()
            print(
                f"Epoch {epoch}/{epochs}"
            )

            print("-" * 50)


            # -----------------------------
            # Training
            # -----------------------------

            train_loss, train_accuracy = (
                self.train_one_epoch()
            )


            # -----------------------------
            # Validation
            # -----------------------------

            val_loss, val_accuracy = (
                self.validate()
            )


            # -----------------------------
            # Scheduler
            # -----------------------------

            if self.scheduler:

                self.scheduler.step(
                    val_loss
                )


            current_lr = (
                self.optimizer
                .param_groups[0]
                ["lr"]
            )


            # -----------------------------
            # Print metrics
            # -----------------------------

            print(
                f"Train Loss      : "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy  : "
                f"{train_accuracy:.4f}"
            )

            print(
                f"Validation Loss : "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation Acc  : "
                f"{val_accuracy:.4f}"
            )

            print(
                f"Learning Rate   : "
                f"{current_lr:.8f}"
            )


            # -----------------------------
            # CSV logging
            # -----------------------------

            with open(
                self.log_file,
                "a",
                newline="",
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    epoch,
                    train_loss,
                    train_accuracy,
                    val_loss,
                    val_accuracy,
                    current_lr,
                ])


            # -----------------------------
            # Check improvement
            # -----------------------------

            improved = (
                val_loss
                < self.best_val_loss
            )


            if improved:

                self.best_val_loss = (
                    val_loss
                )

                self.best_val_accuracy = (
                    val_accuracy
                )

                epochs_without_improvement = 0


                checkpoint_path = (
                    self.save_checkpoint(
                        epoch,
                        val_loss,
                        val_accuracy,
                    )
                )


                print(
                    f"\n✓ New best model saved:"
                )

                print(
                    checkpoint_path
                )


            else:

                epochs_without_improvement += 1


                print(
                    "\nNo validation improvement."
                )

                print(
                    f"Patience: "
                    f"{epochs_without_improvement}"
                    f"/{patience}"
                )


            # -----------------------------
            # Early stopping
            # -----------------------------

            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break


        print()
        print("=" * 60)

        print(
            "TRAINING COMPLETED"
        )

        print("=" * 60)

        print(
            f"Best validation loss: "
            f"{self.best_val_loss:.4f}"
        )

        print(
            f"Best validation accuracy: "
            f"{self.best_val_accuracy:.4f}"
        )