# train_emotion_cnn.py

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm


class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()

        # Input: (1, 48, 48)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (32, 24, 24)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (64, 12, 12)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (128, 6, 6)
        )

        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_dataloaders(train_dir, val_dir, img_size=(48, 48), batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_loss = running_loss / total
        current_acc = 100.0 * correct / total
        loop.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_loss = running_loss / total
        current_acc = 100.0 * correct / total
        loop.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    return running_loss / total, correct / total


def save_checkpoint(save_path, epoch, model, optimizer, total_epochs):
    checkpoint = {
        "epoch": epoch,  # 0-based index
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_epochs_planned": total_epochs,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint_if_exists(save_path, model, optimizer, device, resume_lr=None):
    last_completed = -1
    planned_total_epochs = 0

    if not os.path.exists(save_path):
        print(f"[START FRESH] No checkpoint found at: {save_path}")
        return last_completed, planned_total_epochs

    print(f"[RESUME] Found checkpoint: {save_path}. Loading...")
    ckpt = torch.load(save_path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print("[WARN] state_dict mismatch")
        if missing:
            print("  Missing keys:", missing)
        if unexpected:
            print("  Unexpected keys:", unexpected)

    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Move optimizer tensors to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    except Exception as e:
        print(f"[INFO] Optimizer state not loaded: {e} (starting optimizer fresh)")

    last_completed = int(ckpt.get("epoch", -1))
    planned_total_epochs = int(ckpt.get("num_epochs_planned", 0))

    print(f"[RESUME] Last completed epoch index: {last_completed} (epoch {last_completed + 1})")

    if resume_lr is not None:
        for g in optimizer.param_groups:
            g["lr"] = resume_lr
        print("Current LR(s):", [g["lr"] for g in optimizer.param_groups])

    return last_completed, planned_total_epochs


def parse_args():
    parser = argparse.ArgumentParser(description="Train Emotion CNN (PyTorch)")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training images folder")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation images folder")
    parser.add_argument("--save_path", type=str, default="model_checkpoint.pth", help="Checkpoint file path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume_lr", type=float, default=5e-4, help="LR to use when resuming")
    parser.add_argument("--extra_epochs", type=int, default=30, help="How many additional epochs to run")
    parser.add_argument("--num_classes", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, class_names = build_dataloaders(
        args.train_dir,
        args.val_dir,
        img_size=(48, 48),
        batch_size=args.batch_size,
    )
    print("Classes:", class_names)

    model = EmotionCNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    last_completed, _ = load_checkpoint_if_exists(
        args.save_path,
        model,
        optimizer,
        device,
        resume_lr=args.resume_lr,
    )

    start_epoch = last_completed + 1
    total_epochs = start_epoch + args.extra_epochs
    print(f"[PLAN] Resume from epoch {start_epoch}, run until {total_epochs - 1} (extra epochs: {args.extra_epochs})")

    try:
        for epoch in range(start_epoch, total_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, total_epochs
            )
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device, epoch, total_epochs
            )

            print(
                f"Epoch [{epoch+1}/{total_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
            )

            save_checkpoint(args.save_path, epoch, model, optimizer, total_epochs)

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Saving checkpoint...")
        # Save last completed epoch if interrupted mid-training
        save_checkpoint(args.save_path, max(start_epoch - 1, 0), model, optimizer, total_epochs)

    except Exception as e:
        print(f"\n[ERROR] Unexpected error during training: {e}")
        print("Saving checkpoint...")
        save_checkpoint(args.save_path, max(start_epoch - 1, 0), model, optimizer, total_epochs)
        raise

    finally:
        # Final save (best effort)
        final_epoch = total_epochs - 1
        save_checkpoint(args.save_path, final_epoch, model, optimizer, total_epochs)
        print(f"Checkpoint saved to: {args.save_path}")
        print(f"Last completed epoch index: {final_epoch} (epoch {final_epoch + 1})")


if __name__ == "__main__":
    main()
