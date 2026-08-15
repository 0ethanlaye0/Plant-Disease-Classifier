"""Contains Relevant Helper Functions"""

import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms


device = 'cuda' if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_pred, y_true):
    summ = torch.eq(y_pred, y_true).sum()
    return (summ / len(y_pred)) * 100

def train_model(model: torch.nn.Module,
                data_loader:torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                accuracy_fn,
                opt: torch.optim.Optimizer):
    all_preds, all_labels = [], []
    model.train()
    train_loss, train_acc = 0, 0
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        all_preds += preds
        all_labels += labels
        loss = loss_fn(logits, labels)
        train_loss += loss
        acc = accuracy_fn(preds, labels)
        train_acc += acc
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
    return [all_preds, all_labels]

def test_model(model: torch.nn.Module,
                data_loader:torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                accuracy_fn):
    all_preds, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
            loss = loss_fn(logits, labels)
            test_loss += loss
            acc = accuracy_fn(preds, labels)
            test_acc += acc
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")
    return [torch.cat(all_preds), torch.cat(all_labels)]

def predict(img_path: Path,
            model: torch.nn.Module,
            class_names,
            device: torch.device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logit = model(img_tensor)
        pred = torch.argmax(torch.softmax(logit, dim=1), dim=1)
    return class_names[pred.item()]
