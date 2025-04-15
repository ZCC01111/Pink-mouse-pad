import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from models import UNet, PterygiumClassifier
from utils import PterygiumDataset, load_dataset

def train_segmentation(model, dataloader, criterion, optimizer, device, save_path):
    best_loss = float('inf')
    model.train()
    for epoch in range(20):
        epoch_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/20]")
        for imgs, masks, _ in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)

# def train_classifier(model, dataloader, criterion, optimizer, device, save_path):
#     best_acc = 0
#     model.train()
#     for epoch in range(10):
#         correct = total = 0
#         loop = tqdm(dataloader, desc=f"Classifier Epoch [{epoch+1}/10]")
#         for imgs, labels in loop:
#             imgs, labels = imgs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             preds = torch.argmax(outputs, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#             loop.set_postfix(acc=correct/total)
#
#         acc = correct / total
#         if acc > best_acc:
#             best_acc = acc
#             torch.save(model.state_dict(), save_path)

def train_classifier(model, dataloader, criterion, optimizer, device, save_path):
    best_acc = 0.0
    for epoch in range(10):
        model.train()
        total = 0
        correct = 0
        loop = tqdm(dataloader, desc=f'Epoch {epoch+1}/10')
        for imgs, _, labels in loop:  # 忽略 mask
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            acc = correct / total
            loop.set_postfix(loss=loss.item(), acc=acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('best_models', exist_ok=True)

    train_imgs, test_imgs, train_labels, test_labels, train_masks, test_masks = load_dataset('../data/train')

    seg_model = UNet().to(device)
    seg_loader = DataLoader(PterygiumDataset(train_imgs, train_labels, train_masks), batch_size=4, shuffle=True)
    train_segmentation(seg_model, seg_loader, nn.BCEWithLogitsLoss(), torch.optim.Adam(seg_model.parameters(), 1e-3),
                       device, 'best_models/best_seg.pth')

    clf_model = PterygiumClassifier().to(device)
    clf_loader = DataLoader(PterygiumDataset(train_imgs, train_labels), batch_size=4, shuffle=True)
    train_classifier(clf_model, clf_loader, nn.CrossEntropyLoss(), torch.optim.Adam(clf_model.parameters(), 1e-4),
                     device, 'best_models/best_cls.pth')

if __name__ == '__main__':
    main()
