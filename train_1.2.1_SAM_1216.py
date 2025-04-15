import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models, transforms

# ---------------------------
# 数据集定义
# ---------------------------
class PterygiumDataset(Dataset):
    def __init__(self, img_paths, labels, masks=None, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        label = self.labels[idx]
        if self.masks and self.masks[idx] is not None and os.path.exists(self.masks[idx]):
            mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256))
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros((256, 256), dtype=np.float32)

        area = np.sum(mask) / (256 * 256)
        cx, cy = 0, 0
        if area > 0:
            indices = np.argwhere(mask > 0)
            cy, cx = np.mean(indices, axis=0)
            cx /= 256.0
            cy /= 256.0

        meta_feat = np.array([area, cx, cy], dtype=np.float32)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image), torch.tensor(mask), torch.tensor(label), torch.tensor(meta_feat)


# ---------------------------
# 数据加载函数
# ---------------------------
def load_dataset(data_dir):
    img_paths, mask_paths, labels = [], [], []
    for cls in ['0', '1', '2']:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.exists(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.endswith('.png') and '_label' not in fname:
                img_path = os.path.join(cls_path, fname)
                img_paths.append(img_path)
                labels.append(int(cls))
                if cls == '0':
                    mask_paths.append(None)
                else:
                    base = fname.split('.')[0]
                    mask_paths.append(os.path.join(cls_path, f"{base}_label.png"))
    return train_test_split(img_paths, labels, mask_paths, test_size=0.2, random_state=42)


# ---------------------------
# Dice系数和损失函数
# ---------------------------
def dice_coef(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def compute_loss(pred_cls, true_cls, pred_mask, true_mask):
    pred_mask = F.interpolate(pred_mask, size=true_mask.shape[-2:], mode='bilinear', align_corners=False)
    cls_loss = F.cross_entropy(pred_cls, true_cls)
    dice_loss = 1 - dice_coef(pred_mask, true_mask)
    return cls_loss + 0.5 * dice_loss


# ---------------------------
# 模型定义
# ---------------------------
class PterygiumNet(nn.Module):
    def __init__(self, num_classes=3):
        super(PterygiumNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.class_head = nn.Sequential(
            nn.Linear(512 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, meta_features=None):
        features = self.encoder(x)
        seg = torch.sigmoid(self.seg_head(features))
        pooled = F.adaptive_avg_pool2d(features, 1).view(x.size(0), -1)
        if meta_features is not None:
            pooled = torch.cat([pooled, meta_features], dim=1)
        cls_out = self.class_head(pooled)
        return cls_out, seg


# ---------------------------
# 训练函数
# ---------------------------
def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks, labels, metas in progress_bar:
        images, masks = images.to(device), masks.to(device)
        labels, metas = labels.to(device), metas.to(device)

        optimizer.zero_grad()
        pred_cls, pred_mask = model(images, metas)
        loss = compute_loss(pred_cls, labels, pred_mask, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    scheduler.step()  # 学习率调度器步进
    return total_loss / len(dataloader)


# ---------------------------
# 验证函数
# ---------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    dice_scores = []
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, masks, labels, metas in progress_bar:
            images, masks = images.to(device), masks.to(device)
            labels, metas = labels.to(device), metas.to(device)
            masks = masks.unsqueeze(1)

            pred_cls, pred_mask = model(images, metas)
            pred_mask = F.interpolate(pred_mask, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            pred_mask = pred_mask.squeeze(1)

            preds = torch.argmax(pred_cls, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            dice = dice_coef(pred_mask, masks)
            dice_scores.append(dice.item())

    acc = correct / total
    avg_dice = sum(dice_scores) / len(dice_scores)
    return acc, avg_dice

# ---------------------------
# 图形化
# ---------------------------
def plot_curves(train_losses, val_accs, val_dices, save_path="training_curves.png"):
    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(val_accs, label='Validation Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)
    plt.legend()

    # Dice Coefficient
    plt.subplot(1, 3, 3)
    plt.plot(val_dices, label='Validation Dice', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Validation Dice")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ---------------------------
# 主函数
# ---------------------------
if __name__ == "__main__":
    data_dir = os.path.join("..", "data", "train")
    train_imgs, val_imgs, train_labels, val_labels, train_masks, val_masks = load_dataset(data_dir)

    train_dataset = PterygiumDataset(train_imgs, train_labels, train_masks)
    val_dataset = PterygiumDataset(val_imgs, val_labels, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PterygiumNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每10轮学习率减半

    num_epochs = 30
    train_losses, val_accs, val_dices = [], [], []

    best_acc = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_model(model, train_loader, optimizer, scheduler, device)
        val_acc, val_dice = evaluate_model(model, val_loader, device)

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_dices.append(val_dice)

        print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Dice: {val_dice:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "../saved_models/pterygium_model_1.2.1_SAM_1216.pth")
            print("✅ Saved best model!")

    # 绘图保存曲线
    plot_curves(train_losses, val_accs, val_dices)
