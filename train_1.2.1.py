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
    """
    根据数据集目录中的图像和（可选）掩码，加载用于分割任务的图片、标签和区域元特征。
    对于正常(0)类别，掩码不存在时自动生成全零掩码。
    """
    def __init__(self, img_paths, labels, masks=None, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.masks = masks  # 对于0类，此处为 None
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取图像，并预处理resize到256x256
        image = cv2.imread(self.img_paths[idx])
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0   # 归一化
        image = np.transpose(image, (2, 0, 1))  # 转换为CHW

        label = self.labels[idx]
        # 如果有对应的分割掩码，则读取，否则默认全黑
        if self.masks and self.masks[idx] is not None:
            mask_path = self.masks[idx]
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (256, 256))
                # 二值化：假设掩码中大于0为目标区域
                mask = (mask > 0).astype(np.float32)
            else:
                mask = np.zeros((256, 256), dtype=np.float32)
        else:
            mask = np.zeros((256, 256), dtype=np.float32)

        # 计算翼状胬肉区域的元特征：区域面积占比以及质心坐标归一化值
        area = np.sum(mask) / (256 * 256)
        cx, cy = 0, 0
        if area > 0:
            indices = np.argwhere(mask > 0)
            cy, cx = np.mean(indices, axis=0)  # [row, col]
            cx /= 256.0
            cy /= 256.0

        meta_feat = np.array([area, cx, cy], dtype=np.float32)

        # 将mask加上channel维度
        mask = np.expand_dims(mask, axis=0)
        # 如果有自定义transform，可以在这里做进一步处理
        if self.transform:
            image = self.transform(image)

        return torch.tensor(image), torch.tensor(mask), torch.tensor(label), torch.tensor(meta_feat)


# ---------------------------
# 数据加载函数
# ---------------------------
def load_dataset(data_dir):
    """
    从 data/train 文件夹中读取图像和对应的掩码信息，
    子文件夹"0"、"1"、"2"分别对应正常、建议观察、建议手术。
    返回划分后的训练集与验证集的文件路径列表、标签、掩码路径。
    """
    img_paths, mask_paths, labels = [], [], []
    for cls in ['0', '1', '2']:
        cls_path = os.path.join(data_dir, cls)
        print(f"Looking into: {cls_path}")
        if not os.path.exists(cls_path):
            print(f"⚠️ Warning: Folder not found: {cls_path}")
            continue

        for fname in os.listdir(cls_path):
            # 避免处理掩码文件（文件名中包含 '_label'）
            if fname.endswith('.png') and '_label' not in fname:
                img_path = os.path.join(cls_path, fname)
                img_paths.append(img_path)
                labels.append(int(cls))

                # 对于类别 "0" 没有掩码
                if cls == '0':
                    mask_paths.append(None)
                else:
                    base = fname.split('.')[0]
                    mask_path = os.path.join(cls_path, f"{base}_label.png")
                    mask_paths.append(mask_path)

    print(f"Total samples: {len(img_paths)}")
    return train_test_split(img_paths, labels, mask_paths, test_size=0.2, random_state=42)


# ---------------------------
# Dice系数和损失函数
# ---------------------------
def dice_coef(pred, target, eps=1e-6):
    """
    计算Dice系数，要求pred和target的形状均为 [batch_size, height * width]。
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def compute_loss(pred_cls, true_cls, pred_mask, true_mask):
    """
    计算组合损失，其中包含分类交叉熵损失以及Dice损失
    """
    # 调整预测的掩码大小以匹配真实掩码
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
        # 使用更新后的ResNet18加载预训练权重
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        # 分割头：利用卷积输出1通道分割图
        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        # 分类头（可选）：利用全局池化和元特征进行诊断分类
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
def train_model(model, dataloader, optimizer, device):
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
    return total_loss / len(dataloader)


# ---------------------------
# 验证函数
# ---------------------------
# def evaluate_model(model, dataloader, device):
#     model.eval()
#     correct = 0
#     total = 0
#     dice_scores = []
#     progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
#     with torch.no_grad():
#         for images, masks, labels, metas in progress_bar:
#             images, masks = images.to(device), masks.to(device)
#             labels, metas = labels.to(device), metas.to(device)
#             masks = masks.unsqueeze(1)  # 转为 (batch_size, 1, height, width)
#
#             pred_cls, pred_mask = model(images, metas)
#             # 调整预测掩码: (batch_size, 1, height, width) -> (batch_size, height, width)
#             pred_mask = pred_mask.squeeze(1)
#
#             preds = torch.argmax(pred_cls, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#             # 计算Dice系数
#             dice = dice_coef(pred_mask, masks)
#             dice_scores.append(dice.item())
#
#     acc = correct / total
#     avg_dice = sum(dice_scores) / len(dice_scores)
#     return acc, avg_dice
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

            # 将 masks 转换为四维张量（batch_size, 1, height, width）
            masks = masks.unsqueeze(1)  # shape: (B,1,256,256)

            pred_cls, pred_mask = model(images, metas)
            # 对预测掩码上采样，使其与真实掩码一致
            pred_mask = F.interpolate(pred_mask, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            # 转换预测掩码为 (B, H, W)
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
# 保存分割结果函数
# ---------------------------
def save_segmentation_results(model, dataloader, device, result_folder="Segmentation_Results"):
    """
    对dataloader中的每个样本进行预测，将分割结果转换为3通道图像：
      - R通道像素值为128代表翼状胬肉区域（预测概率>阈值），其他区域为0；
      - G、B通道均为0；
    然后保存到result_folder中，文件名与输入图像前缀一致。
    """
    model.eval()
    os.makedirs(result_folder, exist_ok=True)
    threshold = 0.5  # 分割概率阈值
    with torch.no_grad():
        for images, masks, labels, metas in tqdm(dataloader, desc="Saving Results"):
            images = images.to(device)
            pred_cls, pred_mask = model(images, metas.to(device))
            # 调整分割结果大小至256x256（与输入一致）
            pred_mask = F.interpolate(pred_mask, size=(256,256), mode='bilinear', align_corners=False)
            pred_mask = pred_mask.squeeze(1)  # (batch_size, 256,256)
            pred_mask = (pred_mask > threshold).float()  # 二值化

            # 对每幅图片单独保存
            for i in range(images.size(0)):
                # 获取对应的输入文件名（假设 DataLoader.dataset.img_paths 可用）
                # 注意：这里需要确保 DataLoader 传入的数据集具有 .img_paths 属性
                input_path = dataloader.dataset.img_paths[i]
                # 文件名前缀
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                # 构造3通道分割结果图像：
                # R通道：若预测为翼状胬肉区域，则为128；否则为0。其他通道全0。
                seg = pred_mask[i].cpu().numpy()  # shape (256,256)
                seg_3ch = np.zeros((256, 256, 3), dtype=np.uint8)
                seg_3ch[..., 0] = (seg * 128).astype(np.uint8)
                # 保存为png文件
                cv2.imwrite(os.path.join(result_folder, f"{base_name}.png"), seg_3ch)


# ---------------------------
# 绘图函数
# ---------------------------
def plot_curves(train_losses, val_accs, val_dices):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_dices, label='Validation Dice', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Curve')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()


# ---------------------------
# 主函数
# ---------------------------
if __name__ == "__main__":
    data_dir = os.path.join("..", "data", "train")
    train_imgs, val_imgs, train_labels, val_labels, train_masks, val_masks = load_dataset(data_dir)

    train_dataset = PterygiumDataset(train_imgs, train_labels, train_masks)
    val_dataset = PterygiumDataset(val_imgs, val_labels, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)  # 多线程加载
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PterygiumNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 用于保存训练过程数据
    train_losses, val_accuracies, val_dices = [], [], []
    best_val_dice = 0.0
    best_model_path = os.path.join("..", "saved_models", "pterygium_model_1.2.1.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = train_model(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        val_acc, val_dice = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)
        val_dices.append(val_dice)

        print(f"Validation Accuracy: {val_acc:.4f}, Dice Coefficient: {val_dice:.4f}")

        # 保存最优模型（以Dice系数为指标）
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print("保存最佳模型！")

    plot_curves(train_losses, val_accuracies, val_dices)

    # 若需要对测试样本进行预测并保存分割结果，则加载测试数据集，此处以验证集为示例
    save_segmentation_results(model, val_loader, device, result_folder="Segmentation_Results")
