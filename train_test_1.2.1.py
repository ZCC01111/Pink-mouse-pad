# import sys
# import torch
# import cv2
# import numpy as np
# # from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QPixmap
# # from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout
# from PyQt5.QtGui import QPixmap, QImage
#
# from PyQt5.QtGui import QImage, QPixmap
#
# from PyQt5.QtCore import Qt
#
# # 假设你有一个自定义的模型类
# class YourModel(torch.nn.Module):
#     def __init__(self):
#         super(YourModel, self).__init__()
#         # TODO: 定义你的模型结构
#
#     def forward(self, x):
#         # TODO: 前向推理逻辑
#         return x
#
# # 图像转tensor
# def preprocess(image):
#     img = cv2.resize(image, (256, 256))  # 与训练时保持一致
#     img = img / 255.0
#     img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
#     return img
#
# # 分割输出为掩码图
# def postprocess(mask):
#     mask = torch.sigmoid(mask).squeeze().detach().cpu().numpy()
#     mask_bin = (mask > 0.5).astype(np.uint8)  # 二值化
#
#     # 创建三通道图
#     seg_image = np.zeros((mask_bin.shape[0], mask_bin.shape[1], 3), dtype=np.uint8)
#     seg_image[:, :, 0] = mask_bin * 128  # R通道表示翼状胬肉
#     return seg_image
#
# class SegmentationApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("翼状胬肉分割演示")
#
#         self.image_label = QLabel("原图")
#         self.mask_label = QLabel("分割结果")
#
#         self.btn_load = QPushButton("加载图片")
#         self.btn_load.clicked.connect(self.load_image)
#
#         hbox = QHBoxLayout()
#         hbox.addWidget(self.image_label)
#         hbox.addWidget(self.mask_label)
#
#         vbox = QVBoxLayout()
#         vbox.addLayout(hbox)
#         vbox.addWidget(self.btn_load)
#
#         self.setLayout(vbox)
#
#         self.model = YourModel()
#         self.model.load_state_dict(torch.load("../saved_models/pterygium_model_1.2.1_SAM_1216.pth", map_location="cpu"))
#         self.model.eval()
#
#     def load_image(self):
#         path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
#         if not path:
#             return
#
#         image = cv2.imread(path)
#         self.display_image(image, self.image_label)
#
#         # 推理
#         input_tensor = preprocess(image)
#         with torch.no_grad():
#             output = self.model(input_tensor)
#
#         mask = postprocess(output)
#
#         self.display_image(mask, self.mask_label)
#
#     def display_image(self, img, label):
#         if len(img.shape) == 2:
#             h, w = img.shape
#             qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             h, w, ch = img.shape
#             bytes_per_line = ch * w
#             qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
#
#         pixmap = QPixmap.fromImage(qimg).scaled(256, 256, Qt.KeepAspectRatio)
#         label.setPixmap(pixmap)
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = SegmentationApp()
#     win.show()
#     sys.exit(app.exec_())



# import sys
# import os
# import torch
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QLabel, QWidget, QPushButton, QFileDialog,
#     QVBoxLayout, QHBoxLayout
# )
# from PyQt5.QtGui import QPixmap, QImage
# from PyQt5.QtCore import Qt
# import torch.nn.functional as F
# from torchvision import models
# import torch.nn as nn
#
# # 模型结构（和训练中保持一致）
# class PterygiumNet(nn.Module):
#     def __init__(self, num_classes=3):
#         super(PterygiumNet, self).__init__()
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         self.encoder = nn.Sequential(*list(resnet.children())[:-2])
#         self.seg_head = nn.Sequential(
#             nn.Conv2d(512, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=1)
#         )
#         self.class_head = nn.Sequential(
#             nn.Linear(512 + 3, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x, meta_features=None):
#         features = self.encoder(x)
#         seg = torch.sigmoid(self.seg_head(features))
#         pooled = F.adaptive_avg_pool2d(features, 1).view(x.size(0), -1)
#         if meta_features is not None:
#             pooled = torch.cat([pooled, meta_features], dim=1)
#         cls_out = self.class_head(pooled)
#         return cls_out, seg
#
# # 图像预处理
# def preprocess(image):
#     img = cv2.resize(image, (256, 256))
#     img = img.astype(np.float32) / 255.0
#     area = 0
#     cx, cy = 0, 0
#     meta_feat = np.array([area, cx, cy], dtype=np.float32)
#
#     img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
#     meta_feat = torch.tensor(meta_feat).unsqueeze(0)  # (1, 3)
#     return img, meta_feat
#
# # 分割后处理
# def postprocess(mask):
#     mask = mask.squeeze().cpu().numpy()
#     if mask.ndim != 2:
#         print("Mask shape error:", mask.shape)
#     mask_bin = (mask > 0.5).astype(np.uint8)
#     seg_image = np.zeros((256, 256, 3), dtype=np.uint8)
#     seg_image[:, :, 0] = mask_bin * 128  # 红色通道标记
#     return seg_image
#
# # 图像转 QPixmap，确保内存连续
# def to_pixmap(img):
#     if len(img.shape) == 2:
#         img = np.ascontiguousarray(img)
#         h, w = img.shape
#         qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
#     else:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.ascontiguousarray(img)
#         h, w, ch = img.shape
#         bytes_per_line = ch * w
#         qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
#     return QPixmap.fromImage(qimg).scaled(256, 256, Qt.KeepAspectRatio)
#
# # PyQt5 主应用窗口
# class SegmentationApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("翼状胬肉分割与诊断系统")
#
#         # 自动选择设备
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # 加载模型并转到设备
#         self.model = PterygiumNet().to(self.device)
#         model_path = "../saved_models/pterygium_model_1.2.1_SAM_1216.pth"
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
#
#         # 界面组件
#         self.image_label = QLabel("原图")
#         self.result_label = QLabel("结果图 / 文本")
#         self.btn_load = QPushButton("加载图像")
#         self.btn_load.clicked.connect(self.load_images)
#
#         # 布局
#         hbox = QHBoxLayout()
#         hbox.addWidget(self.image_label)
#         hbox.addWidget(self.result_label)
#
#         vbox = QVBoxLayout()
#         vbox.addLayout(hbox)
#         vbox.addWidget(self.btn_load)
#         self.setLayout(vbox)
#
#     def load_images(self):
#         paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
#         if not paths:
#             return
#
#         for path in paths:
#             image = cv2.imread(path)
#             if image is None:
#                 print(f"无法读取图像: {path}")
#                 continue
#
#             input_tensor, meta_feat = preprocess(image)
#             input_tensor = input_tensor.to(self.device)
#             meta_feat = meta_feat.to(self.device)
#
#             with torch.no_grad():
#                 pred_cls, pred_mask = self.model(input_tensor, meta_feat)
#                 cls_pred = torch.argmax(pred_cls, dim=1).item()
#
#             if cls_pred == 0:
#                 self.image_label.setPixmap(to_pixmap(image))
#                 self.result_label.setText("诊断结果：正常")
#             else:
#                 seg_img = postprocess(pred_mask)
#                 self.image_label.setPixmap(to_pixmap(image))
#                 self.result_label.setPixmap(to_pixmap(seg_img))
#
# # 启动主程序
# if __name__ == "__main__":
#     try:
#         app = QApplication(sys.argv)
#         win = SegmentationApp()
#         win.show()
#         sys.exit(app.exec_())
#     except Exception as e:
#         print("程序异常终止：", e)


import sys
import os
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn


# 模型结构（和训练中保持一致）
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


# 图像预处理
def preprocess(image):
    img = cv2.resize(image, (256, 256))
    img = img.astype(np.float32) / 255.0
    area = 0
    cx, cy = 0, 0
    meta_feat = np.array([area, cx, cy], dtype=np.float32)

    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
    meta_feat = torch.tensor(meta_feat).unsqueeze(0)  # (1, 3)
    return img, meta_feat


# 分割后处理
def postprocess(mask):
    mask = mask.squeeze().cpu().numpy()
    if mask.ndim != 2:
        print("Mask shape error:", mask.shape)
    mask_bin = (mask > 0.5).astype(np.uint8)
    seg_image = np.zeros((256, 256, 3), dtype=np.uint8)
    seg_image[:, :, 0] = mask_bin * 128  # 红色通道标记
    return seg_image


# 图像转 QPixmap，确保内存连续
def to_pixmap(img):
    if len(img.shape) == 2:
        img = np.ascontiguousarray(img)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(256, 256, Qt.KeepAspectRatio)


# 处理图像的线程
class ImageProcessingThread(QThread):
    update_image = pyqtSignal(QPixmap)
    update_text = pyqtSignal(str)
    update_result = pyqtSignal(QPixmap)

    def __init__(self, image_path, model, device):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.device = device

    def run(self):
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError("无法读取图像")

            # 图像预处理
            input_tensor, meta_feat = preprocess(image)
            input_tensor = input_tensor.to(self.device)
            meta_feat = meta_feat.to(self.device)

            # 模型推理
            with torch.no_grad():
                pred_cls, pred_mask = self.model(input_tensor, meta_feat)
                cls_pred = torch.argmax(pred_cls, dim=1).item()

            # 结果处理
            if cls_pred == 0:
                self.update_text.emit("诊断结果：正常")
                self.update_image.emit(to_pixmap(image))
            else:
                seg_img = postprocess(pred_mask)
                self.update_result.emit(to_pixmap(seg_img))
        except Exception as e:
            self.update_text.emit(f"发生错误: {str(e)}")


# PyQt5 主应用窗口
class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("翼状胬肉分割与诊断系统")

        # 自动选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型并转到设备
        self.model = PterygiumNet().to(self.device)
        model_path = "../saved_models/pterygium_model_1.2.1_SAM_1216.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 界面组件
        self.image_label = QLabel("原图")
        self.result_label = QLabel("结果图 / 文本")
        self.btn_load = QPushButton("加载图像")
        self.btn_load.clicked.connect(self.load_images)

        # 布局
        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)
        hbox.addWidget(self.result_label)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.btn_load)
        self.setLayout(vbox)

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
        if not paths:
            return

        for path in paths:
            self.result_label.setText("处理中...")
            self.thread = ImageProcessingThread(path, self.model, self.device)
            self.thread.update_image.connect(self.update_image)
            self.thread.update_text.connect(self.update_text)
            self.thread.update_result.connect(self.update_result)
            self.thread.start()

    def update_image(self, pixmap):
        self.image_label.setPixmap(pixmap)

    def update_text(self, text):
        self.result_label.setText(text)

    def update_result(self, pixmap):
        self.result_label.setPixmap(pixmap)


# 启动主程序
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        win = SegmentationApp()
        win.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("程序异常终止：", e)
