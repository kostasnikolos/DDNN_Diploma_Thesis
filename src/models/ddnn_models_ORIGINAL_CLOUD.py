"""
BACKUP: ORIGINAL CloudCNN Architecture
========================================

This is the ORIGINAL CloudCNN before enhancement.
Use this to restore the original architecture if needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudCNN_Original(nn.Module):
    """
    ORIGINAL Cloud CNN Architecture (before deep enhancement)
    
    Architecture:
    - Input: (batch, 32, 16, 16) feature maps from LocalFeatureExtractor
    - Conv Block 1: 32 → 64 channels, BN, LeakyReLU, Dropout, Pool (→ 8x8)
    - Conv Block 2: 64 → 128 channels, BN, LeakyReLU, Dropout, Pool (→ 4x4)
    - Conv Block 3: 128 → 256 channels, BN, LeakyReLU, Dropout (no pool)
    - Conv Block 4: 256 → 256 channels, BN, LeakyReLU, Dropout, Pool (→ 2x2)
    - Flatten: (batch, 256*2*2) = (batch, 1024)
    - FC1: 1024 → 256 with LeakyReLU
    - FC2: 256 → 64 with LeakyReLU
    - FC3: 64 → NUM_CLASSES (logits)
    """
    
    def __init__(self, num_classes=10):
        super(CloudCNN_Original, self).__init__()
        
        # Convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(p=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """Forward pass through ORIGINAL cloud CNN."""
        # Conv block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        x = self.pool1(x)  # 16x16 → 8x8
        
        # Conv block 2
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        x = self.pool2(x)  # 8x8 → 4x4
        
        # Conv block 3
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = self.dropout3(x)
        
        # Conv block 4
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = self.dropout4(x)
        x = self.pool3(x)  # 4x4 → 2x2
        
        # Flatten and FC layers
        x = x.view(-1, 256 * 2 * 2)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.fc3(x)
        
        return x


# INSTRUCTIONS TO RESTORE ORIGINAL:
# 1. Copy the CloudCNN_Original class above
# 2. Rename it to CloudCNN
# 3. Replace the CloudCNN class in ddnn_models.py with this code
# 4. Or simply use search/replace to restore the original architecture
