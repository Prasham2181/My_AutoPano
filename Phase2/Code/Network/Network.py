"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import lightning as pl
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True

def LossFn_UnSup(predicted_warped_img, warped_img):
    ssim_loss = 1 - kornia.losses.ssim(predicted_warped_img, warped_img, window_size=11)
    return ssim_loss.mean()

def LossFn_Sup(predicted, ground_truth):
    return torch.norm(predicted - ground_truth, dim=1).mean()


class HomographyModel(pl.LightningModule):
    def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        self.hparams = hparams
        self.model = HomographyNet()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class HomographyNet(nn.Module):
    def __init__(self, InputSize=128, OutputSize=8):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        # self.batchNorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(16*16*128, 1024)   # 16 * 8 * 8
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 8)

    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################

        x = F.relu(self.conv1(x))
        x = self.bn1(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.bn2(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = self.bn3(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = self.bn4(self.conv8(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x


class Unsupervised_HomographyNet(nn.Module):
    
    def __init__(self, InputSize=128, OutputSize=8):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(Unsupervised_HomographyNet, self).__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        # self.batchNorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(16*16*128, 1024)   # 16 * 8 * 8
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 8)


        ############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        ############################

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), 
            nn.ReLU(True), 
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )


    ############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    ############################

    def stn(self, xa, H):
        B, C, H_img, W_img = xa.shape
        
        # Use larger epsilon for numerical stability
        eps = 1e-5
        
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H_img, device=xa.device),
            torch.linspace(-1, 1, W_img, device=xa.device),
            indexing='ij'
        )
        
        ones = torch.ones_like(x, device=xa.device)
        grid = torch.stack([x, y, ones], dim=-1).reshape(-1, 3)
        grid = grid.unsqueeze(0).repeat(B, 1, 1)
        
        # Use more stable inverse
        H_inv = torch.linalg.inv(H + eps * torch.eye(3, device=H.device).unsqueeze(0))
        
        grid = grid.transpose(1, 2)
        transformed_grid = torch.bmm(H_inv, grid)
        transformed_grid = transformed_grid.transpose(1, 2)
        
        # More stable normalization
        z = transformed_grid[..., 2:].abs() + eps
        transformed_grid = transformed_grid / z
        
        flow_grid = transformed_grid[..., :2].reshape(B, H_img, W_img, 2)
        
        # Add boundary handling
        flow_grid = torch.clamp(flow_grid, -1.1, 1.1)
        
        output = F.grid_sample(
            xa, 
            flow_grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        return output

    def TensorDLT(self, C_A, H4_pt):
        B, _ = H4_pt.shape
        
        # Reshape inputs
        H4_pt = H4_pt.view(B, 4, 2)
        C_A = C_A.view(B, 4, 2)
        
        # Calculate predicted corners
        predicted_C_B = C_A + H4_pt
        
        # Initialize tensor to store A matrices for all batches
        A = torch.zeros(B, 8, 9, device=H4_pt.device, dtype=H4_pt.dtype)
        b = torch.zeros(B, 8, device=H4_pt.device, dtype=H4_pt.dtype)
        
        # For each point in the batch
        for i in range(B):
            # For each corner point
            for j in range(4):
                ui, vi = C_A[i, j]  # Original points
                ui_prime, vi_prime = predicted_C_B[i, j]  # Predicted points
                
                # Fill rows for x coordinate equation
                A[i, j*2, 0] = 0
                A[i, j*2, 1] = 0
                A[i, j*2, 2] = 0
                A[i, j*2, 3] = -ui
                A[i, j*2, 4] = -vi
                A[i, j*2, 5] = -1
                A[i, j*2, 6] = vi_prime * ui
                A[i, j*2, 7] = vi_prime * vi
                A[i, j*2, 8] = vi_prime
                
                # Fill rows for y coordinate equation
                A[i, j*2+1, 0] = ui
                A[i, j*2+1, 1] = vi
                A[i, j*2+1, 2] = 1
                A[i, j*2+1, 3] = 0
                A[i, j*2+1, 4] = 0
                A[i, j*2+1, 5] = 0
                A[i, j*2+1, 6] = -ui_prime * ui
                A[i, j*2+1, 7] = -ui_prime * vi
                A[i, j*2+1, 8] = -ui_prime
                
                # Fill b vector
                b[i, j*2] = -vi_prime
                b[i, j*2+1] = ui_prime
        
        # Use PyTorch's pinverse to solve the system for each batch
        # H_flat = torch.bmm(torch.pinverse(A), b.unsqueeze(-1)).squeeze(-1)  # Shape: [B, 9]

        U, S, V = torch.svd(A)
        H_flat = V[:, :, -1]  # Last column of V

        
        # Reshape into 3x3 homography matrices
        H = H_flat.view(B, 3, 3)
        
        # Normalize the homography matrices
        # H = H / (H[:, 2:3, 2:3] + 1e-8)
        H = H / (torch.clamp(H[:, 2:3, 2:3], min=1e-3))

        
        return H


    def forward(self, x, C_A):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################

        B = x.size(0)

        xa = x[:, :3, :, :]  # First image: take first 3 channels
        xb = x[:, 3:, :, :]  # Second image: take last 3 channels

        x = F.relu(self.conv1(x))
        x = self.bn1(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.bn2(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = self.bn3(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = self.bn4(self.conv8(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        # Tensor DLT

        # print(x.shape)

        H_3 = self.TensorDLT(C_A, x)

        # Spatial Transform Network

        xa_warped = self.stn(xa, H_3)

        return xa_warped