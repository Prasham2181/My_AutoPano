#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel, HomographyNet, LossFn_Sup, Unsupervised_HomographyNet, LossFn_UnSup
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

from CustomDataset import CustomDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)

        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        I1 = np.float32(cv2.imread(RandImageName))
        Coordinates = TrainCoordinates[RandIdx]

        # Append All Images and Mask
        I1Batch.append(torch.from_numpy(I1))
        CoordinatesBatch.append(torch.tensor(Coordinates))

    return torch.stack(I1Batch), torch.stack(CoordinatesBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
    train_loader,
    val_loader,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of data or for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    ModelType - Supervised or Unsupervised Model

    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """

    # Predict output with forward pass
    if ModelType == "Sup":
        model = HomographyNet().to(device)
        Loss_func = LossFn_Sup
    
    elif ModelType == "UnSup":
        model = Unsupervised_HomographyNet().to(device)
        Loss_func = LossFn_UnSup
    
    else:
        raise ValueError("Invalid ModelType! Choose 'Sup' or 'UnSup'.")

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    # Optimizer = AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    Optimizer = AdamW(model.parameters(), lr=1e-4)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")
    
    # train_loader = DataLoader(train_dataset, batch_size=MiniBatchSize, shuffle=True)

    for Epoch in tqdm(range(StartEpoch, NumEpochs)):
        model.train()

        total_train_loss = 0

        for I1Batch, CoordinatesBatch in tqdm(train_loader):

            I1Batch, CoordinatesBatch = I1Batch.to(device), CoordinatesBatch.to(device)

            if ModelType == "Sup":
                # Predict output with forward pass
                PredicatedCoordinatesBatch = model(I1Batch)
                LossThisBatch = Loss_func(PredicatedCoordinatesBatch, CoordinatesBatch)
            
            elif ModelType == "UnSup":
                # H_pred = model(I1Batch, CoordinatesBatch)  # Predicted homography
                # xa_warped = model.stn(I1Batch, H_pred)  # Apply STN warping
                xa_warped = model(I1Batch, CoordinatesBatch)
                LossThisBatch = Loss_func(xa_warped, I1Batch[:, 3:6, :, :])

            Optimizer.zero_grad()
            LossThisBatch.backward()
            
            Optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            total_train_loss += LossThisBatch.item()

        avg_train_loss = total_train_loss/len(train_loader)

        # print(CheckPointPath)

        SaveName = (
            CheckPointPath
            + str(Epoch)
            + "a"
            + "model.ckpt"
        )

        # print(SaveName)

        torch.save(
            {
                "epoch": Epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )

        # print("\n" + SaveName + " Model Saved...")

        # print("Nahi, Mein iddhar hoon")
        
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_I1Batch, val_coordinatesBatch in val_loader:

                val_I1Batch, val_coordinatesBatch = val_I1Batch.to(device), val_coordinatesBatch.to(device)

                if ModelType == "Sup":
                    # Predict output with forward pass
                    val_Predicted = model(val_I1Batch)
                    val_Loss = Loss_func(val_Predicted, val_coordinatesBatch)

                elif ModelType == "UnSup":
                    # H_pred = model(I1Batch, CoordinatesBatch)  # Predicted homography
                    # xa_warped = model.stn(I1Batch, H_pred)  # Apply STN warping
                    val_xa_warped = model(val_I1Batch, val_coordinatesBatch)
                    val_Loss = Loss_func(val_xa_warped, val_I1Batch[:, 3:6, :, :])

                total_val_loss += val_Loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        Writer.add_scalar("Training Loss", avg_train_loss, Epoch)
        Writer.add_scalar("Validation Loss", avg_val_loss, Epoch)


        
        print("\n" + SaveName + " Model Saved...")


        Writer.add_scalar("LossEveryIter", LossThisBatch.item(), Epoch)
        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()
    
    print("\nTraining Complete!")

def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="/home/smehta1/ComputerVision/MyAutoPano_Phase2/Data_UnSup/Unsupervised/TrainData_Spatch_unsup",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints_UnSup/",
        help="Path to save Checkpoints, Default: ../Checkpoints_UnSup/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=45,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="/home/smehta1/ComputerVision/MyAutoPano_Phase2/Code/Logs_tensorboard",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    if ModelType == "Sup":

        train_dataset_path = "/home/sarthak_m/ComputerVision/P1_MyAutoPano/YourDirectoryID_p1/Phase2/Data/TrainData_Spatch"
        train_dataset_label = "/home/sarthak_m/ComputerVision/P1_MyAutoPano/YourDirectoryID_p1/Phase2/Data/Train_Labels"

        val_dataset_path = "/home/sarthak_m/ComputerVision/P1_MyAutoPano/YourDirectoryID_p1/Phase2/Data/ValidationData_Spatch"
        val_dataset_label = "/home/sarthak_m/ComputerVision/P1_MyAutoPano/YourDirectoryID_p1/Phase2/Data/Val_Labels"
    
    elif ModelType == "UnSup":

        train_dataset_path = "/home/smehta1/ComputerVision/MyAutoPano_Phase2/Data_UnSup/Unsupervised/TrainData_Spatch_unsup"
        train_dataset_label = "/home/smehta1/ComputerVision/MyAutoPano_Phase2/Data_UnSup/Unsupervised/Train_Labels"

        val_dataset_path = "/home/smehta1/ComputerVision/MyAutoPano_Phase2/Data_UnSup/Unsupervised/ValidationData_Spatch_unsup"
        val_dataset_label = "/home/smehta1/ComputerVision/MyAutoPano_Phase2/Data_UnSup/Unsupervised/Val_Labels"

    train_dataset = CustomDataset(train_dataset_path, train_dataset_label)

    train_loader = DataLoader(train_dataset, batch_size=MiniBatchSize, shuffle=True)


    val_dataset = CustomDataset(val_dataset_path, val_dataset_label)

    val_loader = DataLoader(val_dataset, batch_size=MiniBatchSize, shuffle=True)


    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # print(SaveCheckPoint)
    # print(DirNamesTrain)
    
    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    main()
