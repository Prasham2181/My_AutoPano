# # !/usr/bin/evn python

# """
# RBE/CS Fall 2022: Classical and Deep Learning Approaches for
# Geometric Computer Vision
# Project 1: MyAutoPano: Phase 2 Starter Code


# Author(s):
# Lening Li (lli4@wpi.edu)
# Teaching Assistant in Robotics Engineering,
# Worcester Polytechnic Institute
# """



import os
import cv2
import torch
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
from Network.Network import HomographyNet, Unsupervised_HomographyNet
from predict_supervised import (
    load_model, compute_h4pt, compute_H_pred_DLT, extract_Test_patch, resize_images,visualize_h4pt_comparison
)
from Predict_Unsupervised import create_panorama  # Unsupervised function for blending

# -------------------- PATCH EXTRACTION FUNCTION --------------------
def extract_patch():
    """Extracts patches for training and testing."""
    patch_size = 128
    rho = 32
    num_patches_per_image = 3
    test_sample_size = 500

    base_dir = "YourDirectoryID_p1/Phase2/Data"
    train_input_folder = os.path.join(base_dir, "Train/Train")
    val_input_folder = os.path.join(base_dir, "Val/Val")
    test_input_folder = os.path.join(base_dir, "Test/Test")
    phase2_test_input_folder = os.path.join(base_dir, "Phase2_TestSet")

    def get_patch_dirs(prefix):
        return {
            "A": os.path.join(base_dir, f"Patch_A_{prefix}_images_FLD"),
            "B": os.path.join(base_dir, f"Patch_B_{prefix}_images_FLD"),
            "labels": os.path.join(base_dir, f"{prefix}_Labels"),
            "stacked": os.path.join(base_dir, f"{prefix}Data_Spatch"),
        }

    train_dirs = get_patch_dirs("Train")
    val_dirs = get_patch_dirs("Validation")
    test_dirs = get_patch_dirs("Test")
    phase2_test_dirs = get_patch_dirs("Phase2Test")

    for dirs in [train_dirs, val_dirs, test_dirs, phase2_test_dirs]:
        for folder in dirs.values():
            os.makedirs(folder, exist_ok=True)
    os.makedirs(test_input_folder, exist_ok=True)

    # Move random training images to test folder
    train_images = glob.glob(os.path.join(train_input_folder, "*.jpg"))
    np.random.shuffle(train_images)
    test_images = train_images[:test_sample_size]

    for img_path in test_images:
        shutil.move(img_path, test_input_folder)
    print(f"Moved {test_sample_size} images to test folder.")

# -------------------- IMAGE RESIZING FUNCTION -------------------- #ONLY TRIED FOR PANO GENERATION
def resize_images_in_folder(input_folder, output_folder, target_size=(128, 128)):
    """Resizes all images in a folder."""
    os.makedirs(output_folder, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")), key=lambda x: os.path.basename(x))

    if not image_paths:
        print(f"No JPEG images found in {input_folder}.")
        return

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            continue
        
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, resized_img)
        print(f"Resized image saved: {save_path}")

# -------------------- SUPERVISED PANORAMA FUNCTION --------------------
def run_supervised():
    """Runs the supervised homography estimation and visualization."""
    print("\nRunning Supervised Panorama Stitching...\n")

    image_path = r"YourDirectoryID_p1\Phase2\P1Ph2TestSet\Phase2\25.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image file not found -> {image_path}")
        return

    checkpoint_path = r"YourDirectoryID_p1\Phase2\Code\Checkpoints_p1_Phase2\74amodel.ckpt"
    model_sup = HomographyNet()
    model = load_model(model_sup, checkpoint_path)

    result = extract_Test_patch(image_path, patch_size=128, rho=32, return_coords=True)
    if result is None or len(result) < 4:
        print("Error: Patch extraction failed.")
        return
    patch_A, patch_B, H4Pt_gt, patch_coords = result

    patch_A_tensor = torch.tensor(patch_A, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    patch_B_tensor = torch.tensor(patch_B, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    x_input = torch.cat((patch_A_tensor, patch_B_tensor), dim=1)

    with torch.no_grad():
        predicted_H4Pt = model(x_input).squeeze().cpu().numpy()
    
    predicted_H4Pt = (predicted_H4Pt * 64) - 32
    print(f"Predicted H4Pt: {predicted_H4Pt}")

    # Compute the homography matrix using DLT
    H_pred = compute_H_pred_DLT(predicted_H4Pt, patch_size=128)

    # Visualize comparison of predicted vs. ground truth homography
    mean_error = visualize_h4pt_comparison(cv2.imread(image_path), patch_A, predicted_H4Pt, H4Pt_gt, patch_coords)

    print(f"Mean Error Between Predicted and Ground Truth H4Pt: {mean_error:.2f} pixels")

    plt.show()


# -------------------- UNSUPERVISED PANORAMA FUNCTION --------------------
def run_unsupervised():
    """Runs the unsupervised homography estimation."""
    print("\nRunning Unsupervised Panorama Stitching...\n")

    image_path = r"YourDirectoryID_p1\Phase2\P1Ph2TestSet\Phase2\1.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image file not found -> {image_path}")
        return

    checkpoint_path = r"YourDirectoryID_p1\Phase2\Code\Checkpoints_p1_Phase2\8amodel_unsupervised_Updated.ckpt"
    model_unsup = Unsupervised_HomographyNet()
    model = load_model(model_unsup, checkpoint_path)

    result = extract_Test_patch(image_path, patch_size=128, rho=32, return_coords=True)
    if result is None or len(result) < 4:
        print("Error: Patch extraction failed.")
        return
    patch_A, patch_B, _, patch_coords = result

    # Convert patches to tensors (normalize to [0,1])
    patch_A_tensor = torch.tensor(patch_A, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    patch_B_tensor = torch.tensor(patch_B, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    x_input = torch.cat((patch_A_tensor, patch_B_tensor), dim=1)

    x0, y0 = patch_coords
    corners_A = np.array([
        [x0, y0],
        [x0 + 128 - 1, y0],
        [x0 + 128 - 1, y0 + 128 - 1],
        [x0, y0 + 128 - 1]
    ], dtype=np.float32)
    C_A = torch.tensor(corners_A, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        warped_patch = model(x_input, C_A)  
    warped_patch_np = warped_patch.squeeze().permute(1, 2, 0).cpu().numpy()
    warped_patch_np = (warped_patch_np * 255).clip(0, 255).astype(np.uint8)

    # Create and visualize panorama
    panorama = create_panorama(warped_patch_np, patch_B, overlap=64)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(patch_A, cv2.COLOR_BGR2RGB))
    plt.title("Original Patch A")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title("Stitched Panorama")
    plt.axis("off")

    plt.show()

# -------------------- MAIN FUNCTION  --------------------
def main():
    """Runs the extraction and the selected mode automatically."""
    # extract_patch()  # Extracts patches

    mode = "supervised"  # Change to "unsupervised" if needed

    if mode == "supervised":
        run_supervised()
    elif mode == "unsupervised":
        run_unsupervised()
    else:
        print("Invalid mode selected!")

if __name__ == "__main__":
    main()
