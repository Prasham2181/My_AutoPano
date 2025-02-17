
# # Code starts here:

from httpx import patch
import numpy as np
import cv2
import glob
import os

# Add any python libraries here

import os
import glob
import cv2
import numpy as np
import shutil
import torch
from Network.Network import HomographyNet,Unsupervised_HomographyNet  
import matplotlib.pyplot as plt


def load_model(model,checkpoint_path):
    # Load the trained model
    # model = HomographyNet()
    
    # Load the checkpoint (fix the issue of 'weights_only=True', as it is not a valid parameter for torch.load)
    # checkpoint = torch.load(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  

    print("Model loaded successfully.")

    # Print Model Parameters Just to check the output for the checkpoint is not none, No other significance so do not uncomment if not debugging 
    # print("\n  Model Parameters:\n")are 
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Shape: {param.shape}")
    #     # print(param)  
    return model

# load_model()


def compute_h4pt(model,patch_A,patch_B):
    # patch_A = cv2.imread(patch_A)
    # patch_B = cv2.imread(patch_B)
    if patch_A is None or patch_B is None:
        print(" Image nai mil rahi! wapis check karo")
        return None
    patch_A = torch.tensor(patch_A, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, 128, 128)
    patch_B = torch.tensor(patch_B, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, 128, 128)
    stacked_patches = torch.cat((patch_A,patch_B),dim=1)
    print(f"Stacked patches are: {stacked_patches}")
    print(stacked_patches.shape)
    with torch.no_grad():
        predicted_H4Pt = model(stacked_patches).squeeze().cpu().numpy()
        # predicted_H4Pt = model(stacked_patches)
        print(f"predicted_H4Pt: {predicted_H4Pt}")
        predicted_H4Pt = (predicted_H4Pt * 64) - 32
        print(f"Scaled predicted_H4Pt: {predicted_H4Pt}")

    return predicted_H4Pt
     




def compute_H_pred_DLT(predicted_H4Pt, patch_size=128):
    """
    Computes Homography Matrix H using Direct Linear Transformation (DLT)
    from predicted H4Pt.
    """

    # Step 1: Define Original Corner Points (Patch A)
    corners_A = np.array([
        [0, 0],  # Top-left
        [patch_size - 1, 0],  # Top-right
        [0, patch_size - 1],  # Bottom-left
        [patch_size - 1, patch_size - 1]  # Bottom-right
    ], dtype=np.float32)

    corners_B = corners_A + predicted_H4Pt.reshape(4, 2)  # Apply predicted displacement
    print(f"Computed corners_B Matrix:\n{corners_B}")

    A = []
    for i in range(4):  # Iterate through 4 corresponding points
        x, y = corners_A[i]
        x_prime, y_prime = corners_B[i]

        A.append([
            -x, -y, -1,  0,  0,  0,  x * x_prime, y * x_prime, x_prime
        ])
        A.append([
            0,  0,  0, -x, -y, -1,  x * y_prime, y * y_prime, y_prime
        ])
    
    A = np.array(A, dtype=np.float32)

    U, S, Vt = np.linalg.svd(A)  # SVD ecomposition
    H = Vt[-1, :].reshape(3, 3)  # 

    # Normalize H to make H[2,2] = 1
    H_pred = H / H[2, 2]

    print(f"Computed Homography Matrix using DLT:\n{H}")

    return H_pred



def extract_Test_patch(img_path, patch_size=128, rho=32,return_coords=True):
  
    image = cv2.imread(img_path)
    
    if image is None:
        print(f" Error: Unable to load image {img_path}")
        return None, None, None

    height, width, _ = image.shape  

    if height < patch_size + 2 * rho or width < patch_size + 2 * rho:
        print(f" Skipping {img_path} (too small: {width}x{height})")
        return None, None, None

    # Select a random patch position within safe bounds
    x = np.random.randint(rho, width - patch_size - rho)
    y = np.random.randint(rho, height - patch_size - rho)

    # Extract the original patch (patch_A)
    patch_A = image[y:y+patch_size, x:x+patch_size]

    # Define original corner points
    corners_A = np.array([
        [x, y],
        [x + patch_size, y],
        [x, y + patch_size],
        [x + patch_size, y + patch_size]
    ], dtype=np.float32)

    t_x, t_y = np.random.randint(-rho//2, rho//2, size=(2,))
    t_xy = np.tile(np.array([t_x, t_y], dtype=np.float32), (4, 1))
    corners_B = corners_A + np.random.randint(-rho, rho, size=(4, 2)).astype(np.float32) + t_xy

    H_A_to_B = cv2.getPerspectiveTransform(corners_A, corners_B)
    
    # Apply inverse transformation to generate warped image
    warped_img = cv2.warpPerspective(image, np.linalg.inv(H_A_to_B), (width, height), 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Extract warped patch (patch_B)
    patch_B = warped_img[y:y+patch_size, x:x+patch_size]

    # Compute displacement H4Pt = corners_B - corners_A
    H4Pt_gt = corners_B - corners_A
    # print(f" Extracted patch from ({x}, {y}) to ({x+patch_size}, {y+patch_size})")
    if return_coords:
        return patch_A, patch_B, H4Pt_gt, (x, y)
    return patch_A, patch_B, H4Pt_gt

def visualize_h4pt_comparison(image, patch_A, predicted_H4Pt, H4Pt_gt, patch_coords, patch_size=128):
    
    # Ensure predicted_H4Pt is in the correct shape (4,2)
    if predicted_H4Pt.shape == (8,):
        predicted_H4Pt = predicted_H4Pt.reshape(4, 2)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(121)
    full_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(full_img_rgb)
    
    x, y = patch_coords
    
    corners_A = np.array([
        [x, y],
        [x + patch_size - 1, y],
        [x + patch_size - 1, y + patch_size - 1],
        [x, y + patch_size - 1]
    ], dtype=np.float32)
    
    corners_pred = corners_A + predicted_H4Pt
    corners_gt = corners_A + H4Pt_gt
    
    # Draw original patch (blue)
    plt.plot(corners_A[[0,1,2,3,0], 0], corners_A[[0,1,2,3,0], 1], 
             'b-', label='Original', linewidth=2)
    
    # Draw predicted transformation (red)
    plt.plot(corners_pred[[0,1,2,3,0], 0], corners_pred[[0,1,2,3,0], 1], 
             'r-', label='Predicted', linewidth=2)
    
    # Draw ground truth transformation (green)
    plt.plot(corners_gt[[0,1,2,3,0], 0], corners_gt[[0,1,2,3,0], 1], 
             'g-', label='Ground Truth', linewidth=2)
    
    plt.title('Full Image with Homography Visualization')
    plt.legend(loc='upper right')
    plt.axis('on')
    
    # Plot 2: Zoomed in view of the patch region
    plt.subplot(122)
    
    # Calculate the bounding box that encompasses all transformations
    all_points = np.vstack([corners_A, corners_pred, corners_gt])
    min_x, min_y = np.min(all_points, axis=0) - 20
    max_x, max_y = np.max(all_points, axis=0) + 20
    
    # Show zoomed region
    plt.imshow(full_img_rgb)
    plt.xlim(min_x, max_x)
    plt.ylim(max_y, min_y)  # Reverse y-axis for proper image coordinates
    
    # Draw boxes in zoomed view
    plt.plot(corners_A[[0,1,2,3,0], 0], corners_A[[0,1,2,3,0], 1], 
             'b-', label='Original', linewidth=2)
    plt.plot(corners_pred[[0,1,2,3,0], 0], corners_pred[[0,1,2,3,0], 1], 
             'r-', label='Predicted', linewidth=2)
    plt.plot(corners_gt[[0,1,2,3,0], 0], corners_gt[[0,1,2,3,0], 1], 
             'g-', label='Ground Truth', linewidth=2)
    
    plt.title('Zoomed View of Transformation')
    plt.legend(loc='upper right')
    plt.axis('on')
    plt.grid(True)
    
    mean_error = np.mean(np.linalg.norm(predicted_H4Pt - H4Pt_gt, axis=1))
    plt.text(0.02, 0.98, f'Mean Error: {mean_error:.2f} pixels', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

    return mean_error
def resize_images(image_path1, image_path2):
    
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # Check if images were successfully loaded
    if img1 is None:
        print(f"Error: Unable to load image from {image_path1}")
        return None, None
    if img2 is None:
        print(f"Error: Unable to load image from {image_path2}")
        return None, None
    
    # Resize images to 128x128 pixels using linear interpolation
    resized_img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_CUBIC)
    resized_img2 = cv2.resize(img2, (128, 128), interpolation=cv2.INTER_CUBIC)
    
    # Display the first resized image in a window with a given window name
   
    return resized_img1, resized_img2

# Example usage:
# path1 = "path/to/first_image.jpg"
# path2 = "path/to/second_image.jpg"
# img1_resized, img2_resized = resize_images(path1, path2)


def main():
    # img_path = r"YourDirectoryID_p1\Phase2\Data\Train\Train\1.jpg"
    img_path = r"YourDirectoryID_p1\Phase2\P1Ph2TestSet\Phase2\25.jpg"
    # img_path = r"YourDirectoryID_p1\Phase2\Data\Val\Val\1.jpg"
    image_1_path= r"YourDirectoryID_p1\Phase1\Data\Train\Set1\1.jpg"
    image_2_path= r"YourDirectoryID_p1\Phase1\Data\Train\Set1\2.jpg"
   

    image = cv2.imread(img_path)
    
    patch_A, patch_B, H4Pt_gt, patch_coords = extract_Test_patch(img_path, patch_size=128, rho=32, return_coords=True)
    
    # checkpoint_path = r"YourDirectoryID_p1\Phase2\Code\Checkpoints_p1_Phase2\49amodel_super.ckpt"
    checkpoint_path = r"YourDirectoryID_p1\Phase2\Code\Checkpoints_p1_Phase2\74amodel.ckpt"
    model_sup = HomographyNet()

    model = load_model(model_sup,checkpoint_path)
    img1_resized, img2_resized = resize_images(image_1_path, image_2_path)

    predict_H4Pt = compute_h4pt(model, patch_A, patch_B)
    predict_H_pred_DLT = compute_H_pred_DLT(predict_H4Pt, patch_size=128)
    mean_error = visualize_h4pt_comparison(image, patch_A, predict_H4Pt, H4Pt_gt, patch_coords)
    print(f"Mean error between predicted and ground truth H4Pt: {mean_error:.2f} pixels")
    
if __name__ == "__main__":
    main()
