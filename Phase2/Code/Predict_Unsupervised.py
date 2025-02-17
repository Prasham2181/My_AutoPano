

import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
from Network.Network import Unsupervised_HomographyNet  # Import your unsupervised model

def load_model(model, checkpoint_path):
    """
    Loads a PyTorch model from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")
    return model

def extract_Test_patch(img_path, patch_size=128, rho=32, return_coords=True):
    """
    Extracts a random patch from the input image and applies a random perspective
    transformation to generate a warped patch.

    Returns:
      patch_A: the original patch,
      patch_B: the warped (synthetically transformed) patch,
      H4Pt_gt: the ground truth 4-point displacement (unused by the unsupervised model),
      patch_coords: the (x,y) coordinates (upper-left) of the patch in the full image.
    """
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to load image {img_path}")
        return None, None, None, None

    height, width, _ = image.shape  
    if height < patch_size + 2 * rho or width < patch_size + 2 * rho:
        print(f"Skipping {img_path} (too small: {width}x{height})")
        return None, None, None, None

    # Randomly select a patch position within safe bounds.
    x = np.random.randint(rho, width - patch_size - rho)
    y = np.random.randint(rho, height - patch_size - rho)

    # Extract the original patch (patch_A)
    patch_A = image[y:y+patch_size, x:x+patch_size]

    # Define the original corner points of patch_A.
    corners_A = np.array([
        [x, y],
        [x + patch_size, y],
        [x, y + patch_size],
        [x + patch_size, y + patch_size]
    ], dtype=np.float32)

    # Create a perturbed version of the corners to generate patch_B.
    t_x, t_y = np.random.randint(-rho//2, rho//2, size=(2,))
    t_xy = np.tile(np.array([t_x, t_y], dtype=np.float32), (4, 1))
    corners_B = corners_A + np.random.randint(-rho, rho, size=(4, 2)).astype(np.float32) + t_xy

    # Compute the perspective transform and warp the image.
    H_A_to_B = cv2.getPerspectiveTransform(corners_A, corners_B)
    warped_img = cv2.warpPerspective(image, np.linalg.inv(H_A_to_B), (width, height), 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    patch_B = warped_img[y:y+patch_size, x:x+patch_size]
    H4Pt_gt = corners_B - corners_A

    if return_coords:
        return patch_A, patch_B, H4Pt_gt, (x, y)
    return patch_A, patch_B, H4Pt_gt

def create_panorama(warped_patch, patch_B, overlap=64):
    """
    Creates a simple panorama by stitching together the warped patch (output from the unsupervised model)
    and patch_B. A horizontal overlap (default 64 pixels) is blended between the two patches.
    """
    # Both warped_patch and patch_B are assumed to be of size (128, 128, 3)
    h, w, _ = warped_patch.shape
    panorama_width = w + (w - overlap)
    panorama = np.zeros((h, panorama_width, 3), dtype=np.uint8)

    # Place warped_patch on the left side.
    panorama[:, :w, :] = warped_patch

    # Blend the overlapping region.
    # The overlapping region uses the rightmost 'overlap' columns of warped_patch and the leftmost 'overlap'
    # columns of patch_B.
    alpha = np.linspace(0, 1, overlap).reshape(1, overlap, 1)  # from 0 to 1 horizontally
    blend_left = warped_patch[:, w - overlap:w, :].astype(np.float32)
    blend_right = patch_B[:, :overlap, :].astype(np.float32)
    blended = blend_left * (1 - alpha) + blend_right * alpha
    blended = blended.astype(np.uint8)
    panorama[:, w - overlap:w, :] = blended

    # Place the remaining (non-overlapping) part of patch_B on the right.
    panorama[:, w:panorama_width, :] = patch_B[:, overlap:w, :]

    return panorama

def main():
    # Path to the test image (update as needed)
    img_path = r"YourDirectoryID_p1\Phase2\P1Ph2TestSet\Phase2\1.jpg.jpg"
    image = cv2.imread(img_path)
    if image is None:
        print("Error loading image.")
        return

    # Extract a test patch and its warped version from the image.
    patch_A, patch_B, H4Pt_gt, patch_coords = extract_Test_patch(img_path, patch_size=128, rho=32, return_coords=True)
    if patch_A is None:
        print("Error extracting patch.")
        return

    # Path to the unsupervised model checkpoint (update as needed)
    checkpoint_path = r"YourDirectoryID_p1\Phase2\Code\Checkpoints_p1_Phase2\8amodel_unsupervised_Updated.ckpt"
    model_unsup = Unsupervised_HomographyNet()
    model = load_model(model_unsup, checkpoint_path)

    # Prepare input for the unsupervised model.
    # Convert patch_A and patch_B to tensors of shape (1, 3, 128, 128) and normalize to [0,1].
    patch_A_tensor = torch.tensor(patch_A, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    patch_B_tensor = torch.tensor(patch_B, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    # Concatenate along the channel dimension to form a (1, 6, 128, 128) tensor.
    x_input = torch.cat((patch_A_tensor, patch_B_tensor), dim=1)
    
    # Compute the 4 corner coordinates (C_A) for patch_A in the full image.
    # Here, patch_coords is the (x,y) location of the upper-left corner.
    x0, y0 = patch_coords
    corners_A = np.array([
        [x0, y0],
        [x0 + 128 - 1, y0],
        [x0 + 128 - 1, y0 + 128 - 1],
        [x0, y0 + 128 - 1]
    ], dtype=np.float32)
    C_A = torch.tensor(corners_A, dtype=torch.float32).unsqueeze(0)

    # Run the unsupervised model to obtain the warped version of patch_A.
    # (Internally, your model computes the homography H and warps patch_A accordingly.)
    with torch.no_grad():
        warped_patch = model(x_input, C_A)  # Expected shape: (1, 3, 128, 128)
    warped_patch_np = warped_patch.squeeze().permute(1, 2, 0).cpu().numpy()
    # Convert from [0,1] float to [0,255] uint8.
    warped_patch_np = (warped_patch_np * 255).clip(0, 255).astype(np.uint8)

    # Now, create a panorama using the warped patch (from patch_A) and patch_B.
    panorama = create_panorama(warped_patch_np, patch_B, overlap=64)

    # Display the results: Patch A, Patch B, Warped Patch, and Panorama.
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(patch_A, cv2.COLOR_BGR2RGB))
    plt.title("Patch A (Original)")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(patch_B, cv2.COLOR_BGR2RGB))
    plt.title("Patch B (Synthetic Warp)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(warped_patch_np, cv2.COLOR_BGR2RGB))
    plt.title("Warped Patch (Model Output)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title("Panorama")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
