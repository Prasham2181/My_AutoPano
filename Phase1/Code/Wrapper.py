# !/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""
# Prasham Soni & Sarthak Mehta 
# Phase 1: My autoPano


from calendar import c
import dis
import enum
from multiprocessing import process
import re
# from networkx import numeric_mixing_matrix
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

def plot(img,output_folder,filename=None, cmap=None):

    os.makedirs(output_folder, exist_ok=True)

    # Display the image
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()

    # Save the image if a filename is provided
    if filename:
        save_path = os.path.join(output_folder, filename)
        plt.imsave(save_path, img, cmap=cmap)

def corner_detection(img, idx):

	img = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Harris corner detection using goodFeaturesToTrack
	corners = cv2.goodFeaturesToTrack(gray, 5000, 0.0001, 3, useHarrisDetector=True, k=0.04)
	corners = np.squeeze(corners).astype(np.intp)  # Convert to 2D array of shape (N, 2)

	# Compute corner scores (pixel intensities at corner locations)
	corner_scores = np.array([gray[y, x] for x, y in corners])

	# Draw circles at detected corners
	for corner in corners:
		x, y = corner
		cv2.circle(img, (x, y), 2, (0, 0, 255), -1) 
	
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

	# plot(img, output_folder=f"../Code/ImageCorners/Corner{idx}.png")

	# cv2.imshow("pano1", pano)
	cv2.imshow("img1", img)
	# cv2.imsave()

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return corners, corner_scores

def read_set_of_images(folder_path):
	folder_path = os.path.normpath(folder_path)

	img_files = []
	paths = os.path.join(folder_path, "*.jpg")
	
	img_files = glob.glob(paths)

	img_files = sorted(img_files, key=lambda x: os.path.basename(x))
	
	return img_files

def downsample_image(img, scale=0.25):
    """
    Downsamples the given image by 'scale' in each dimension.
    For example, scale=0.5 makes width and height half.
    """
    height, width = img.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Use INTER_LINEAR for general resizing
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_img

def anms(corners, scores, N_best=100):
	N_strong = len(corners)
	
	# Initialize radius array with infinity
	r = np.inf * np.ones(N_strong)
	
	# ANMS logic
	for i in range(N_strong):
		for j in range(N_strong):
			if scores[j] > scores[i]:
				ed = np.sum((corners[i] - corners[j]) ** 2)  # Compute squared Euclidean distance
				if ed < r[i]:
					r[i] = ed
	
	# Sorted the  corners based on descending order of radius
	sorted_indices = np.argsort(-r)
	best_indices = sorted_indices[:N_best]
	best_corners = corners[best_indices]

	return best_corners

def feature_descriptor(best_corners, image):

	feature_corners = best_corners
	coords = len(feature_corners)

	coords_vec = []

	# img = cv2.imread(image)

	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	feature_vectors = []

	for i in range(coords):

		matrix = np.zeros((41, 41))

		x = feature_corners[i][0]
		y = feature_corners[i][1]

		if x - 20 >= 0 and x + 20 < gray_img.shape[1] and y - 20 >= 0 and y + 20 < gray_img.shape[0]:
			matrix = gray_img[y-20:y+21, x-20:x+21]

			blur_patch = cv2.GaussianBlur(matrix, (3,3), 0)

			resize_patch = cv2.resize(blur_patch, (8,8), interpolation=cv2.INTER_LINEAR)

			feature_vector = resize_patch.flatten()

			standardized_patch = (feature_vector-feature_vector.mean())/feature_vector.std()

			feature_vectors.append(standardized_patch)

			coords_vec.append((x,y))
		
		else:
			print(f"Skipping point ({x}, {y}) as it is too close to the edge.")
			continue

	return feature_vectors, coords_vec

def list_array_to_keypoints(coordinates_array):
    keypoints = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in coordinates_array]
    return keypoints

def match_features(feature_vectors1, feature_vectors2, coords1, coords2, ratio_threshold=0.4):
    """
    Match features between two images using ratio test
    """
    matches = []
    matched_coords1 = []
    matched_coords2 = []
    
    # Convert feature vectors to numpy arrays for faster computation
    vectors1 = np.array(feature_vectors1)
    vectors2 = np.array(feature_vectors2)
    
    # For each feature in first image
    for i, feat1 in enumerate(vectors1):
        # Compute distances to all features in second image
        distances = np.sum((vectors2 - feat1) ** 2, axis=1)
        
        # Find indices of two best matches
        sorted_idx = np.argsort(distances)
        best_idx = sorted_idx[0]
        second_best_idx = sorted_idx[1]
        
        # Apply ratio test
        ratio = distances[best_idx] / distances[second_best_idx]
        
        if ratio < ratio_threshold:
            # Create DMatch object for visualization
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = best_idx
            match.distance = distances[best_idx]
            
            matches.append(match)
            matched_coords1.append(coords1[i])
            matched_coords2.append(coords2[best_idx])
    
    return matches, matched_coords1, matched_coords2

def visualize_matches(img1, img2, keypoints1, keypoints2, matches, idx1, idx2):
	"""
	Visualize matched features between two images
	"""
 
	output_dir = os.path.join(os.getcwd(), 'Code', 'feature_matches')
	os.makedirs(output_dir, exist_ok=True)

	matched_img = cv2.drawMatches(img1, keypoints1, 
						img2, keypoints2, 
						matches, 
						None,
						matchColor=(0,0,255),      # Single color for all lines
						singlePointColor=None,       # Don't draw keypoints
						matchesMask=None,            # Draw all matches
						flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)  # Default flags

	output_path = os.path.join(output_dir, f'matches_{idx1}_{idx2}.png')
	cv2.imwrite(output_path, matched_img)
	print(f"Saved matches visualization to: {output_path}")

	plt.figure(figsize=(15, 5))
	plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	plt.title(f'Feature Matches between images {idx1} and {idx2} ({len(matches)} matches)')
	plt.show()
	plt.close()

def ransac(keypoints1, keypoints2, matches, iterations=8000, threshold=0.2):
    all_points_1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    all_points_2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if len(matches) < 4:
        print("Not enough matches to compute a homography!")
        return None, []

    best_inliers_count = 0
    best_H = None
    best_inliers = None

    for i in range(iterations):
        try:
            selected_matches = np.random.choice(matches, 4, replace=False)
        except ValueError:
            print(f"Error: Unable to sample 4 matches from {len(matches)} available matches.")
            continue

        selected_points_1 = np.float32([keypoints1[m.queryIdx].pt for m in selected_matches]).reshape(-1, 1, 2)
        selected_points_2 = np.float32([keypoints2[m.trainIdx].pt for m in selected_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(selected_points_1, selected_points_2, 0)
        if H is None:
            continue

        transformed_points = cv2.perspectiveTransform(all_points_1, H)

        distances = np.sqrt(np.sum((all_points_2 - transformed_points) ** 2, axis=2)).flatten()

        inliers = distances < threshold

        inliers_count = np.sum(inliers)
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_H = H
            best_inliers = inliers

        if best_inliers_count / len(matches) > 0.95:  
            print(f"Early stopping at iteration {i} with {best_inliers_count} inliers.")
            break

    inlier_matches = [matches[j] for j in range(len(matches)) if best_inliers[j]] if best_inliers is not None else []

    # Recalculate homography with all inliers
    if len(inlier_matches) > 4:
        final_points1 = np.float32([keypoints1[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        final_points2 = np.float32([keypoints2[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        best_H, _ = cv2.findHomography(final_points1, final_points2, cv2.RANSAC, threshold)

    return best_H, inlier_matches

# # Cylindrical Wrapping and blending 

# def cylindrical_warp(img, focal_length):
#     """
#     Perform cylindrical warping of an image given a focal length.
#     """
#     h, w = img.shape[:2]
#     K = np.array([[focal_length, 0, w / 2],
#                   [0, focal_length, h / 2],
#                   [0, 0, 1]])  # Intrinsic camera matrix

#     # Initialize cylindrical projection
#     cylindrical_img = np.zeros_like(img)
#     cylindrical_mask = np.zeros((h, w), dtype=np.uint8)

#     for y in range(h):
#         for x in range(w):
#             # Convert (x, y) to normalized device coordinates
#             xn = (x - K[0, 2]) / K[0, 0]
#             yn = (y - K[1, 2]) / K[1, 1]

#             # Project onto cylinder
#             theta = np.arctan(xn)
#             h_ = yn / np.sqrt(xn**2 + 1)

#             # Convert back to pixel coordinates
#             x_cyl = int(K[0, 2] + focal_length * theta)
#             y_cyl = int(K[1, 2] + focal_length * h_)

#             # Check bounds
#             if 0 <= x_cyl < w and 0 <= y_cyl < h:
#                 cylindrical_img[y_cyl, x_cyl] = img[y, x]
#                 cylindrical_mask[y_cyl, x_cyl] = 1

#     # Fill missing pixels using inpainting
#     inpainted_img = cv2.inpaint(cylindrical_img, 255 - cylindrical_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

#     return inpainted_img


# def warping_blending(img1, img2, h, focal_length=700):
#     """
#     Perform cylindrical warping and blending for panorama stitching.
#     """
#     # Apply cylindrical warping to both images
#     img1_cyl = cylindrical_warp(img1, focal_length)
#     img2_cyl = cylindrical_warp(img2, focal_length)

#     # Dimensions of the warped images
#     h1, w1 = img2_cyl.shape[:2]
#     h2, w2 = img1_cyl.shape[:2]

#     # Corner points of img2
#     pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
#     pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

#     # Warp the corners of img1 according to homography h
#     pts2_ = cv2.perspectiveTransform(pts2, h)

#     # Combine corners from img2 (reference) and the warped corners of img1
#     pts = np.concatenate((pts1, pts2_), axis=0)

#     # Find the bounding box of all these corners
#     [xmin, ymin] = np.int32(pts.min(axis=0).ravel())
#     [xmax, ymax] = np.int32(pts.max(axis=0).ravel())

#     # Compute width and height of the new canvas
#     width = xmax - xmin
#     height = ymax - ymin

#     # Translation to shift the panorama so that all coordinates are >= 0
#     t = [-xmin, -ymin]

#     # Create the translation matrix
#     Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)

#     # Warp img1 onto the new (width x height) canvas using Ht * h
#     result = cv2.warpPerspective(img1_cyl, Ht.dot(h), (width, height))

#     # Place img2 into the result, offset by t
#     y_start = t[1]
#     y_end = t[1] + h1
#     x_start = t[0]
#     x_end = t[0] + w1

#     result[y_start:y_end, x_start:x_end] = img2_cyl

#     cv2.imshow('Cylindrical Panorama', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return result


def warping_blending(img1, img2, h):
	h1, w1 = img2.shape[:2]
	h2, w2 = img1.shape[:2]

	pts1 = np.float32([[0, 0],
						[0, h1],
						[w1, h1],
						[w1, 0]]).reshape(-1, 1, 2)
	pts2 = np.float32([[0, 0],
						[0, h2],
						[w2, h2],
						[w2, 0]]).reshape(-1, 1, 2)

	pts2_ = cv2.perspectiveTransform(pts2, h)

	pts = np.concatenate((pts1, pts2_), axis=0)

	[xmin, ymin] = np.int32(pts.min(axis=0).ravel())
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel())

	# #  clipping function can worlk but gives 2 images only at
	# xmin = max(xmin, 0)
	# ymin = max(ymin, 0)
	# xmax = max(xmax, img1.shape[1], img2.shape[1])  
	# ymax = max(ymax, img1.shape[0], img2.shape[0])
 
	# Compute width and height of the new canvas
	width = xmax - xmin
	height = ymax - ymin

	t = [-xmin, -ymin]

	print(f"xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
	print(f"width={width}, height={height}")

	if width <= 0 or height <= 0:
		print("Warning: Invalid canvas size. Returning original img1.")
		return img1

	Ht = np.array([
		[1, 0, t[0]],
		[0, 1, t[1]],
		[0, 0, 1]
	], dtype=np.float32)

	result = cv2.warpPerspective(img1, Ht.dot(h), (width, height))

	# NOTE: We must ensure the region [t[1]:, t[0]:] fits in `result`
	y_start = t[1]
	y_end = t[1] + h1
	x_start = t[0]
	x_end = t[0] + w1

	if y_end > result.shape[0] or x_end > result.shape[1]:
		print("Warning: img2 goes out of bounds in the stitched canvas.")

	result[y_start:y_end, x_start:x_end] = img2

	cv2.imshow('Result', result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return result

def main():
    """
    Read a set of images for Panorama stitching
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # train_folder = os.path.normpath(os.path.join(current_dir, '..', 'Data', 'Train', 'Set3'))
    train_folder = os.path.normpath(os.path.join(current_dir, '..', 'Data', 'Train', 'TestSet1'))
    # train_folder = os.path.normpath(os.path.join(current_dir, '..', 'Data', 'Train', 'CustomSet1'))
    # train_folder = os.path.normpath(os.path.join(current_dir, '..', 'Data', 'Train', 'TestSet4_updated'))
	# Note: We had to re arrange the images in the testset 4 as the original sequence was not processing due to no feature matches after the 4th image

    image_set = read_set_of_images(train_folder)
    
    pano = cv2.imread(image_set[0])
    pano = downsample_image(pano, scale=0.5)  # <-- Downsample the initial panorama

    for idx in range(1, len(image_set)):
        img = cv2.imread(image_set[idx])
        img = downsample_image(img, scale=0.5)  # <-- Downsample each new image

        cv2.imshow("pano1", pano)
        cv2.imshow("img1", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        """
        Corner Detection
        Save Corner detection output as corners.png
        """
        print(f"Detecting corners for current panorama and image {idx}...")
        corners_pano, scores_pano = corner_detection(pano, idx-1)
        corners_img, scores_img = corner_detection(img, idx)

        cv2.imshow("pano1", pano)
        cv2.imshow("img1", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
        print("ANMS Started....")
        best_corners_panorama = anms(corners_pano, scores_pano, N_best=600)
        best_corners_img = anms(corners_img, scores_img, N_best=600)

        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """
        print("Extracting feature descriptor....")
        feat_vec_pano, coords_pano = feature_descriptor(best_corners_panorama, pano)
        feat_vec_img, coords_img = feature_descriptor(best_corners_img, img)

        """
        Feature Matching
        Save Feature Matching output as matching.png
        """
        print("Matching feature....")
        matches, matches_coords_pano, matches_coords_img = match_features(
            feat_vec_pano, feat_vec_img,
            coords_pano, coords_img,
            ratio_threshold=0.5  # or 0.7, etc.
        )

        keypoints_pano = list_array_to_keypoints(coords_pano)
        keypoints_img = list_array_to_keypoints(coords_img)

        visualize_matches(pano, img, keypoints_pano, keypoints_img, matches, idx-1, idx)

        """
        Refine: RANSAC, Estimate Homography
        """
        H, inliers = ransac(keypoints_pano, keypoints_img, matches, iterations=10000, threshold=2)
        det = np.linalg.det(H)
        print("Det(H) =", det)
        if H is not None:
            ransac_output = visualize_matches(pano, img, keypoints_pano, keypoints_img, inliers, idx-1, idx)

        """
        Image Warping + Blending
        Save Panorama output as mypano_{idx}.png
        """
        print(f"Before warping: pano shape = {pano.shape}, new image shape = {img.shape}")
        pano = warping_blending(pano, img, H)
        print(f"After warping: pano shape = {pano.shape}")

        # Save the panorama after blending with the current image
        output_filename = f"mypano_{idx}.png"
        output_path = os.path.join(current_dir, output_filename)
        cv2.imwrite(output_path, pano)
        print(f"Panorama saved as {output_filename}")


if __name__ == "__main__":
    main()
