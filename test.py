import cv2
import numpy as np
from models.superpoint import SuperPoint
from models.utils import frame2tensor
from models.superglue import SuperGlue
import torch

import matplotlib.pyplot as plt

image1 = cv2.imread("RealsenceImages/img0.jpg", cv2.COLOR_BGR2GRAY)
image2 = cv2.imread("RealsenceImages/img6.jpg", cv2.COLOR_BGR2GRAY)

# Resize or preprocess images if needed


# Load the pre-trained SuperPoint model
superpoint = SuperPoint({'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024})

# Convert images to tensors
image1_tensor = frame2tensor(image1, 'cpu')[:, :, :, :,0]  # Use 'cpu' if no GPU
image2_tensor = frame2tensor(image2, 'cpu')[:, :, :, :,0]

# print("Tensor shape:", image1_tensor[:, :, :, :,0].shape)  # Should be [1, 1, height, width]

# Detect keypoints and descriptors
pred1 = superpoint({'image': image1_tensor})
pred2 = superpoint({'image': image2_tensor})
keypoints1, descriptors1 = pred1['keypoints'], pred1['descriptors']
keypoints2, descriptors2 = pred2['keypoints'], pred2['descriptors']

keypoints1 = to_tensor(keypoints1, 'cpu')

# Load the pre-trained SuperGlue model
superglue = SuperGlue({'weights': 'indoor'})

# Prepare inputs for SuperGlue
input_superglue = {
    'keypoints0': keypoints1.unsqueeze(0),
    'keypoints1': keypoints2.unsqueeze(0),
    'descriptors0': descriptors1.unsqueeze(0),
    'descriptors1': descriptors2.unsqueeze(0),
    'image0': image1_tensor.unsqueeze(0),
    'image1': image2_tensor.unsqueeze(0),
}

# Get matches
output = superglue(input_superglue)
matches = output['matches0'][0].cpu().numpy()  # Matches from Image 1 to Image 2

# Coordinates of your 4 points in Image 1
points_image1 = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

# Find nearest keypoints to your selected points
def find_nearest_keypoint(point, keypoints):
    distances = np.linalg.norm(keypoints - point, axis=1)
    return np.argmin(distances)

indices_image1 = [find_nearest_keypoint(pt, keypoints1.cpu().numpy()) for pt in points_image1]

# Get matched points in Image 2
points_image2 = []
for idx in indices_image1:
    matched_idx = matches[idx]
    if matched_idx > -1:  # Valid match
        points_image2.append(keypoints2[matched_idx].cpu().numpy())

points_image2 = np.array(points_image2)


def draw_matches(img1, img2, pts1, pts2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].scatter(pts1[:, 0], pts1[:, 1], c='r')
    ax[0].set_title("Image 1 Points")
    
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].scatter(pts2[:, 0], pts2[:, 1], c='b')
    ax[1].set_title("Image 2 Matches")
    
    plt.show()

draw_matches(image1, image2, points_image1, points_image2)


cv2.imshow("image", points_image2)
cv2.waitKey(0)