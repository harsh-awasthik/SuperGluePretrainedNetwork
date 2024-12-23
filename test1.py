import cv2
import torch
from function import *
import numpy as np
import matplotlib.cm as cm

from models.superpoint import SuperPoint
from models.superglue import SuperGlue

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SuperPoint and SuperGlue configurations
superpoint_config = {
    'descriptor_dim': 256,
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 1024,
}
superglue_config = {
    'weights': 'indoor',  # Use 'outdoor' for outdoor images
    'sinkhorn_iterations': 20,
    'match_threshold': 0.2,
}

# Initialize models
superpoint = SuperPoint(superpoint_config).to(device)
superglue = SuperGlue(superglue_config).to(device)

# Function to load and preprocess images
def load_and_preprocess_anchor(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    width = int(image.shape[1] * 0.25)
    height = int(image.shape[0] * 0.25)
    dim = (width, height)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # lst = get_n_points(8, image)
    # image = draw_quadrilateral(image, lst)
    image_tensor = torch.from_numpy(image / 255.0).float()[None, None, :, :].to(device)
    return image, image_tensor

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    width = int(image.shape[1] * 0.25)
    height = int(image.shape[0] * 0.25)
    dim = (width, height)

    # Resize the image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_tensor = torch.from_numpy(image / 255.0).float()[None, None, :, :].to(device)
    return image, image_tensor

# Load and preprocess two images
image1_path = "img/img1.jpg"
image2_path = "img/img2.jpg"
image1, image1_tensor = load_and_preprocess_anchor(image1_path)
image2, image2_tensor = load_and_preprocess_image(image2_path)

# Extract keypoints using SuperPoint
data1 = superpoint({'image': image1_tensor})
data2 = superpoint({'image': image2_tensor})

# Prepare data for SuperGlue, including 'scores0' and 'scores1'
keypoints1, descriptors1, scores1 = data1['keypoints'][0], data1['descriptors'][0], data1['scores'][0]
keypoints2, descriptors2, scores2 = data2['keypoints'][0], data2['descriptors'][0], data2['scores'][0]

data = {
    'keypoints0': keypoints1[None],
    'keypoints1': keypoints2[None],
    'descriptors0': descriptors1[None],
    'descriptors1': descriptors2[None],
    'scores0': scores1[None],  # Add 'scores0'
    'scores1': scores2[None],  # Add 'scores1'
    'image0': image1_tensor,
    'image1': image2_tensor,
}

# Perform keypoint matching using SuperGlue
data = {k: v.to(device) for k, v in data.items()}
matches = superglue(data)
matches = matches['matches0'][0].cpu().numpy()

# Visualize matches
def visualize_matches(image1, image2, keypoints1, keypoints2, matches):
    keypoints1 = keypoints1.cpu().numpy()
    keypoints2 = keypoints2.cpu().numpy()
    valid_matches = matches > -1
    matched_keypoints1 = keypoints1[valid_matches]
    matched_keypoints2 = keypoints2[matches[valid_matches]]

    # Create a blank canvas for visualization
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    # Place images side by side
    canvas[:h1, :w1, :] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1 + w2, :] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    # Draw lines between matched keypoints
    for pt1, pt2 in zip(matched_keypoints1, matched_keypoints2):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2 + [w1, 0]))  # Offset x-coordinates for the second image
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    return canvas

# Generate and save the output
output_image = visualize_matches(image1, image2, keypoints1, keypoints2, matches)

# Display the matches
cv2.imshow("Matched Keypoints", output_image)
cv2.imwrite("matched_keypoints.jpg", output_image)  # Save the output image
cv2.waitKey(0)
cv2.destroyAllWindows()
