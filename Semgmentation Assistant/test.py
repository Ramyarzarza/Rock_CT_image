import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# Initialize global variables
drawing = False
brush_size = 5
mask = None

def draw_freehand(event, x, y, flags, param):
    global drawing, mask, image

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        drawing = True
        cv2.circle(mask, (x, y), brush_size, 255, -1)

    elif event == cv2.EVENT_MOUSEMOVE:  # Draw while moving the mouse
        if drawing:
            cv2.circle(mask, (x, y), brush_size, 255, -1)

    elif event == cv2.EVENT_LBUTTONUP:  # Stop drawing and apply segmentation
        drawing = False
        apply_segmentation()

def apply_segmentation():
    global image, mask

    # Extract selected region using the mask
    selected_pixels = cv2.bitwise_and(image, image, mask=mask)

    # Reshape selected region for clustering
    pixels = selected_pixels.reshape(-1, 3)
    pixels = pixels[np.any(pixels != 0, axis=1)]  # Remove black (unselected) pixels

    if pixels.size == 0:
        print("No pixels selected for segmentation.")
        return

    # Estimate bandwidth for Mean Shift
    bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=500)

    # Apply Mean Shift clustering
    mean_shift = MeanShift(bandwidth=5 , bin_seeding=True)
    mean_shift.fit(pixels)

    # Get cluster centers and labels
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_

    # Change lighter cluster centers to white
    lightness_threshold = 60  # Threshold for considering a color "light"
    for i, center in enumerate(cluster_centers):
        if np.mean(center) > lightness_threshold:  # Calculate average intensity
            cluster_centers[i] = [255, 255, 255]  # Set to white

    # Create segmented pixels
    segmented_pixels = cluster_centers[labels].astype(np.uint8)
    segmented_region = np.zeros_like(selected_pixels)
    segmented_region[np.any(selected_pixels != 0, axis=2)] = segmented_pixels

    # Add the segmented region to the main image
    white_mask = cv2.inRange(segmented_region, (255, 255, 255), (255, 255, 255))
    image[white_mask == 255] = [255, 255, 255]

    # Clear the mask for the next iteration
    mask.fill(0)

# Load the image
image_path = 'aligned_250.png'  # Replace with the path to your image
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load image.")
    exit()

# Create a black mask with the same dimensions as the image
mask = np.zeros(image.shape[:2], dtype=np.uint8)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_freehand)

while True:
    # Display the current state of the image
    display_image = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow("Image", display_image)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit
    if key == ord("q"):
        break

cv2.destroyAllWindows()
