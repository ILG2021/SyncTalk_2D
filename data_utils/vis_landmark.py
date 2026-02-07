import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm


def visualize_landmarks(image_path, landmark_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return

    # Read the landmark points
    if not os.path.exists(landmark_path):
        return

    lms = []
    with open(landmark_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                lms.append([float(parts[0]), float(parts[1])])

    lms = np.array(lms, dtype=np.int32)

    # Draw landmarks on the image (Green circles)
    for i, (x, y) in enumerate(lms):
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
        # Show indices for each landmark
        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Save the result
    cv2.imwrite(output_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize human face landmarks (110 points)")
    parser.add_argument('--name', type=str, required=True,
                        help="Name of the person in the dataset directory, e.g., May")
    parser.add_argument('--bg_name', type=str, default="",
                        help="Specific background material name (sub-folder in dataset)")
    parser.add_argument('--video_path', type=str, default="",
                        help="Custom video path (if provided, uses its name as a sub-folder)")
    args = parser.parse_args()

    # Define paths following the logic in inference_328.py
    dataset_dir = os.path.join("dataset", args.name)

    if args.video_path:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        dataset_dir = os.path.join(dataset_dir, video_name)
    elif args.bg_name:
        dataset_dir = os.path.join(dataset_dir, args.bg_name)
    elif not os.path.exists(os.path.join(dataset_dir, "full_body_img")):
        # Auto-select first subdirectory if root is empty
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if subdirs:
            print(
                f"[WARN] No --bg_name provided and root dataset is empty. Auto-selecting first material: {subdirs[0]}")
            dataset_dir = os.path.join(dataset_dir, subdirs[0])

    img_dir = os.path.join(dataset_dir, "full_body_img")
    lms_dir = os.path.join(dataset_dir, "landmarks")
    vis_dir = os.path.join(dataset_dir, "vis_landmarks")  # Output directory

    if not os.path.exists(img_dir):
        print(f"[Error] Image directory not found: {img_dir}")
        exit(1)

    if not os.path.exists(lms_dir):
        print(f"[Error] Landmarks directory not found: {lms_dir}")
        exit(1)

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    print(f"Visualizing landmarks for dataset directory: {dataset_dir}...")
    img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not img_names:
        print(f"No images found in {img_dir}")
        exit(0)

    for img_name in tqdm(img_names):
        img_path = os.path.join(img_dir, img_name)
        # Landmarks usually have .lms extension in this project
        lms_name = os.path.splitext(img_name)[0] + ".lms"
        lms_path = os.path.join(lms_dir, lms_name)
        out_path = os.path.join(vis_dir, img_name)

        visualize_landmarks(img_path, lms_path, out_path)

    print(f"\nVisualization complete! Check the output folder: {os.path.abspath(vis_dir)}")
