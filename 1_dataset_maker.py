from datasets import load_dataset
import cv2
import os
from tqdm import tqdm
import numpy as np


def display_and_save_image(image, save_directory, image_index, show_image=False):
    if show_image:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image_filename = f"{image_index}.jpg"
    image_path = os.path.join(save_directory, image_filename)
    cv2.imwrite(image_path, image)


def main():
    dataset = load_dataset("bhadresh-savani/photo-to-cartoon")
    save_dir_features = "dataset_2/features"
    save_dir_labels = "dataset_2/labels"

    # Create directories if they don't exist
    os.makedirs(save_dir_features, exist_ok=True)
    os.makedirs(save_dir_labels, exist_ok=True)

    # Use tqdm to add a progress bar
    for i, example in enumerate(tqdm(dataset['train'])):
        imageA = example['imageA']
        imageB = example['imageB']

        # Convert PIL images to numpy arrays
        imageA_np = np.array(imageA)
        imageB_np = np.array(imageB)

        # Convert images to BGR format as OpenCV uses BGR by default
        imageA_bgr = cv2.cvtColor(imageA_np, cv2.COLOR_RGB2BGR)
        imageB_bgr = cv2.cvtColor(imageB_np, cv2.COLOR_RGB2BGR)

        # Display and save imageA in features directory
        display_and_save_image(imageA_bgr, save_dir_features, i + 1)

        # Display and save imageB in labels directory
        display_and_save_image(imageB_bgr, save_dir_labels, i + 1)


if __name__ == "__main__":
    main()
