import cv2
import numpy as np
import sys

# Usage: python resize_and_border.py input.jpg output.jpg

def main():
    if len(sys.argv) != 3:
        print("Usage: python resize_and_border.py input.jpg output.jpg")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Read image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read {input_path}")
        sys.exit(1)

    # Resize by factor 4
    h, w = img.shape[:2]
    img_small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)

    # Add black border (40 pixels)
    border_size = 40
    img_bordered = cv2.copyMakeBorder(
        img_small, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Save result
    cv2.imwrite(output_path, img_bordered)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
