import cv2
import os
import sys

# Windows Python 3.8+ DLL loading fix for CuPy using pip wheels
if os.name == 'nt':
    try:
        for p in sys.path:
            nvrtc_path = os.path.join(p, 'nvidia', 'cuda_nvrtc', 'bin')
            if os.path.exists(nvrtc_path):
                os.add_dll_directory(nvrtc_path)
                os.environ['CUDA_PATH'] = os.path.dirname(nvrtc_path)
    except Exception:
        pass

import cupy as cp
import numpy as np
import sys
import os

# Read the kernel code
kernel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernel.cu')
with open(kernel_path, 'r') as f:
    kernel_code = f.read()

# Compile the raw kernel using CuPy
pixel_sort_kernel = cp.RawKernel(kernel_code, 'pixel_sort_kernel')

def apply_pixel_sort(img_np, threshold, mode):
    height, width, channels = img_np.shape
    if channels != 3:
        raise ValueError("Image must have 3 channels (BGR)")

    # Ensure array is contiguous and uint8
    img_np = np.ascontiguousarray(img_np, dtype=np.uint8)
    
    # Move to GPU
    img_cp = cp.asarray(img_np)
    
    # Calculate grid and block sizes
    # 1 thread per row
    block_size = 256
    grid_size = (height + block_size - 1) // block_size
    
    # Execute kernel
    pixel_sort_kernel((grid_size,), (block_size,), (img_cp, width, height, threshold, mode))
    
    # Copy back to CPU
    result_np = cp.asnumpy(img_cp)
    return result_np

def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = 'images.jpg' # Default from your ipynb
        if not os.path.exists(img_path):
            # Create a beautiful dummy gradient image if none exists
            img_path = 'dummy.jpg'
            x = np.linspace(0, 255, 800)
            y = np.linspace(0, 255, 600)
            X, Y = np.meshgrid(x, y)
            dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
            dummy_img[:, :, 0] = (X + Y) / 2 # B
            dummy_img[:, :, 1] = X           # G
            dummy_img[:, :, 2] = Y           # R
            cv2.imwrite(img_path, dummy_img)
            print("Created a dummy image for testing since images.jpg was not found.")
            
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return

    # Configuration state
    modes = {
        1: "Sort Brightest (T > Threshold)",
        2: "Sort Darkest (T < Threshold)",
        3: "First Band Only (T > Threshold)"
    }
    current_mode = 1
    threshold = 300

    window_name = "CUDA Pixel Sorter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Resize window to fit screen if image is too large
    screen_width, screen_height = 1280, 720
    disp_width, disp_height = img.shape[1], img.shape[0]
    if disp_width > screen_width or disp_height > screen_height:
        scale = min(screen_width / disp_width, screen_height / disp_height)
        cv2.resizeWindow(window_name, int(disp_width * scale), int(disp_height * scale))
    else:
        cv2.resizeWindow(window_name, disp_width, disp_height)

    print("\n==============================")
    print("CUDA Pixel Sorter Initialized")
    print("==============================")
    print("Controls:")
    print("  '1' : Mode 1 - Sort Brightest")
    print("  '2' : Mode 2 - Sort Darkest")
    print("  '3' : Mode 3 - First Band Only")
    print("  'w' : Increase Threshold (+10)")
    print("  's' : Decrease Threshold (-10)")
    print("  'c' : Save Output Image (sorted.png)")
    print("  'ESC' or 'q' : Quit\n")

    needs_update = True
    display_img = img.copy()

    while True:
        if needs_update:
            print(f"Applying Effect: {modes[current_mode]} | Threshold: {threshold}")
            # Ensure image is not modified in-place by copying for the input
            display_img = apply_pixel_sort(img.copy(), threshold, current_mode)
            
            # Draw overlay info (creates a subtle background for text)
            overlay = display_img.copy()
            cv2.rectangle(overlay, (5, 5), (550, 100), (0, 0, 0), -1)
            display_img = cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0)
            
            cv2.putText(display_img, f"Mode: {modes[current_mode]}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_img, f"Threshold: {threshold} (Press w/s to change)", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            needs_update = False

        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord('q'): # ESC or q
            break
        elif key == ord('1'):
            current_mode = 1
            needs_update = True
        elif key == ord('2'):
            current_mode = 2
            needs_update = True
        elif key == ord('3'):
            current_mode = 3
            needs_update = True
        elif key == ord('w'):
            threshold = min(765, threshold + 10)
            needs_update = True
        elif key == ord('s'):
            threshold = max(0, threshold - 10)
            needs_update = True
        elif key == ord('c'):
            # Save without the text overlay
            clean_output = apply_pixel_sort(img.copy(), threshold, current_mode)
            cv2.imwrite("sorted.png", clean_output)
            print("-> Successfully saved glitch art to 'sorted.png'")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
