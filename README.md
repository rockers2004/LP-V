# CUDA Pixel Sorting Glitch Art App

A professional, high-performance pixel sorting application built with Python, OpenCV, and CUDA C++. It creates a glitch art effect by selectively sorting pixels in horizontal streaks based on brightness thresholds.

## Requirements

The application uses **CuPy** to compile and run the raw CUDA C++ Quick Sort kernel on your GPU dynamically. This means you do not need to deal with CMake or complex Visual Studio linking for OpenCV and CUDA!

Ensure you have a modern NVIDIA GPU and the CUDA Toolkit installed on your Windows machine.

### Installation

1. Install Python (3.8+ recommended).
2. Install the necessary Python packages using pip. **Note:** Make sure you install the correct `cupy` version matching your installed CUDA toolkit (e.g., `cupy-cuda11x` or `cupy-cuda12x`).

```powershell
pip install opencv-python cupy-cuda12x numpy
```
*(If you have CUDA 11, use `cupy-cuda11x` instead).*

## How to Run

Simply run the Python application. If you don't provide an image path, it will automatically generate a colorful dummy image for you to play with!

```powershell
python app.py
```
Or, pass an image directly:
```powershell
python app.py path/to/your/image.jpg
```

## Controls

Once the window opens, you can control the effect dynamically in real-time using your keyboard:

- `1` : **Mode 1 (Sort Brightest)** - Sorts the pixels in horizontal streaks where the pixel brightness is *greater* than the threshold.
- `2` : **Mode 2 (Sort Darkest)** - Sorts the pixels in horizontal streaks where the pixel brightness is *less* than the threshold.
- `3` : **Mode 3 (First Band Only)** - Creates an interesting "tear" effect by only sorting the first segment of pixels it finds in each row that matches the threshold.
- `w` : **Increase Threshold** (+10)
- `s` : **Decrease Threshold** (-10)
- `c` : **Save Output Image** (Saves the clean glitch art as `sorted.png` in the current directory)
- `ESC` or `q` : **Quit**

## How it works

The core of the pixel sorting logic is written in `kernel.cu` using native CUDA C++:
1. **Thread Mapping**: The kernel assigns one GPU thread to process one entire row of the image, processing all rows simultaneously in parallel.
2. **CUDA Quick Sort**: The kernel utilizes a fully iterative, in-place Quick Sort algorithm implemented natively in CUDA to sort the individual segments within the row.
