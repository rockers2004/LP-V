# CUDA Pixel Sorting Glitch Art Studio

A professional, high-performance pixel sorting application built with a FastAPI Python backend, a modern React/Vite frontend, and a native CUDA C++ kernel. It creates studio-grade glitch art effects by selectively sorting pixels using spatial-aware masking, multi-style sorting (Hue, Saturation, Lightness, RGB), and various sorting modes, all accelerated by your GPU.

## Prerequisites

- **NVIDIA GPU** with CUDA Toolkit installed
- **Python 3.8+**
- **Node.js** (v14+ recommended) and npm

## Installation

### 1. Backend Setup (Python)

Navigate to the project root and install the Python dependencies. It is recommended to use a virtual environment.

```powershell
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` includes `cupy-cuda12x`. If you are using CUDA 11, please change it to `cupy-cuda11x` in the `requirements.txt` file before running the install command.

### 2. Frontend Setup (React/Vite)

Navigate to the frontend directory and install the Node.js dependencies.

```powershell
cd web/frontend
npm install
```

## Running the Application

You need to run both the FastAPI backend and the React frontend simultaneously.

### 1. Start the Backend server

Open a terminal, navigate to the backend directory, and start the server:
```powershell
# Make sure your virtual environment is active
cd web/backend
python main.py
```
The backend API will start at `http://localhost:8000`.

*(Alternatively, you can run the legacy OpenCV UI script from the project root by running `python app.py`)*

### 2. Start the Frontend server

Open a new terminal, navigate to the frontend directory, and start the Vite development server:
```powershell
cd web/frontend
npm run dev
```
The frontend will typically be accessible at `http://localhost:5173`. Open this URL in your browser to access the Glitch Art Studio.

## Usage

1. **Import Media**: Click "File" or the "BROWSE" button to load an image.
2. **Interval Style**: Choose between None, Threshold, or Random for glitch intervals.
3. **Threshold Behavior**: Select whether to sort pixels exceeding or below the threshold.
4. **Sort Direction**: Change the direction of the pixel sort (L→R, R←L, T↓B, B↑T).
5. **Sorting Value**: Choose to sort by Lightness, Hue, Saturation, or individual RGB channels.
6. **Spatial Masking**: Apply selective glitching by drawing a bounding box to sort either inside or outside the region.
7. **Export**: Click "RENDER TO CANVAS" to preview, and "EXPORT AS JPEG" to download your creation!

## How it works

The core of the pixel sorting logic is written in `kernel.cu` using native CUDA C++:
- **Thread Mapping**: The kernel assigns one GPU thread to process one entire row of the image simultaneously in parallel.
- **CUDA Quick Sort**: An iterative, in-place Quick Sort algorithm runs directly on the GPU to sort the individual pixel segments.
- **CuPy Integration**: The FastAPI backend uses CuPy dynamically to compile and execute the CUDA kernel, meaning you don't have to deal with CMake or complex Visual Studio linking.
