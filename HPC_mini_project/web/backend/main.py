from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import io
import sys

# Windows Python 3.8+ DLL loading fix for CuPy using pip wheels
if os.name == 'nt':
    try:
        for p in sys.path:
            nvrtc_path = os.path.join(p, 'nvidia', 'cuda_nvrtc', 'bin')
            if os.path.exists(nvrtc_path):
                os.add_dll_directory(nvrtc_path)
                os.environ['CUDA_PATH'] = os.path.dirname(nvrtc_path)
                print(f"Added DLL directory and set CUDA_PATH: {nvrtc_path}")
    except Exception:
        pass

import cv2
import cupy as cp
import numpy as np

app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CUDA kernel
kernel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kernel.cu')
try:
    with open(kernel_path, 'r') as f:
        kernel_code = f.read()
    pixel_sort_kernel = cp.RawKernel(kernel_code, 'pixel_sort_kernel')
    print("Successfully loaded CuPy Kernel.")
except Exception as e:
    print(f"Error loading kernel: {e}")
    pixel_sort_kernel = None

def apply_pixel_sort(img_np, threshold, mode, direction, sorting_style=0, masking=0, mask_rect=""):
    if pixel_sort_kernel is None:
        raise RuntimeError("CUDA Kernel not initialized. Please ensure the CUDA Toolkit and NVRTC are properly installed for GPU compilation.")
        
    height, width, channels = img_np.shape
    if channels != 3:
        raise ValueError("Image must have 3 channels (BGR)")

    # Build masking array
    mask_np = np.ones((height, width), dtype=np.uint8) * 255
    if masking > 0 and mask_rect:
        try:
            parts = mask_rect.split(',')
            x = max(0.0, min(1.0, float(parts[0])))
            y = max(0.0, min(1.0, float(parts[1])))
            w = max(0.0, min(1.0, float(parts[2])))
            h_rect = max(0.0, min(1.0, float(parts[3])))
            
            px = int(x * width)
            py = int(y * height)
            pw = int(w * width)
            ph = int(h_rect * height)
            
            if masking == 1: # Inside mask
                mask_np[:] = 0
                mask_np[py:py+ph, px:px+pw] = 255
            elif masking == 2: # Outside mask
                mask_np[py:py+ph, px:px+pw] = 0
        except Exception as e:
            print("Mask parsing error:", e)

    img_np_contig = np.ascontiguousarray(img_np, dtype=np.uint8)
    img_cp = cp.asarray(img_np_contig)
    mask_cp = cp.asarray(mask_np)

    # Handle Direction using Rotations/Flips to keep the CUDA kernel simple and fast (1D horizontal sort)
    if direction == 1: # Right to Left
        img_cp = cp.flip(img_cp, 1)
        mask_cp = cp.flip(mask_cp, 1)
    elif direction == 2: # Top to Bottom
        img_cp = cp.asarray(cv2.rotate(cp.asnumpy(img_cp), cv2.ROTATE_90_COUNTERCLOCKWISE))
        mask_cp = cp.asarray(cv2.rotate(cp.asnumpy(mask_cp), cv2.ROTATE_90_COUNTERCLOCKWISE))
        height, width = img_cp.shape[0], img_cp.shape[1]
    elif direction == 3: # Bottom to Top
        img_cp = cp.asarray(cv2.rotate(cp.asnumpy(img_cp), cv2.ROTATE_90_CLOCKWISE))
        mask_cp = cp.asarray(cv2.rotate(cp.asnumpy(mask_cp), cv2.ROTATE_90_CLOCKWISE))
        height, width = img_cp.shape[0], img_cp.shape[1]

    block_size = 256
    grid_size = (height + block_size - 1) // block_size
    
    # Execute the CUDA kernel
    pixel_sort_kernel((grid_size,), (block_size,), (img_cp, mask_cp, np.int32(width), np.int32(height), np.int32(threshold), np.int32(mode), np.int32(sorting_style), np.int32(masking)))
    
    result_np = cp.asnumpy(img_cp)
    
    # Reverse the transformations
    if direction == 1:
        result_np = cv2.flip(result_np, 1)
    elif direction == 2:
        result_np = cv2.rotate(result_np, cv2.ROTATE_90_CLOCKWISE)
    elif direction == 3:
        result_np = cv2.rotate(result_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    return result_np

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    threshold: int = Form(...),
    mode: int = Form(...),
    direction: int = Form(0), # Default LTR
    sorting_style: int = Form(0),
    masking: int = Form(0),
    mask_rect: str = Form("")
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image format"}

    try:
        # Process image purely on GPU
        result_img = apply_pixel_sort(img, threshold, mode, direction, sorting_style, masking, mask_rect)
        
        # Encode back to JPEG
        _, encoded_img = cv2.imencode('.jpg', result_img)
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Restart Trigger

