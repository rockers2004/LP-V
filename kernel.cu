extern "C" {

__device__ void swap_uchar3(uchar3* a, uchar3* b) {
    uchar3 temp = *a;
    *a = *b;
    *b = temp;
}

__device__ int get_brightness(uchar3 pixel) {
    // Note: CuPy/OpenCV images are usually loaded in BGR format, 
    // but we just sum them to get a simple brightness measure.
    return (int)pixel.x + (int)pixel.y + (int)pixel.z;
}

// Iterative quick sort for a segment [l, h] in the array
__device__ void quickSort(uchar3* arr, int l, int h) {
    // We use a custom local stack for iterative Quick Sort.
    // The maximum size of the stack is bounded. By pushing the larger
    // sub-array first, the stack depth is at most O(log N).
    // An image width of 4096 means max depth ~12. 64 is very safe.
    int stack[64];
    int top = -1;

    stack[++top] = l;
    stack[++top] = h;

    while (top >= 0) {
        // Pop h and l
        h = stack[top--];
        l = stack[top--];

        // Partition
        // Use middle element as pivot to avoid O(N) worst case stack overflow on sorted data
        int mid = l + (h - l) / 2;
        swap_uchar3(&arr[mid], &arr[h]);
        
        uchar3 pivot = arr[h];
        int pivot_val = get_brightness(pivot);
        int i = (l - 1);

        for (int j = l; j <= h - 1; j++) {
            if (get_brightness(arr[j]) < pivot_val) {
                i++;
                swap_uchar3(&arr[i], &arr[j]);
            }
        }
        swap_uchar3(&arr[i + 1], &arr[h]);
        int p = i + 1;

        // Push smaller sub-array last to minimize stack depth
        if (p - 1 - l > h - p - 1) {
            if (p - 1 > l) {
                stack[++top] = l;
                stack[++top] = p - 1;
            }
            if (p + 1 < h) {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
        } else {
            if (p + 1 < h) {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
            if (p - 1 > l) {
                stack[++top] = l;
                stack[++top] = p - 1;
            }
        }
    }
}

__global__ void pixel_sort_kernel(uchar3* image, int width, int height, int threshold, int mode) {
    // 1 thread = 1 row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= height) return;

    uchar3* row_data = &image[row * width];
    
    int segment_start = -1;
    
    for (int col = 0; col < width; col++) {
        int brightness = get_brightness(row_data[col]);
        bool condition = false;
        
        if (mode == 1 || mode == 3) {
            condition = (brightness > threshold); // Brightest
        } else if (mode == 2) {
            condition = (brightness < threshold); // Darkest
        }
        
        if (condition) {
            if (segment_start == -1) {
                segment_start = col;
            }
        } else {
            if (segment_start != -1) {
                int segment_end = col - 1;
                if (segment_end > segment_start) {
                    quickSort(row_data, segment_start, segment_end);
                }
                segment_start = -1;
                
                // If mode 3 (First band), we stop after the first sorted band
                if (mode == 3) return;
            }
        }
    }
    
    // Sort the last segment if it extends to the end of the row
    if (segment_start != -1) {
        int segment_end = width - 1;
        if (segment_end > segment_start) {
            quickSort(row_data, segment_start, segment_end);
        }
    }
}

} // extern "C"
