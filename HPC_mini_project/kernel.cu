extern "C" {

__device__ void swap_uchar3(uchar3* a, uchar3* b) {
    uchar3 temp = *a;
    *a = *b;
    *b = temp;
}

__device__ float get_sort_value(uchar3 pixel, int style) {
    // 0: Lightness, 1: Hue, 2: Saturation, 3: Red, 4: Green, 5: Blue
    if (style >= 3) {
        if (style == 3) return (float)pixel.z; // Red (OpenCV BGR)
        if (style == 4) return (float)pixel.y; // Green
        if (style == 5) return (float)pixel.x; // Blue
    }
    
    float b = pixel.x / 255.0f;
    float g = pixel.y / 255.0f;
    float r = pixel.z / 255.0f;

    float cmax = r > g ? (r > b ? r : b) : (g > b ? g : b);
    float cmin = r < g ? (r < b ? r : b) : (g < b ? g : b);
    float delta = cmax - cmin;

    if (style == 0) return cmax * 255.0f; // Lightness

    if (style == 1) { // Hue
        float h = 0.0f;
        if (delta > 0.0f) {
            if (cmax == r) {
                h = 60.0f * ((g - b) / delta);
                if (h < 0.0f) h += 360.0f;
            }
            else if (cmax == g) h = 60.0f * (((b - r) / delta) + 2.0f);
            else if (cmax == b) h = 60.0f * (((r - g) / delta) + 4.0f);
        }
        return h;
    }

    if (style == 2) { // Saturation
        float s = cmax == 0.0f ? 0.0f : (delta / cmax);
        return s * 255.0f;
    }
    
    return cmax * 255.0f;
}

// Simple hash-based PRNG for CUDA
__device__ unsigned int rand_hash(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// Iterative quick sort for a segment [l, h] in the array
__device__ void quickSort(uchar3* arr, int l, int h, int sorting_style) {
    int stack[64];
    int top = -1;

    stack[++top] = l;
    stack[++top] = h;

    while (top >= 0) {
        h = stack[top--];
        l = stack[top--];

        int mid = l + (h - l) / 2;
        swap_uchar3(&arr[mid], &arr[h]);
        
        uchar3 pivot = arr[h];
        float pivot_val = get_sort_value(pivot, sorting_style);
        int i = (l - 1);

        for (int j = l; j <= h - 1; j++) {
            if (get_sort_value(arr[j], sorting_style) < pivot_val) {
                i++;
                swap_uchar3(&arr[i], &arr[j]);
            }
        }
        swap_uchar3(&arr[i + 1], &arr[h]);
        int p = i + 1;

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

__global__ void pixel_sort_kernel(uchar3* image, unsigned char* mask, int width, int height, int threshold, int mode, int sorting_style, int masking) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= height) return;

    uchar3* row_data = &image[row * width];
    unsigned char* mask_data = mask ? &mask[row * width] : nullptr;
    
    int segment_start = -1;
    
    // For random mode (mode 5)
    unsigned int rng_state = rand_hash(row + 19937);
    int current_band_target = 0;
    int current_band_len = 0;
    bool is_sorting_band = false;

    if (mode == 5) {
        rng_state = rand_hash(rng_state);
        current_band_target = rng_state % (threshold > 0 ? threshold : 1);
        is_sorting_band = true;
    }
    
    for (int col = 0; col < width; col++) {
        bool condition = false;
        
        if (mode == 1 || mode == 3) {
            float val = get_sort_value(row_data[col], sorting_style);
            condition = (val > threshold); // Brightest equivalent
        } else if (mode == 2) {
            float val = get_sort_value(row_data[col], sorting_style);
            condition = (val < threshold); // Darkest equivalent
        } else if (mode == 4) {
            condition = true; // Full line sort
        } else if (mode == 5) {
            if (current_band_len >= current_band_target) {
                rng_state = rand_hash(rng_state);
                current_band_target = rng_state % (threshold > 0 ? threshold : 1);
                current_band_len = 0;
                is_sorting_band = !is_sorting_band;
            }
            current_band_len++;
            condition = is_sorting_band;
        }

        // Spatial Masking
        if (masking > 0 && mask_data) {
            // If the mask pixel is 0, we do not sort this pixel.
            if (mask_data[col] == 0) {
                condition = false;
            }
        }
        
        if (condition) {
            if (segment_start == -1) {
                segment_start = col;
            }
        } else {
            if (segment_start != -1) {
                int segment_end = col - 1;
                if (segment_end > segment_start) {
                    quickSort(row_data, segment_start, segment_end, sorting_style);
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
            quickSort(row_data, segment_start, segment_end, sorting_style);
        }
    }
}

} // extern "C"
