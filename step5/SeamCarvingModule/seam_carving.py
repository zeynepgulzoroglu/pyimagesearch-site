import numpy as np
import cv2

def compute_energy_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = np.abs(grad_x) + np.abs(grad_y)
    return energy_map

def remove_seam(image, energy_map):
    h, w = energy_map.shape
    seam = np.zeros(h, dtype=int)
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, h):
        for j in range(w):
            if j == 0:
                min_energy = min(M[i-1, j], M[i-1, j+1])
                backtrack[i, j] = np.argmin([M[i-1, j], M[i-1, j+1]])
            elif j == w - 1:
                min_energy = min(M[i-1, j-1], M[i-1, j])
                backtrack[i, j] = np.argmin([M[i-1, j-1], M[i-1, j]])
            else:
                min_energy = min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])
                backtrack[i, j] = np.argmin([M[i-1, j-1], M[i-1, j], M[i-1, j+1]])

            M[i, j] += min_energy

    seam_idx = np.argmin(M[-1])
    seam[h-1] = seam_idx
    for i in range(h-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]

    return seam

def remove_seam_from_image(image, seam):
    h, w = image.shape[:2]
    output = np.zeros((h, w-1, 3), dtype=np.uint8)
    for i in range(h):
        j = seam[i]
        output[i, :, :] = np.delete(image[i, :, :], j, axis=0)
    return output

def seam_carving(image, num_seams):
    for _ in range(num_seams):
        energy_map = compute_energy_map(image)
        seam = remove_seam(image, energy_map)
        image = remove_seam_from_image(image, seam)
    return image

image = cv2.imread("castle.png")

num_seams_to_remove = 50
result_image = seam_carving(image, num_seams_to_remove)

cv2.imwrite("result.jpg", result_image)