import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def improved_watershed_segmentation(image):
    # Konversi gambar ke RGB jika grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Langkah 1: Transformasi ruang warna (ke ruang warna Lab)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Langkah 2: Perhitungan gradien morfologi
    kernel = np.ones((3, 3), np.uint8)
    gradient_l = cv2.morphologyEx(l_channel, cv2.MORPH_GRADIENT, kernel)
    gradient_a = cv2.morphologyEx(a_channel, cv2.MORPH_GRADIENT, kernel)
    gradient_b = cv2.morphologyEx(b_channel, cv2.MORPH_GRADIENT, kernel)
    
    # Kombinasikan gradien dari setiap channel
    gradient = cv2.addWeighted(gradient_l, 0.4, gradient_a, 0.3, 0)
    gradient = cv2.addWeighted(gradient, 1, gradient_b, 0.3, 0)
    
    # Langkah 3: Rekonstruksi citra gradien
    # Dilakukan dengan opening dan closing morfologi
    opening = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    # Langkah 4: Ekstraksi penanda menggunakan thresholding Otsu
    ret, markers = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Langkah 5: Kalibrasi gradien
    # Menggunakan citra biner untuk mengkalibrasi gradien asli
    calibrated_gradient = cv2.bitwise_and(gradient, gradient, mask=markers.astype(np.uint8))
    
    # Pastikan nilai minimum untuk watershed
    sure_bg = cv2.dilate(markers.astype(np.uint8), kernel, iterations=3)
    dist_transform = cv2.distanceTransform(markers.astype(np.uint8), cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Mencari area yang tidak diketahui
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Penandaan komponen
    ret, markers_watershed = cv2.connectedComponents(sure_fg)
    markers_watershed = markers_watershed + 1
    markers_watershed[unknown == 255] = 0
    
    # Langkah 6: Segmentasi watershed
    markers_watershed = cv2.watershed(image, markers_watershed)
    
    # Tandai batas dengan warna merah
    image[markers_watershed == -1] = [0, 0, 255]
    
    return image, markers_watershed, gradient, closing, sure_fg

# Tambahkan fungsi eksperimen tambahan untuk tugas
def experiment_with_different_color_spaces(image):
    """
    Membandingkan hasil segmentasi watershed pada ruang warna berbeda.
    """
    # Ruang warna RGB
    rgb_result, _, _, _, _ = improved_watershed_segmentation(image)
    
    # Ruang warna HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_result, _, _, _, _ = improved_watershed_segmentation(hsv)
    hsv_result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)
    
    # Ruang warna YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb_result, _, _, _, _ = improved_watershed_segmentation(ycrcb)
    ycrcb_result = cv2.cvtColor(ycrcb_result, cv2.COLOR_YCrCb2BGR)
    
    return rgb_result, hsv_result, ycrcb_result

def evaluate_segmentation(segmented_image, ground_truth):
    """
    Mengevaluasi hasil segmentasi dengan metrik seperti IoU (Intersection over Union).
    """
    # Konversi ke biner
    segmented_binary = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    ret, segmented_binary = cv2.threshold(segmented_binary, 1, 255, cv2.THRESH_BINARY)
    
    # Menghitung IoU
    intersection = np.logical_and(segmented_binary, ground_truth)
    union = np.logical_or(segmented_binary, ground_truth)
    iou = np.sum(intersection) / np.sum(union)
    
    return iou