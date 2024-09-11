import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_histogram(image, plot_no, title):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    plt.subplot(2, 3, (plot_no + 3))
    plt.plot(hist, color='black')
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 56])

def calculate_cdf(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    plt.figure(figsize=(10, 5))
    return cdf_normalized

def histogram_specification(input_image, target_image):
    cdf_input = calculate_cdf(input_image)
    cdf_target = calculate_cdf(target_image)
    mapping = np.zeros(256)
    for i in range(256):
        # Find the pixel value in the target image with the closest CDF value
        diff = np.abs(cdf_input[i] - cdf_target)
        mapping[i] = np.argmin(diff)

    output_image = cv2.LUT(input_image, mapping.astype('uint8'))
    return output_image

def show_images_and_histograms(input_image, target_image, matched_image):
    plt.figure(figsize=(13, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(target_image, cmap='gray')
    plt.title('Target Image')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(matched_image, cmap='gray')
    plt.title('Matched Image')
    plt.axis('off')
    
    draw_histogram(input_image, 1, 'Input Image Histogram')
    draw_histogram(target_image, 2, 'Target Image Histogram')
    draw_histogram(matched_image, 3, 'Matched Image Histogram')
    plt.show()

if __name__ == "__main__":
  input_image_path = 'baker.jpg'
  target_image_path = 'wallpaper.jpg'
  input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
  target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
  if input_image is not None and target_image is not None:
      matched_image = histogram_specification(input_image, target_image)
      show_images_and_histograms(input_image, target_image, matched_image)
  else:
      print("Error: Unable to load input or target image.")