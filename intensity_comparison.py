import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_cdf_from_histogram(image, equalized_image, bins=256, range=(0, 256)):
    hist1, bins1 = np.histogram(image.flatten(), 256, [0, 256])
    hist2, bins2 = np.histogram(equalized_image.flatten(), 256, [0, 256])

    cdf1 = np.cumsum(hist1)
    cdf_normalized1 = cdf1 / cdf1[-1]

    cdf2 = np.cumsum(hist2)
    cdf_normalized2 = cdf2 / cdf2[-1]

    plt.plot(cdf_normalized1, cdf_normalized2, marker='.', linestyle='-', color='b')
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.title('Intensity comparison')
    plt.grid(True)
    plt.show()

def manual_histogram_equalization(image):
    flat_image = image.flatten()
    hist, bins = np.histogram(flat_image, bins=256, range=[0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')

    equalized_image = cdf_final[flat_image]
    equalized_image = equalized_image.reshape(image.shape)
    return equalized_image

def equalize_histogram(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    equalized_image = manual_histogram_equalization(image)
    plot_cdf_from_histogram(image, equalized_image)

if __name__ == "__main__":
  plt.figure(figsize=(12, 7))
  equalize_histogram('baker.jpg')
  plt.show()
