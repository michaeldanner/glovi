import calibrate2
import cv2

if __name__ == '__main__':
    # Bild einlesen
    image_path = './data/IMG_20240708_130727.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Größe des Gitters (8x18)
    pattern_size = (18, 8)
    img, px = calibrate2.calibrate_circles(image, pattern_size)
    print(px)