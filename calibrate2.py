import cv2
import numpy as np


def calibrate_circles(image, pattern_size, show=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Kreismuster erkennen
    ret, centers = cv2.findCirclesGrid(gray, pattern_size, cv2.CALIB_CB_SYMMETRIC_GRID)

    if ret:
        # Kreise auf dem Originalbild zeichnen
        cv2.drawChessboardCorners(image, pattern_size, centers, ret)

        # Eckpunkte des erkannten Gitters ermitteln
        top_left = centers[0][0]
        top_right = centers[17][0]
        bottom_left = centers[-18][0]
        bottom_right = centers[-1][0]

        # Zielkoordinaten für die Transformation definieren
        width = int(np.linalg.norm(top_right - top_left))
        height = int(np.linalg.norm(bottom_left - top_left))
        ratio = (width/pattern_size[0]) / (height/pattern_size[1])
        print('ratio:', ratio)
        height = height * ratio
        dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype="float32")
        src_points = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")

        px_cm = width/pattern_size[0]*2

        height, width = gray.shape

        # Transformationsmatrix berechnen und anwenden
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, M, (width, height))

        # Ausgerichtetes Bild speichern
        output_path = './warped_image.jpeg'
        cv2.imwrite(output_path, warped)
        print(f"Das ausgerichtete Bild wurde gespeichert: {output_path}")

        if show:  # Das ausgerichtete Bild anzeigen
            cv2.imshow("Ausgerichtetes Bild", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return warped, px_cm

    else:
        print("Kein Schachbrettmuster erkannt")

        return None, None


if __name__ == '__main__':
    # Bild einlesen
    image_path = './img04.jpeg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Größe des Gitters (8x18)
    pattern_size = (18, 8)
    img, shape = calibrate_circles(image, pattern_size, True)
