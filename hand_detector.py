import cv2
import mediapipe as mp
import numpy as np


# Funktion, um die nächste nicht-Null-Pixel zu finden
def find_closest_non_zero(matrix, dist_transform, start_point):
    min_dist = float('inf')
    closest_pixel = None
    start_x, start_y = start_point

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x] != 0:
                dist = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_pixel = (x, y)
    return closest_pixel, min_dist


def draw_hand(image, px_cm):

    height, width, _ = image.shape
    # MediaPipe Hand-Tracking initialisieren
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Bild konvertieren
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Handerkennung durchführen
    results = hands.process(image_rgb)

    lm = results.multi_hand_landmarks

    # Ergebnisse überprüfen
    if lm:
        # Canny-Kantendetektor anwenden
        hand_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(hand_gray, 100, 200)
        hand_landmarks_list = None

        # Distance Transform anwenden
        dist_transform = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)

        for hand_landmarks in lm:
            # Handmarkierungen auf dem Originalbild zeichnen
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_landmarks_list = hand_landmarks.landmark

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            if idx == 2:  # THUMB_MCP
                start_point = (hand_landmarks.x * width, hand_landmarks.y * height)
                closest_pixel, min_dist = find_closest_non_zero(edges, dist_transform, start_point)
                # print(closest_pixel, min_dist * 2.05 / px_cm)  # todo: check calculation
                print("THUMB_MCP: " + str(min_dist * 2.05 / px_cm))
            elif idx == 3:  # THUMB_IP
                start_point = (hand_landmarks.x * width, hand_landmarks.y * height)
                closest_pixel, min_dist = find_closest_non_zero(edges, dist_transform, start_point)
                print("THUMB_IP: " + str(min_dist * 2.05 / px_cm))
            elif idx == 6:  # INDEX_FINGER_PIP
                start_point = (hand_landmarks.x * width, hand_landmarks.y * height)
                closest_pixel, min_dist = find_closest_non_zero(edges, dist_transform, start_point)
                print("INDEX_FINGER_PIP: " + str(min_dist * 2.05 / px_cm))
            elif idx == 7:  # INDEX_FINGER_DIP
                start_point = (hand_landmarks.x * width, hand_landmarks.y * height)
                closest_pixel, min_dist = find_closest_non_zero(edges, dist_transform, start_point)
                print("INDEX_FINGER_DIP: " + str(min_dist * 2.05 / px_cm))

        # Kanten auf Originalbild zeichnen
        edge_image = image.copy()
        edge_image[edges != 0] = [0, 255, 0]

        # Ausgerichtetes Bild speichern
        output_path = './hand_detected_image.jpeg'
        cv2.imwrite(output_path, edge_image)
        print(f"Das Bild mit erkannter Hand wurde gespeichert: {output_path}")

        # Das ausgerichtete Bild anzeigen
        cv2.imshow("Hand Erkanntes Bild", edge_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Keine Hand erkannt")


if __name__ == '__main__':
    # Bild einlesen
    image_path = './warped_image.jpeg'
    image_path = './data/IMG_20240708_125529.jpg'
    image = cv2.imread(image_path)
    draw_hand(image, 30.11111)