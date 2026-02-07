import cv2
import numpy as np

class LandingSpotDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

    def preprocess_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def detect_landing_spots(self):
        processed_image = self.preprocess_image()
        # Using Hough Circle Transform to detect circular landing spots
        circles = cv2.HoughCircles(processed_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=30, minRadius=5, maxRadius=30)
        return circles

    def safety_score(self, circles):
        if circles is None:
            return 0  # No landing spots detected
        safe_count = 0
        total_count = len(circles[0])
        for circle in circles[0]:
            radius = circle[2]
            area = np.pi * (radius ** 2)
            if area > 1000:  # Assuming area > 1000 is a safety indicator
                safe_count += 1
        return (safe_count / total_count) * 100 if total_count > 0 else 0

    def detect_and_score(self):
        circles = self.detect_landing_spots()
        score = self.safety_score(circles)
        return circles, score

# Example usage:
# detector = LandingSpotDetector('path/to/image.jpg')
# landing_spots, safety_score = detector.detect_and_score()