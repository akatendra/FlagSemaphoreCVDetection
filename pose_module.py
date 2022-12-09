import cv2 as cv
import mediapipe as mp
import time
import math


class PoseDetector:

    def __init__(self, static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        # Configuration
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        # Configuration

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_image_mode,
                                      self.model_complexity,
                                      self.smooth_landmarks,
                                      self.enable_segmentation,
                                      self.smooth_segmentation,
                                      self.min_detection_confidence,
                                      self.min_tracking_confidence)
        self.results = None
        self.lm_list = []

    def find_pose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                # The color is in (B, G, R) format
                # https://stackoverflow.com/questions/69240807/how-to-change-colors-of-the-tracking-points-and-connector-lines-on-the-output-vi
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_draw.DrawingSpec(
                                                color=(255, 0, 0), thickness=2,
                                                circle_radius=2),
                                            self.mp_draw.DrawingSpec(
                                                color=(0, 255, 0), thickness=2,
                                                circle_radius=2))
        return img

    def find_position(self, img, draw=True):
        self.lm_list = []
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(idx, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([idx, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv.circle(img, (x1, y1), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def find_straight_angle(self, img, p1, p2, draw=True):
        # Get the landmarks
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]

        # Calculate the Angle
        angleR = math.atan2(y2 - y1, x2 - x1)
        angleD = round(math.degrees(angleR), 1)

        # Draw
        if draw:
            cv.circle(img, (x1, y1), 5, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv.circle(img, (x2, y2), 5, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv.putText(img, str(angleD), (x2 - 50, y2 + 50),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angleD


def main():
    cap = cv.VideoCapture('videos/1.mp4')
    p_time = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lm_list = detector.find_position(img)
        if len(lm_list) != 0:
            print(lm_list[14])
            cv.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255),
                      cv.FILLED)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3,
                   (255, 0, 0), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
