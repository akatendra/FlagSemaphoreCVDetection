import cv2 as cv
import time
import pose_module as pm

alphabet = {
    (135, 90): 'A',
    (180, 90): 'B',
    (-135, 90): 'C',
    (-90, 90): 'D',
    (90, -45): 'E',
    (90, 0): 'F',
    (90, 45): 'G',
    (180, 135): 'H',
    (-135, 135): 'I',
    (-90, 0): 'J',
    (135, -90): 'K',
    (135, -45): 'L',
    (135, 0): 'M',
    (135, 45): 'N',
    (180, -135): 'O',
    (180, -90): 'P',
    (180, -45): 'Q',
    (180, 0): 'R',
    (180, 45): 'S',
    (-135, -90): 'T',
    (-135, -45): 'U',
    (-90, 45): 'V',
    (-45, 0): 'W',
    (-45, 45): 'X',
    (-135, 0): 'Y',
    (45, 0): 'Z',
    (-90, -45): 'Numerical sign',
    (-135, 45): 'Cancel'
}


def angle_normalizer(angle, angle_gap=22.5):
    angle_out = '?'
    if -angle_gap < angle < 0 or 0 < angle < angle_gap:
        angle_out = 0
    elif -45 - angle_gap < angle < -45 + angle_gap:
        angle_out = -45
    elif -90 - angle_gap < angle < -90 + angle_gap:
        angle_out = -90
    elif -135 - angle_gap < angle < -135 + angle_gap:
        angle_out = -135
    elif -180 <= angle < -180 + angle_gap or 180 - angle_gap < angle <= 180:
        angle_out = 180
    elif 45 - angle_gap < angle < 45 + angle_gap:
        angle_out = 45
    elif 90 - angle_gap < angle < 90 + angle_gap:
        angle_out = 90
    elif 135 - angle_gap < angle < 135 + angle_gap:
        angle_out = 135
    return angle_out


cap = cv.VideoCapture('videos/1.mp4')
p_time = 0
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        #     print(lm_list[14])
        # cv.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255),
        #           cv.FILLED)

        # Right hand
        right_hand = detector.find_straight_angle(img, 14, 16)

        # Left hand
        left_hand = detector.find_straight_angle(img, 13, 15)
        print((right_hand, left_hand))

        # Normalize angles
        right_hand = angle_normalizer(right_hand)
        left_hand = angle_normalizer(left_hand)

        key = (right_hand, left_hand)
        print(key)
        if key in alphabet:
            letter = alphabet[(right_hand, left_hand)]
        else:
            letter = ''
        print(letter)

    cv.rectangle(img, (1110, 450), (1260, 650), (0, 255, 0), cv.FILLED)
    if letter == 'Numerical sign':
        letter = 'Num'
        cv.putText(img, str(letter), (1140, 600), cv.FONT_HERSHEY_PLAIN, 3,
               (255, 0, 0), 2)
    elif letter == 'Cancel':
        cv.putText(img, str(letter), (1140, 600), cv.FONT_HERSHEY_PLAIN, 2,
                   (255, 0, 0), 2)
    else:
        cv.putText(img, str(letter), (1140, 600), cv.FONT_HERSHEY_PLAIN, 10,
                   (255, 0, 0), 5)
    # FPS calculation
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    # FPS calculation

    # FPS output over image
    cv.putText(img, f'FPS: {str(int(fps))}', (20, 50), cv.FONT_HERSHEY_PLAIN,
               3,
               (255, 0, 0), 3)
    # FPS output over image

    # Output
    cv.imshow("Image", img)
    cv.waitKey(1)
