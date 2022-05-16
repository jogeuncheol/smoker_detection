# Google MediaPipe
# Pose tutorial
# body landmarks
# 0 : nose
# 7, 8: left_ear, right_ear
# 9 : mouth_left
# 10: mouth_right
# right hand 16: right_wrist 18:right_pinky 20: right_index 22: right_thumb
# left hand 15: left_wrist 17: left_pinky 19: left_index 21: left_thumb
# 11: left_shoulder 12: right_shoulder

import cv2
import numpy as np
import mediapipe as mp
import time
from queue import Queue
from collections import deque
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

bg = cv2.createBackgroundSubtractorMOG2(history=42, varThreshold=16, detectShadows=False)
bg2 = cv2.createBackgroundSubtractorMOG2(history=42, varThreshold=16, detectShadows=False)
kg = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=64, detectShadows=True)
# kg = cv2.createBackgroundSubtractorKNN(history=42, dist2Threshold=16, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

video_path = "E:/workspace/video_sample/E_smoke.mp4"
try:
    video = cv2.VideoCapture(int(video_path))
except:
    video = cv2.VideoCapture(video_path)

with mp_pose.Pose(
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    outer_ROI = []
    is_inside = False
    ROI_ttl = 0
    frame = 0
    queue = Queue(3)
    dq = deque()
    while video.isOpened():
        success, ori_image = video.read()
        if not success:
            print("video.read fail.")
            continue
        image = ori_image.copy()
        image = cv2.resize(image, dsize=(960, 480))
        image_height, image_width, _ = image.shape
        gray = cv2.resize(ori_image, dsize=(960, 480))
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        bg_mask = bg.apply(gray, 0, 0.00001)

        # 성능을 향상시키려면 선택적으로 이미지를 참조로 전달할 수 없는 것으로 표시합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if not results.pose_landmarks:
            continue
        # 포즈 랜드마크
        Nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        R_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        L_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        R_mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
        L_mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
        R_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        L_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        L_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        R_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]

        # ear-nose 머리의 방향 계산
        # 좌측 : 0, 우측 : 1, 양측 : -1
        nose_x = Nose.x * image_width
        l_ear_x = L_ear.x * image_width
        r_ear_x = R_ear.x * image_width
        head_direction = 0
        if nose_x < l_ear_x and nose_x < r_ear_x:
            head_direction = 0
        elif nose_x > l_ear_x and nose_x > r_ear_x:
            head_direction = 1
        else:
            head_direction = -1

        # 오른손 말단과 오른쪽 입가
        cv2.drawMarker(
            image,
            (int(R_hand.x * image_width), int(R_hand.y * image_height)),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=42)
        cv2.drawMarker(
            image,
            (int(R_mouth.x * image_width), int(R_mouth.y * image_height)),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=42)

        R_SHOULDER_coord = [int(R_shoulder.x * image_width), int(R_shoulder.y * image_height)]
        L_SHOULDER_coord = [int(L_shoulder.x * image_width), int(L_shoulder.y * image_height)]
        ROI_PADDING = abs(R_SHOULDER_coord[0] - L_SHOULDER_coord[0])# // 3
        # ROI
        if not is_inside:
            outer_ROI = [
                int(Nose.x * image_width) - ROI_PADDING,
                int(Nose.y * image_height) - ROI_PADDING,
                ROI_PADDING * 2,
                ROI_PADDING * 2
            ]
            is_inside = True
            ROI_ttl = time.time()
        elif is_inside:
            if outer_ROI[0] > int(Nose.x * image_width) \
                    or outer_ROI[0] + outer_ROI[2] < int(Nose.x * image_width):
                is_inside = False
                ROI_ttl = time.time()

        cv2.rectangle(
            image,
            (outer_ROI[0], outer_ROI[1]),
            (outer_ROI[0] + outer_ROI[2], outer_ROI[1] + outer_ROI[3]),
            (255, 0, 0),
            2)

        # 이미지에 포즈 주석을 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # GMM 적용 시점. ROI 생존시간이 5 이상일때 적용
        th_image = []
        if (time.time() - ROI_ttl) > 5:
            bg2_mask = kg.apply(gray, 0, 0.005)
            sub_mask = cv2.bitwise_and(bg_mask, bg2_mask)
            # cv2.imshow('BG_sub 0.005', bg2_mask)
            # crop_image == original_image
            crop_image = image[outer_ROI[1]:outer_ROI[1]+outer_ROI[3], outer_ROI[0]:outer_ROI[0]+outer_ROI[2]]
            # crop_image_binary == crop sub_mask(bg - bg2)
            crop_image_binary = sub_mask[outer_ROI[1]:outer_ROI[1]+outer_ROI[3], outer_ROI[0]:outer_ROI[0]+outer_ROI[2]]
            cv2.imshow('crop_image_binary', crop_image_binary)

            ret, th_image = cv2.threshold(crop_image_binary, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
            th_image = cv2.medianBlur(th_image, ksize=3)
            # cv2.imshow('th_image', th_image)

            # smoke detector
            conv_image = cv2.resize(th_image, dsize=(100, 100))
            retn, conv_image = cv2.threshold(conv_image, thresh=125, maxval=255, type=cv2.THRESH_BINARY)
            # cv2.imshow('100x100', conv_image)
            # 5 x 5 영역씩 훑어보면서 밀도 계산 후 위치 추정을 통해 연기인지 구분
            smoke_map = []
            for i in range(0, 100, 5):
                line = []
                for j in range(0, 100, 5):
                    count = 0
                    for k in range(5):
                        for l in range(5):
                            if conv_image[i + k][j + l] > 125:
                                count += 1
                    if count > 12:
                        line.append(255)
                    else:
                        line.append(0)
                smoke_map.append(line)
            np_smoke_map_image = np.array(smoke_map).astype(np.uint8)
            resize_smoke_map = cv2.resize(np_smoke_map_image, dsize=(outer_ROI[2], outer_ROI[3]))
            cv2.imshow('smoke_map', resize_smoke_map)

            # 머리 방향에 따른 mask nose_x 대신 어깨 y 축 기준으로 수정해야 할듯
            # smoke_map_mask_255 = np.ones((outer_ROI[2], outer_ROI[3]), dtype=np.uint8) * 255
            smoke_map_mask = np.ones((outer_ROI[2], outer_ROI[3]), dtype=np.uint8) * 255
            direction_box = np.array(
                [[0, 0],
                 [nose_x - outer_ROI[0], 0],
                 [nose_x - outer_ROI[0], outer_ROI[3]-1],
                 [0, outer_ROI[3]-1]], dtype=np.int32
            )
            if head_direction == 0 or head_direction == 1:
                cv2.fillPoly(smoke_map_mask, [direction_box], color=(0, 0, 0))
            if head_direction == 0:
                smoke_map_mask = cv2.flip(smoke_map_mask, 1)
            cv2.imshow('smoke_map_mask', smoke_map_mask)
            and_smoke_map = cv2.bitwise_and(resize_smoke_map, smoke_map_mask)
            cv2.imshow('smoke_map_masking', and_smoke_map)

            contours, _ = cv2.findContours(resize_smoke_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 코, 입 좌표를 포함하는 contour 찾기
            Nose_coord = [Nose.x * image_width, Nose.y * image_height]
            R_mouth_coord = [R_mouth.x * image_width, R_mouth.y * image_height]
            L_mouth_coord = [L_mouth.x * image_width, L_mouth.y * image_height]
            find_smoke_contour = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if x < Nose_coord[0] - outer_ROI[0] < x+w and \
                    y < Nose_coord[1] - outer_ROI[1] < y+h:
                    find_smoke_contour.append(cnt)
                if x < R_mouth_coord[0] - outer_ROI[0] < x+w and \
                    y < R_mouth_coord[1] - outer_ROI[1] < y+h:
                    find_smoke_contour.append(cnt)
                if x < L_mouth_coord[0] - outer_ROI[0] < x+w and \
                    y < L_mouth_coord[1] - outer_ROI[1] < y+h:
                    find_smoke_contour.append(cnt)

            # cv2.drawContours(crop_image, contours, -1, (0, 255, 0), 3)
            # Nose_coord = [Nose.x * image_width, Nose.y * image_height]
            # crop_nose_x = Nose_coord[0] - outer_ROI[0]
            # crop_nose_y = Nose_coord[1] - outer_ROI[1]
            # shortest_dist = []
            # contour_arr = []
            # for cnt in contours:
            #     x, y, w, h = cv2.boundingRect(cnt)
            #     if not shortest_dist:
            #         shortest_dist.append(x)
            #         shortest_dist.append(y)
            #         contour_arr.append(cnt)
            #     else:
            #         dist1 = (crop_nose_x - shortest_dist[0])**2 + (crop_nose_y - shortest_dist[1])**2
            #         dist2 = (crop_nose_x - x)**2 + (crop_nose_y - y)**2
            #         if dist1 > dist2:
            #             shortest_dist[0] = x
            #             shortest_dist[1] = y
            #             contour_arr.pop()
            #             contour_arr.append(cnt)
            cv2.drawContours(crop_image, contours, -1, (0, 255, 0), 2)
            cv2.drawContours(crop_image, find_smoke_contour, -1, (255, 0, 0), 2)

            cv2.imshow('crop_image', crop_image)

            # path = './data/' + str(frame) + '.jpg'
            # cv2.imwrite(path, th_image)

        cv2.imshow('MediaPipe Pose', image)
        # cv2.imshow('BG_sub 0.00001', bg_mask)
        cv2.waitKey(0)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame += 1

video.release()
