import mediapipe as mp
import cv2
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection

# 비디오 경로
video_path = 'E:/workspace/video_sample/room2.mp4'

# 이미지 저장 폴더
try:
    if not os.path.exists('./capture'):
        os.makedirs('./capture')
except OSError:
    print('Error')
directory = './capture/'

# 이미지 이름 저장
video_name = video_path.split('/')[-1].replace('.', '_')
print(video_name)
try:
    video = cv2.VideoCapture(int(video_path))
except:
    video = cv2.VideoCapture(video_path)

with mp_pose.Pose(
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    ROI = []
    frame = 0
    ROI_ttl = 0
    is_inside = False
    while video.isOpened():
        success, original_image = video.read()
        if not success:
            print("video.read fail.")
            break
        image = cv2.resize(original_image, dsize=(1280, 720))
        image_height, image_width, _ = image.shape
        cut_image = image.copy()

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
        L_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
        R_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

        R_SHOULDER_coord = [int(R_shoulder.x * image_width), int(R_shoulder.y * image_height)]
        L_SHOULDER_coord = [int(L_shoulder.x * image_width), int(L_shoulder.y * image_height)]
        ROI_PADDING = abs(R_SHOULDER_coord[0] - L_SHOULDER_coord[0])  # // 3
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

        # 이미지에 포즈 주석을 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        if (time.time() - ROI_ttl) > 3:
            ROI_cut_image = cut_image[outer_ROI[1]:outer_ROI[1] + outer_ROI[3],
                            outer_ROI[0]:outer_ROI[0] + outer_ROI[2]]
            cv2.imshow('ROI image', ROI_cut_image)
            path = directory + video_name + str(frame) + '.jpg'
            cv2.imwrite(path, ROI_cut_image)

        cv2.rectangle(
            image,
            (outer_ROI[0], outer_ROI[1]),
            (outer_ROI[0] + outer_ROI[2], outer_ROI[1] + outer_ROI[3]),
            (255, 0, 0),
            2)
        cv2.imshow('Image', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame += 1
    cv2.destroyAllWindows()
    video.release()
