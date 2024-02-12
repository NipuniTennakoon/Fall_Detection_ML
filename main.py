import cv2
import mediapipe as mp

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Import the names of coco names
classNames = []
classFile = 'cocoNames'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose  # Import pose estimation models

# video_path = 'Video.mp4'
cap = cv2.VideoCapture(0)

# Variables for fall detection
fall_count = 0
fall_threshold = 1  # Number of consecutive frames to detect a fall
is_fallen = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.55)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if len(classNames) > 0 and classId - 1 < len(classNames):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # To identify person
                    if classId == 1:
                        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)  # Make detection
                        image.flags.writeable = True  # Recolor back to BGR
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        # Fall detection logic
                        if results.pose_landmarks is not None:
                            landmarks = results.pose_landmarks.landmark

                            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y

                            if left_ankle < hip and right_ankle < hip:
                                if not is_fallen:
                                    fall_count += 1
                                    if fall_count >= fall_threshold:
                                        print("Fall detected!")
                                        # Add your fall handling code here

                                    is_fallen = True
                            else:
                                is_fallen = False
                                fall_count = 0

                            # Render detection
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                             circle_radius=2),
                                                      mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2,
                                                                             circle_radius=2))

                        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
