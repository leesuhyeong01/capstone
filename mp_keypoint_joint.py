import cv2
import mediapipe as mp
import json
import os
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 손 관절 각도 계산을 위한 부모-자식 관절 정의
HAND_PARENTS = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
HAND_CHILDREN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# 각도 계산에 사용할 벡터 쌍
HAND_ANGLE_PAIRS = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (12, 13), (13, 14), (14, 15), (16, 17), (17, 18), (18, 19)]

# 포즈 관절 각도 계산 (어깨-팔꿈치-손목)
POSE_ANGLE_TRIPLES = [(11, 13, 15), (12, 14, 16)]  # 왼쪽, 오른쪽 팔

def compute_angle_between_vectors(v1, v2):
    """두 벡터 사이의 각도를 계산 (도 단위)"""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def compute_angle_between_three_points(p1, p2, p3):
    """세 점으로 이루어진 각도 계산 (p2가 각도의 꼭지점)"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    return compute_angle_between_vectors(v1, v2)

def extract_angles_only(video_path, output_json):
    # 비디오 경로 확인
    if not os.path.exists(video_path):
        print(f"오류: 비디오 파일을 찾을 수 없습니다. 경로: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다. 경로: {video_path}")
        return False
        
    # 비디오 정보 출력
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"비디오 정보: {width}x{height}, {fps}fps, {frame_count}프레임")

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    pose = mp_pose.Pose(static_image_mode=False)

    all_frames_data = []
    frame_idx = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 진행 상황 출력 (100프레임마다)
        if frame_idx % 100 == 0:
            print(f"프레임 처리 중: {frame_idx}/{frame_count}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(frame_rgb)
        hands_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        frame_data = {"frame": frame_idx, "angles": {}}
        has_data = False  # 데이터가 감지되었는지 확인하는 플래그

        # 손 각도
        hand_angles = {"left": {}, "right": {}}
        
        if hands_results.multi_hand_landmarks:
            has_data = True
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                hand_type = "left" if handedness.classification[0].label == "Left" else "right"
                
                # 손의 모든 랜드마크 좌표를 배열로 변환
                joint = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                
                # 각도 계산
                angles = {}
                # 부모와 자식 관절 간의 벡터 계산
                v1 = joint[HAND_PARENTS, :3]  # 부모 관절
                v2 = joint[HAND_CHILDREN, :3]  # 자식 관절
                v = v2 - v1  # 벡터 계산
                
                # 벡터 정규화
                v_norm = np.zeros_like(v)
                for i in range(len(v)):
                    norm = np.linalg.norm(v[i])
                    if norm > 0:
                        v_norm[i] = v[i] / norm
                
                # 각도 계산 (15개의 각도)
                for idx, (i, j) in enumerate(HAND_ANGLE_PAIRS):
                    dot_product = np.clip(np.sum(v_norm[i] * v_norm[j]), -1.0, 1.0)
                    angle = np.degrees(np.arccos(dot_product))
                    # 각도 이름 간소화
                    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
                    if i < 4:  # 엄지
                        joint_name = f"{finger_names[0]}_{i}"
                    elif i < 8:  # 검지
                        joint_name = f"{finger_names[1]}_{i-4}"
                    elif i < 12:  # 중지
                        joint_name = f"{finger_names[2]}_{i-8}"
                    elif i < 16:  # 약지
                        joint_name = f"{finger_names[3]}_{i-12}"
                    else:  # 소지
                        joint_name = f"{finger_names[4]}_{i-16}"
                    
                    angles[joint_name] = float(angle)
                
                hand_angles[hand_type] = angles
                
            frame_data["angles"]["hands"] = hand_angles

        # 포즈 각도 (어깨, 팔꿈치, 손목)
        pose_angles = {}
        if pose_results.pose_landmarks:
            has_data = True
            
            # 포즈 각도 계산 (팔꿈치 각도)
            for idx, (a, b, c) in enumerate(POSE_ANGLE_TRIPLES):
                if all(pose_results.pose_landmarks.landmark[i].visibility > 0.5 for i in [a, b, c]):
                    p1 = [pose_results.pose_landmarks.landmark[a].x, pose_results.pose_landmarks.landmark[a].y, pose_results.pose_landmarks.landmark[a].z]
                    p2 = [pose_results.pose_landmarks.landmark[b].x, pose_results.pose_landmarks.landmark[b].y, pose_results.pose_landmarks.landmark[b].z]
                    p3 = [pose_results.pose_landmarks.landmark[c].x, pose_results.pose_landmarks.landmark[c].y, pose_results.pose_landmarks.landmark[c].z]
                    angle = compute_angle_between_three_points(p1, p2, p3)
                    side = "left" if a == 11 else "right"
                    pose_angles[f"elbow_{side}"] = float(angle)
            
            frame_data["angles"]["pose"] = pose_angles

        # 얼굴 각도 (눈, 입 등의 각도)
        face_angles = {}
        if face_results.multi_face_landmarks:
            has_data = True
            for face_landmarks in face_results.multi_face_landmarks:
                # 눈 각도 (눈썹-눈-뺨)
                # 왼쪽 눈
                left_eye_brow = [face_landmarks.landmark[55].x, face_landmarks.landmark[55].y, face_landmarks.landmark[55].z]
                left_eye = [face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z]
                left_cheek = [face_landmarks.landmark[50].x, face_landmarks.landmark[50].y, face_landmarks.landmark[50].z]
                left_eye_angle = compute_angle_between_three_points(left_eye_brow, left_eye, left_cheek)
                face_angles["left_eye"] = float(left_eye_angle)
                
                # 오른쪽 눈
                right_eye_brow = [face_landmarks.landmark[285].x, face_landmarks.landmark[285].y, face_landmarks.landmark[285].z]
                right_eye = [face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z]
                right_cheek = [face_landmarks.landmark[280].x, face_landmarks.landmark[280].y, face_landmarks.landmark[280].z]
                right_eye_angle = compute_angle_between_three_points(right_eye_brow, right_eye, right_cheek)
                face_angles["right_eye"] = float(right_eye_angle)
                
                # 입 각도 (왼쪽 입꼬리-입 중앙-오른쪽 입꼬리)
                mouth_left = [face_landmarks.landmark[61].x, face_landmarks.landmark[61].y, face_landmarks.landmark[61].z]
                mouth_center = [face_landmarks.landmark[13].x, face_landmarks.landmark[13].y, face_landmarks.landmark[13].z]
                mouth_right = [face_landmarks.landmark[291].x, face_landmarks.landmark[291].y, face_landmarks.landmark[291].z]
                mouth_angle = compute_angle_between_three_points(mouth_left, mouth_center, mouth_right)
                face_angles["mouth"] = float(mouth_angle)
            
            frame_data["angles"]["face"] = face_angles

        if has_data:
            all_frames_data.append(frame_data)
            processed_frames += 1

        frame_idx += 1

    cap.release()
    
    # 결과 확인
    if len(all_frames_data) == 0:
        print("경고: 감지된 랜드마크가 없습니다. 비디오에 사람이 나오는지 확인하세요.")
        return False
        
    print(f"처리 완료: 총 {frame_idx}개 프레임 중 {processed_frames}개에서 랜드마크 감지됨")
    
    # JSON 파일 저장
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_frames_data, f, indent=4)
        print(f"결과가 {output_json}에 저장되었습니다.")
        return True
    except Exception as e:
        print(f"JSON 파일 저장 중 오류 발생: {e}")
        return False

# 사용 예제
video_path = "C:/Users/musek/OneDrive/바탕 화면/capstone/input.mp4"  # 비디오 파일 경로
extract_angles_only(video_path, "output_angles_only.json")
