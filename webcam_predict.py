import cv2
import mediapipe as mp
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import tempfile
import time
from collections import deque

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

def extract_angles_from_frame(frame, face_mesh, hands, pose):
    """단일 프레임에서 관절 각도를 추출하여 반환"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(frame_rgb)
    hands_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    angles = {}
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
            angles_dict = {}
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
                
                angles_dict[joint_name] = float(angle)
            
            hand_angles[hand_type] = angles_dict
            
        angles["hands"] = hand_angles

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
        
        angles["pose"] = pose_angles

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
        
        angles["face"] = face_angles

    return angles, has_data, (face_results, hands_results, pose_results)

def extract_features_from_angles(angles_data):
    """
    각도 데이터에서 특성 벡터 추출
    """
    features = []
    
    # 손 각도 추출
    hands = angles_data.get("hands", {})
    for hand_type in ["left", "right"]:
        hand = hands.get(hand_type, {})
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            for i in range(4):  # 각 손가락당 최대 4개의 관절
                key = f"{finger}_{i}"
                features.append(hand.get(key, 0.0))
    
    # 포즈 각도 추출
    pose = angles_data.get("pose", {})
    for key in ["elbow_left", "elbow_right"]:
        features.append(pose.get(key, 0.0))
    
    # 얼굴 각도 추출
    face = angles_data.get("face", {})
    for key in ["left_eye", "right_eye", "mouth"]:
        features.append(face.get(key, 0.0))
    
    return features

def normalize_sequence_length(sequence, target_length=100):
    """
    시퀀스 길이를 정규화하는 함수
    """
    seq_len = len(sequence)
    
    # 시퀀스가 너무 짧으면 패딩 추가
    if seq_len < target_length:
        # 마지막 프레임을 복제하여 패딩
        padding = [sequence[-1]] * (target_length - seq_len)
        normalized_seq = sequence + padding
    # 시퀀스가 너무 길면 균등하게 샘플링
    elif seq_len > target_length:
        indices = np.linspace(0, seq_len-1, target_length, dtype=int)
        normalized_seq = [sequence[i] for i in indices]
    else:
        normalized_seq = sequence
        
    return normalized_seq

def real_time_webcam_prediction(model_path, label_encoder_path, scaler_path, sequence_length=100, buffer_size=30):
    """
    웹캠을 사용하여 실시간으로 제스처를 인식하는 함수
    
    Args:
        model_path: 저장된 모델 경로
        label_encoder_path: 레이블 인코더 경로
        scaler_path: 스케일러 경로
        sequence_length: 정규화할 시퀀스 길이
        buffer_size: 프레임 버퍼 크기 (몇 개의 최근 프레임을 유지할지)
    """
    # 모델 및 관련 파일 로드
    try:
        model = tf.keras.models.load_model(model_path)
        
        with open(label_encoder_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            classes = np.array(label_data['classes'])
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        print("모델과 관련 파일 로드 완료")
    except Exception as e:
        print(f"모델 또는 관련 파일 로드 중 오류 발생: {e}")
        return
    
    # 웹캠 설정
    cap = cv2.VideoCapture(0)  # 0: 기본 웹캠
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    # 창 설정
    window_name = "실시간 제스처 인식"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # MediaPipe 솔루션 초기화
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    pose = mp_pose.Pose(static_image_mode=False)
    
    # 프레임 버퍼 초기화
    feature_buffer = deque(maxlen=buffer_size)
    prediction_cooldown = 0  # 예측 사이의 쿨다운 프레임 (성능 최적화)
    current_prediction = "대기 중..."
    current_confidence = 0.0
    
    # FPS 측정 변수
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    print("웹캠 처리 시작 (종료하려면 'q' 키를 누르세요)")
    
    # 메인 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break
            
        # 좌우 반전 (자연스러운 거울 효과)
        frame = cv2.flip(frame, 1)
        
        # 프레임 처리
        angles_data, has_data, (face_results, hands_results, pose_results) = extract_angles_from_frame(
            frame, face_mesh, hands, pose
        )
        
        # 랜드마크가 감지된 경우에만 특성 추출
        if has_data:
            features = extract_features_from_angles(angles_data)
            feature_buffer.append(features)
        
        # 복사본에 랜드마크 시각화
        annotated_frame = frame.copy()
        
        # 손 랜드마크 그리기
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # 포즈 랜드마크 그리기
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # 얼굴 메시 랜드마크 그리기 (간소화된 버전)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        # 충분한 프레임이 모였고 쿨다운이 끝났을 때 예측 수행
        if len(feature_buffer) >= buffer_size and prediction_cooldown <= 0:
            # 특성 시퀀스 정규화
            normalized_features = normalize_sequence_length(list(feature_buffer), target_length=sequence_length)
            
            # 특성 스케일링
            X = np.array([normalized_features])
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(original_shape)
            
            # 예측
            if len(classes) > 2:
                predictions = model.predict(X_scaled, verbose=0)
                predicted_class_index = np.argmax(predictions, axis=1)
                current_confidence = np.max(predictions, axis=1)[0]
            else:
                predictions = model.predict(X_scaled, verbose=0)
                predicted_class_index = (predictions > 0.5).astype(int).flatten()
                current_confidence = predictions.flatten()[0]
            
            current_prediction = classes[predicted_class_index[0]]
            prediction_cooldown = 5  # 5프레임마다 예측
        else:
            prediction_cooldown -= 1
        
        # FPS 계산
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:  # 1초마다 FPS 업데이트
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # 예측 결과 표시
        result_text = f"제스처: {current_prediction} ({current_confidence:.2f})"
        cv2.putText(
            annotated_frame, 
            result_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0) if current_confidence > 0.7 else (0, 165, 255), 
            2
        )
        
        # FPS 및 버퍼 상태 표시
        fps_text = f"FPS: {fps:.1f}, 버퍼: {len(feature_buffer)}/{buffer_size}"
        cv2.putText(
            annotated_frame,
            fps_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )
        
        # 프레임 표시
        cv2.imshow(window_name, annotated_frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("웹캠 처리 종료")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='비디오 또는 웹캠에서 제스처 인식')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch', 'ui', 'webcam'],
                        help='실행 모드 (single: 단일 비디오, batch: 폴더 내 모든 비디오, ui: 비디오 시각화, webcam: 실시간 웹캠)')
    parser.add_argument('--video', type=str, help='입력 비디오 파일 경로 (single, ui 모드)')
    parser.add_argument('--dir', type=str, help='비디오 디렉토리 경로 (batch 모드)')
    parser.add_argument('--model', type=str, default='lstm_model.h5', help='모델 파일 경로')
    parser.add_argument('--labels', type=str, default='label_encoder.json', help='레이블 인코더 파일 경로')
    parser.add_argument('--scaler', type=str, default='scaler.pkl', help='스케일러 파일 경로')
    parser.add_argument('--buffer', type=int, default=30, help='웹캠 모드에서 프레임 버퍼 크기')
    
    args = parser.parse_args()
    
    # 모델 및 관련 파일 경로 확인
    model_path = args.model
    label_encoder_path = args.labels
    scaler_path = args.scaler
    
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다. 경로: {model_path}")
        return
    
    if not os.path.exists(label_encoder_path):
        print(f"오류: 레이블 인코더 파일을 찾을 수 없습니다. 경로: {label_encoder_path}")
        return
    
    if not os.path.exists(scaler_path):
        print(f"오류: 스케일러 파일을 찾을 수 없습니다. 경로: {scaler_path}")
        return
    
    # 모드에 따른 처리
    if args.mode == 'webcam':
        # 웹캠 모드 실행
        real_time_webcam_prediction(
            model_path, label_encoder_path, scaler_path, buffer_size=args.buffer
        )
    elif args.mode == 'single':
        if not args.video:
            print("오류: --video 인자가 필요합니다.")
            return
            
        if not os.path.exists(args.video):
            print(f"오류: 비디오 파일을 찾을 수 없습니다. 경로: {args.video}")
            return
            
        predicted_class, confidence = predict_from_video(
            args.video, model_path, label_encoder_path, scaler_path
        )
        
        if predicted_class is not None:
            print(f"최종 예측 결과: {predicted_class} (확률: {confidence:.4f})")
    
    elif args.mode == 'batch':
        if not args.dir:
            print("오류: --dir 인자가 필요합니다.")
            return
            
        if not os.path.isdir(args.dir):
            print(f"오류: 디렉토리를 찾을 수 없습니다. 경로: {args.dir}")
            return
            
        batch_process_videos(
            args.dir, model_path, label_encoder_path, scaler_path
        )
    
    elif args.mode == 'ui':
        if not args.video:
            print("오류: --video 인자가 필요합니다.")
            return
            
        if not os.path.exists(args.video):
            print(f"오류: 비디오 파일을 찾을 수 없습니다. 경로: {args.video}")
            return
            
        process_video_with_ui(
            args.video, model_path, label_encoder_path, scaler_path
        )

# 이전 함수들 (predict_from_video, process_video_with_ui, batch_process_videos)은 
# 그대로 사용하되, 필요한 경우에만 호출합니다.

if __name__ == "__main__":
    main()
