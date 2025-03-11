import cv2
import mediapipe as mp
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import tempfile

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

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

def extract_angles_from_video(video_path):
    """비디오에서 관절 각도를 추출하여 JSON 형식으로 반환"""
    
    # 비디오 경로 확인
    if not os.path.exists(video_path):
        print(f"오류: 비디오 파일을 찾을 수 없습니다. 경로: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다. 경로: {video_path}")
        return None
        
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
        return None
        
    print(f"처리 완료: 총 {frame_idx}개 프레임 중 {processed_frames}개에서 랜드마크 감지됨")
    
    return all_frames_data

def extract_features_from_frames(frames_data):
    """
    프레임 데이터에서 특성 벡터 추출
    """
    frame_features = []
    for frame in frames_data:
        features = []
        angles = frame.get("angles", {})
        
        # 손 각도 추출
        hands = angles.get("hands", {})
        for hand_type in ["left", "right"]:
            hand = hands.get(hand_type, {})
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                for i in range(4):  # 각 손가락당 최대 4개의 관절
                    key = f"{finger}_{i}"
                    features.append(hand.get(key, 0.0))
        
        # 포즈 각도 추출
        pose = angles.get("pose", {})
        for key in ["elbow_left", "elbow_right"]:
            features.append(pose.get(key, 0.0))
        
        # 얼굴 각도 추출
        face = angles.get("face", {})
        for key in ["left_eye", "right_eye", "mouth"]:
            features.append(face.get(key, 0.0))
        
        frame_features.append(features)
    
    return frame_features

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

def predict_from_video(video_path, model_path, label_encoder_path, scaler_path, sequence_length=100):
    """
    비디오에서 제스처를 예측하는 함수
    
    Args:
        video_path: 입력 비디오 파일 경로
        model_path: 저장된 모델 경로
        label_encoder_path: 레이블 인코더 경로
        scaler_path: 스케일러 경로
        sequence_length: 표준화할 시퀀스 길이
    
    Returns:
        predicted_class: 예측된 클래스
        confidence: 예측 확률
    """
    print(f"비디오 처리 시작: {video_path}")
    
    # 1. 비디오에서 관절 각도 추출
    frames_data = extract_angles_from_video(video_path)
    if frames_data is None:
        print("비디오 처리 실패")
        return None, 0.0
    
    # 2. 특성 추출
    features = extract_features_from_frames(frames_data)
    
    # 3. 시퀀스 길이 정규화
    normalized_features = normalize_sequence_length(features, target_length=sequence_length)
    
    # 4. 모델 및 관련 파일 로드
    try:
        model = tf.keras.models.load_model(model_path)
        
        with open(label_encoder_path, 'r') as f:
            label_data = json.load(f)
            classes = np.array(label_data['classes'])
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        print("모델과 관련 파일 로드 완료")
    except Exception as e:
        print(f"모델 또는 관련 파일 로드 중 오류 발생: {e}")
        return None, 0.0
    
    # 5. 특성 스케일링
    X = np.array([normalized_features])
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(original_shape)
    
    # 6. 예측
    print("예측 수행 중...")
    if len(classes) > 2:
        predictions = model.predict(X_scaled)
        predicted_class_index = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)[0]
    else:
        predictions = model.predict(X_scaled)
        predicted_class_index = (predictions > 0.5).astype(int).flatten()
        confidence = predictions.flatten()[0]
    
    predicted_class = classes[predicted_class_index[0]]
    
    print(f"예측 결과: {predicted_class} (확률: {confidence:.4f})")
    
    return predicted_class, confidence

def process_video_with_ui(video_path, model_path, label_encoder_path, scaler_path):
    """
    비디오를 처리하고 예측 결과를 시각적으로 표시하는 함수
    
    Args:
        video_path: 입력 비디오 파일 경로
        model_path: 저장된 모델 경로
        label_encoder_path: 레이블 인코더 경로
        scaler_path: 스케일러 경로
    """
    # 1. 첫 번째 단계: 각도 추출 및 예측
    predicted_class, confidence = predict_from_video(
        video_path, model_path, label_encoder_path, scaler_path
    )
    
    if predicted_class is None:
        print("예측 실패")
        return
    
    # 2. 두 번째 단계: 예측 결과로 비디오 재생하며 표시
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다. 경로: {video_path}")
        return
    
    # MediaPipe 솔루션 초기화 (시각화용)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    pose = mp_pose.Pose(static_image_mode=False)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    
    # 창 이름 설정
    window_name = "Gesture Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe 처리
        hands_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)
        
        # 원본 프레임 복사 (그리기용)
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
        
        # 예측 결과 표시
        result_text = f"예측: {predicted_class} ({confidence:.2f})"
        cv2.putText(
            annotated_frame, 
            result_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0) if confidence > 0.7 else (0, 165, 255), 
            2
        )
        
        # 프레임 표시
        cv2.imshow(window_name, annotated_frame)
        
        # q 키를 누르면 종료
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"최종 예측 결과: {predicted_class} (확률: {confidence:.4f})")

def batch_process_videos(videos_dir, model_path, label_encoder_path, scaler_path):
    """
    디렉토리 내의 모든 비디오 파일 일괄 처리
    
    Args:
        videos_dir: 비디오 파일들이 있는 디렉토리 경로
        model_path: 저장된 모델 경로
        label_encoder_path: 레이블 인코더 경로
        scaler_path: 스케일러 경로
    """
    # 비디오 파일 리스트 가져오기 (mp4, avi, mov 파일)
    video_files = []
    for ext in ["mp4", "avi", "mov"]:
        video_files.extend(glob.glob(os.path.join(videos_dir, f"*.{ext}")))
    
    if not video_files:
        print(f"오류: {videos_dir} 디렉토리에 비디오 파일이 없습니다.")
        return
    
    print(f"총 {len(video_files)}개의 비디오 파일을 찾았습니다.")
    
    # 결과 저장할 CSV 파일
    results_file = os.path.join(videos_dir, "prediction_results.csv")
    with open(results_file, 'w') as f:
        f.write("파일명,예측결과,신뢰도\n")
    
    # 각 비디오 파일 처리
    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        print(f"\n처리 중: {video_filename}")
        
        # 예측 수행
        predicted_class, confidence = predict_from_video(
            video_path, model_path, label_encoder_path, scaler_path
        )
        
        # 결과 기록
        if predicted_class is not None:
            with open(results_file, 'a') as f:
                f.write(f"{video_filename},{predicted_class},{confidence:.4f}\n")
            print(f"예측 결과: {predicted_class} (확률: {confidence:.4f})")
        else:
            with open(results_file, 'a') as f:
                f.write(f"{video_filename},실패,0\n")
            print("예측 실패")
    
    print(f"\n모든 비디오 처리 완료! 결과는 {results_file}에 저장되었습니다.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='비디오에서 제스처 인식')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch', 'ui'],
                        help='실행 모드 (single: 단일 비디오, batch: 폴더 내 모든 비디오, ui: 시각화)')
    parser.add_argument('--video', type=str, help='입력 비디오 파일 경로 (single, ui 모드)')
    parser.add_argument('--dir', type=str, help='비디오 디렉토리 경로 (batch 모드)')
    parser.add_argument('--model', type=str, default='lstm_model.h5', help='모델 파일 경로')
    parser.add_argument('--labels', type=str, default='label_encoder.json', help='레이블 인코더 파일 경로')
    parser.add_argument('--scaler', type=str, default='scaler.pkl', help='스케일러 파일 경로')
    
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
    if args.mode == 'single':
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

if __name__ == "__main__":
    main()
