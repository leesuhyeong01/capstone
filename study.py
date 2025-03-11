import numpy as np
import pandas as pd
import json
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(json_paths, label_map=None):
    """
    JSON 파일들을 로드하고 전처리하는 함수
    
    Args:
        json_paths: JSON 파일 경로 리스트
        label_map: 파일 이름에서 레이블을 추출하는 딕셔너리(선택사항)
    
    Returns:
        sequences: 전처리된 시퀀스 데이터
        labels: 각 시퀀스에 대한 레이블
    """
    all_sequences = []
    all_labels = []
    
    for json_path in json_paths:
        # 파일 이름에서 레이블 추출 (예: "간호사_13.json" -> "간호사")
        if label_map is None:
            label = os.path.basename(json_path).split('_')[0]
        else:
            filename = os.path.basename(json_path)
            label = None
            for key, patterns in label_map.items():
                if any(pattern in filename for pattern in patterns):
                    label = key
                    break
            if label is None:
                print(f"경고: {filename}에 대한 레이블을 찾을 수 없습니다. 건너뜁니다.")
                continue
        
        # JSON 파일 로드
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"오류: {json_path} 파일을 로드하는 중 문제가 발생했습니다: {e}")
            continue
        
        # 각 프레임의 특성 추출
        frame_features = []
        for frame in data:
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
        
        # 최소 프레임 수 확인
        if len(frame_features) < 10:
            print(f"경고: {json_path}의 프레임 수가 너무 적습니다. 건너뜁니다.")
            continue
        
        all_sequences.append(frame_features)
        all_labels.append(label)
    
    return all_sequences, all_labels

# 시퀀스 길이 정규화 함수
def normalize_sequence_length(sequences, target_length=100):
    """
    모든 시퀀스의 길이를 동일하게 만드는 함수
    
    Args:
        sequences: 다양한 길이의 시퀀스 리스트
        target_length: 목표 시퀀스 길이
        
    Returns:
        normalized_sequences: 정규화된 시퀀스 리스트
    """
    normalized_sequences = []
    
    for seq in sequences:
        seq_len = len(seq)
        
        # 시퀀스가 너무 짧으면 패딩 추가
        if seq_len < target_length:
            # 마지막 프레임을 복제하여 패딩
            padding = [seq[-1]] * (target_length - seq_len)
            normalized_seq = seq + padding
        # 시퀀스가 너무 길면 균등하게 샘플링
        elif seq_len > target_length:
            indices = np.linspace(0, seq_len-1, target_length, dtype=int)
            normalized_seq = [seq[i] for i in indices]
        else:
            normalized_seq = seq
            
        normalized_sequences.append(normalized_seq)
    
    return normalized_sequences

# LSTM 모델 생성 함수
def create_lstm_model(input_shape, num_classes):
    """
    LSTM 모델 아키텍처 정의
    
    Args:
        input_shape: 입력 데이터 형태 (시퀀스 길이, 특성 수)
        num_classes: 분류할 클래스 수
        
    Returns:
        model: 컴파일된 LSTM 모델
    """
    model = Sequential()
    
    # LSTM 레이어
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    
    # 분류 레이어
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    # 출력 레이어
    if num_classes > 2:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    else:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    
    # 모델 컴파일
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

# 메인 함수
def train_lstm_model(data_dir, model_save_path="gesture_model.h5", sequence_length=100, epochs=50):
    """
    LSTM 모델 학습 메인 함수
    
    Args:
        data_dir: JSON 파일이 저장된 디렉토리 경로
        model_save_path: 학습된 모델을 저장할 경로
        sequence_length: 표준화할 시퀀스 길이
        epochs: 학습 에포크 수
    """
    # JSON 파일 경로 가져오기
    json_paths = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not json_paths:
        print(f"오류: {data_dir}에서 JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(json_paths)}개의 JSON 파일을 찾았습니다.")
    
    # 데이터 로드 및 전처리
    sequences, labels = load_and_preprocess_data(json_paths)
    
    if not sequences:
        print("오류: 처리할 수 있는 시퀀스 데이터가 없습니다.")
        return
    
    print(f"로드된 시퀀스: {len(sequences)}, 레이블: {len(labels)}")
    
    # 레이블 인코딩
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # 클래스 정보 출력
    num_classes = len(label_encoder.classes_)
    print(f"클래스: {label_encoder.classes_}")
    print(f"총 클래스 수: {num_classes}")
    
    # 시퀀스 길이 정규화
    normalized_sequences = normalize_sequence_length(sequences, target_length=sequence_length)
    
    # 데이터를 numpy 배열로 변환
    X = np.array(normalized_sequences)
    y = np.array(encoded_labels)
    
    # 데이터 형태 확인
    print(f"데이터 형태: {X.shape}, 레이블 형태: {y.shape}")
    
    # 특성 스케일링
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(original_shape)
    
    # 학습 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # 모델 생성
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape, num_classes)
    
    # 모델 요약 정보 출력
    model.summary()
    
    # 콜백 정의
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy')
    ]
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 모델 평가
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"테스트 정확도: {test_acc:.4f}")
    
    # 학습 과정 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('모델 정확도')
    plt.ylabel('정확도')
    plt.xlabel('에포크')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('모델 손실')
    plt.ylabel('손실')
    plt.xlabel('에포크')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # 혼동 행렬 계산 및 시각화
    y_pred = np.argmax(model.predict(X_test), axis=1) if num_classes > 2 else (model.predict(X_test) > 0.5).astype(int).flatten()
    
    conf_matrix = tf.math.confusion_matrix(y_test, y_pred).numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.title('혼동 행렬')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 레이블 인코더 저장
    with open('label_encoder.json', 'w') as f:
        json.dump(
            {'classes': label_encoder.classes_.tolist()},
            f
        )
    
    # 스케일러 저장
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"모델이 {model_save_path}에 저장되었습니다.")
    print(f"레이블 인코더가 label_encoder.json에 저장되었습니다.")
    print(f"스케일러가 scaler.pkl에 저장되었습니다.")
    print(f"학습 과정 그래프가 training_history.png에 저장되었습니다.")
    print(f"혼동 행렬이 confusion_matrix.png에 저장되었습니다.")

# 추론 함수
def predict_gesture(model_path, json_path, label_encoder_path, scaler_path, sequence_length=100):
    """
    저장된 모델을 사용하여 새 JSON 데이터에서 제스처 예측
    
    Args:
        model_path: 저장된 모델 경로
        json_path: 예측할 JSON 파일 경로
        label_encoder_path: 레이블 인코더 정보가 저장된 JSON 파일 경로
        scaler_path: 저장된 스케일러 경로
        sequence_length: 시퀀스 길이
    """
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 레이블 인코더 로드
    with open(label_encoder_path, 'r') as f:
        label_data = json.load(f)
        classes = np.array(label_data['classes'])
    
    # 스케일러 로드
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 데이터 로드 및 전처리
    sequences, _ = load_and_preprocess_data([json_path])
    
    if not sequences:
        print(f"오류: {json_path}에서 유효한 데이터를 찾을 수 없습니다.")
        return
    
    # 시퀀스 길이 정규화
    normalized_sequences = normalize_sequence_length(sequences, target_length=sequence_length)
    
    # numpy 배열로 변환
    X = np.array(normalized_sequences)
    
    # 특성 스케일링
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(original_shape)
    
    # 예측
    if len(classes) > 2:
        predictions = model.predict(X_scaled)
        predicted_class_index = np.argmax(predictions, axis=1)
        prediction_probability = np.max(predictions, axis=1)
    else:
        predictions = model.predict(X_scaled)
        predicted_class_index = (predictions > 0.5).astype(int).flatten()
        prediction_probability = predictions.flatten()
    
    predicted_class = classes[predicted_class_index][0]
    
    print(f"예측된 제스처: {predicted_class} (확률: {prediction_probability[0]:.4f})")
    
    return predicted_class, prediction_probability[0]


if __name__ == "__main__":
#     # 학습 함수 호출
 train_lstm_model(
         data_dir="C:/Users/sh/Desktop/capstone/json/",  # JSON 파일이 저장된 디렉토리
         model_save_path = "C:/Users/sh/Desktop/capstone/lstm_model.keras", # 모델을 저장할 경로
         sequence_length=100,  # 시퀀스 길이 표준화
         epochs=50  # 학습 에포크 수
     )
    
#     # 추론 함수 호출 (선택사항)
#     # predict_gesture(
#     #     model_path="gesture_model.h5",
#     #     json_path="./new_data/test_gesture.json",
#     #     label_encoder_path="label_encoder.json",
#     #     scaler_path="scaler.pkl"
#     # )
