import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras import layers

# 이미지 경로와 라벨을 저장할 리스트 생성
data = []
labels = []
# 이미지 데이터가 있는 디렉토리 경로
data_path = "image_240118"

# 이미지 데이터와 라벨링 정보를 읽어옴
for img in os.listdir(data_path):
    image = cv2.imread(os.path.join(data_path, img))

    height, width, _ = image.shape
    # 아래 반절 이미지만 사용
    # half_height = height // 2  # 이미지 아래 반쪽의 높이
    # bottom_half = image[half_height:, :, :]
    # image = cv2.resize(bottom_half, (64, 64))  # 이미지 크기 조정

    # 일부 이미지 비율 조절하여 사용
    split_ratio1 = 0.4  # 위쪽 기준
    split_height = int(height * split_ratio1)  # 원하는 비율에 따라 분할할 높이 계산
    split_ratio2 = 0.9 #아래쪽 기준
    split_height2 = int(height * split_ratio2)
    lower_part = image[split_height:split_height2, :, :]  # 중간 부분 추출
    image = cv2.resize(lower_part, (64, 64))  # 이미지 크기 조정

    # 파일 이름에서 라벨링 정보를 추출하여 처리
    #label = int(img.split('_')[-1].split('.')[0])  # 파일 이름에서 라벨 정보 추출
    label = int(img.split('_')[0])

    if label == 1000:
        data.append(image)
        labels.append(0)  # 라벨 0
    elif label == 1100:
        data.append(image)
        labels.append(1)  # 라벨 1
    elif label == 1010:
        data.append(image)
        labels.append(2)  # 라벨 2
    # 그외 라벨이면 파일 삭제
    else :
        os.remove(os.path.join(data_path, img))
        print(f"File {img} deleted.")

# 데이터와 라벨을 넘파이 배열로 변환
data = (np.array(data, dtype='float32')/127.5) -1
#data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)

#X_train, X_val, Y_train, Y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train, X_rest, Y_train, Y_rest = train_test_split(data, labels, test_size=0.2, random_state=42 )

# 나머지 데이터를 valid와 test로 나눕니다. 여기서는 전체 데이터 기준으로 20%를 valid로, 10%를 test로 사용합니다.
# valid와 test를 각각 전체 데이터 기준으로 2:1 비율로 나눕니다.
X_valid, X_test, Y_valid, Y_test = train_test_split(X_rest, Y_rest, test_size=1/2, random_state=43)

model = keras.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

# callbacks: early stopping and autosave
callback_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', restore_best_weights = True, patience=10)
callback_save = tf.keras.callbacks.ModelCheckpoint(
    filepath = './/model//model_{epoch:02d}.h5',
    save_freq = 'epoch' # 매 에포크마다 저장
    )

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=50

                    , batch_size = 50, callbacks=[callback_earlystopping, callback_save])
model.save('keras_model.h5')

# x_test, y_test를 사용하여 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, Y_test)

print(f"Test Accuracy: {test_accuracy}")
print(f"Test Loss: {test_loss}")
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title('hist')
plt.show()
