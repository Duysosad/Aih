# USAGE
# python classify.py --image examples/bulbasaur_plush.png

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# xây dựng trình phân tích tham số và phân tích các tham số
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# Tải ảnh lên
image = cv2.imread(args["image"])
output = image.copy()
 
# tiền xử lý hình ảnh để phân loại
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# tải mạng nơ-ron tích chập đã được huấn luyện và bộ nhị phân hóa nhãn
print("[INFO] loading network...")
model = load_model('pokedex.h5')
lb = pickle.loads(open('lb.pickle', "rb").read())

# phân loại hình ảnh đầu vào
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# kiểm tra xem dự đoán có chính xác không dựa trên tên file
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# tạo nhãn và vẽ nhãn lên ảnh
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# hiển thị ảnh kết quả
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)