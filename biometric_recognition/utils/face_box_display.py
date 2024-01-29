import cv2
import os
from pathlib import Path
from root_path import root
import dlib
import cv2
import os


def face_box_display(pos_, dir_, dim):
	# 圖片的座標訊息
	x, y, width, height = pos_

	# 讀取圖片
	image = cv2.imread(dir_)

	if dim != None:
		dim = (width, height)
		image = cv2.resize(image, dim)

	# 畫出框
	cv2.rectangle(image, (int(x - width / 2), int(y - height / 2)), (int(x + width / 2), int(y + height / 2)),
				  (0, 255, 0), 2)  # (0, 255, 0) 是綠色框的顏色
	print((x, y), (x + width, y + height))
	# 切割圖片
	# cropped_image = image[y:y + height, x:x + width]

	# 指定目錄保存
	output_directory = 'output_directory'
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)

	output_path = os.path.join(output_directory, 'dlib_image.jpg')

	# 保存切割後的圖片
	cv2.imwrite(output_path, image)


def rect_normalized_xywh(x_center, y_center, width, height):
	cv2_img = cv2.imread(img)
	x_center = x_center // cv2_img.shape[1]
	y_center = y_center // cv2_img.shape[0]
	width = width // cv2_img.shape[1]
	height = height // cv2_img.shape[0]

	return x_center, y_center, width, height


def face_land_mark(input_path):
	# 加载dlib的人脸检测器
	detector = dlib.get_frontal_face_detector()
	# 加载dlib的人脸关键点检测器
	predictor = dlib.shape_predictor(f"{root}/model/shape_predictor_68_face_landmarks.dat")  # 需要下载预训练模型

	# 读取输入图像
	image = cv2.imread(input_path)

	# 将图像从BGR格式转换为灰度格式（dlib人脸检测需要灰度图像）
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# 使用人脸检测器检测人脸
	faces = detector(gray)

	# 遍历每个检测到的人脸
	for face in faces:
		# 使用关键点检测器检测人脸关键点
		landmarks = predictor(image, face)

		# 遍历每个关键点并在图像上绘制它们
		for n in range(0, 68):
			x = landmarks.part(n).x
			y = landmarks.part(n).y
			cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # 在关键点处绘制红色圆圈

	# 保存带有标记特征点的图像
	cv2.imwrite("output.jpg", image)

	# 显示输出图像
	cv2.imshow("Output", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


face_land_mark(r"C:\Users\user\Github\biometric_recognition\biometric_recognition\data\yolo_data\train\images\S1052020_5.png")
# script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir = Path(script_dir).parent
# dir_img = f"{str(script_dir)}/data/train_data/image/S1252001/image_S1252001_00000.png"
# face_box_display((954, 495, 284, 284), dir_img, dim=None)
