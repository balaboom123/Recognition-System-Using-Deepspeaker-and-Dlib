import cv2
import os
from pathlib import Path

def face_box_display(pos_, dir_, dim):
	# 圖片的座標訊息
	x, y, width, height = pos_

	# 讀取圖片
	image = cv2.imread(dir_)

	if dim != None:
		dim = (width, height)
		image = cv2.resize(image, dim)

	# 畫出框
	cv2.rectangle(image, (int(x - width/2), int(y - height/2)), (int(x + width/2), int(y + height/2)), (0, 255, 0), 2)  # (0, 255, 0) 是綠色框的顏色
	print((x, y), (x + width, y + height))
	# 切割圖片
	# cropped_image = image[y:y + height, x:x + width]

	# 指定目錄保存
	output_directory = 'output_directory'
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)

	output_path = os.path.join(output_directory, 'cropped_image.jpg')

	# 保存切割後的圖片
	cv2.imwrite(output_path, image)


script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = Path(script_dir).parent
dir_img = f"{str(script_dir)}/data/train_data/image/S1252001/image_S1252001_00000.png"
face_box_display((921, 645, 898, 848), dir_img, dim=None)
