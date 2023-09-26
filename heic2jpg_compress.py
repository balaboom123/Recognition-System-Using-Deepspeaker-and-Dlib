import cv2
from PIL import Image
import os


def heic2jpg():
	input_folder = "path/to/your/input/folder"
	output_folder = "path/to/your/output/folder"

	# 創建輸出資料夾（如果不存在）
	os.makedirs(output_folder, exist_ok=True)

	# 列出資料夾內所有檔案
	file_list = os.listdir(input_folder)

	# 對每個檔案進行處理
	for file_name in file_list:
		if file_name.lower().endswith(".heic") or file_name.lower().endswith(".jpg"):
			# 讀取圖片
			img_path = os.path.join(input_folder, file_name)
			img = Image.open(img_path)

			# 將HEIC轉換成JPG
			if file_name.lower().endswith(".heic"):
				new_file_name = os.path.splitext(file_name)[0] + ".jpg"
				new_img_path = os.path.join(output_folder, new_file_name)
				img.save(new_img_path, "JPEG")
			else:
				new_img_path = os.path.join(output_folder, file_name)

			# 壓縮圖片
			compressed_img = img.copy()
			compressed_img.save(new_img_path, "JPEG", quality=20)

			print(f"{file_name} 轉換並壓縮完成")


script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "DJI_0798.JPG")
new_img_path = os.path.join(script_dir, "compression20.jpg")

img = Image.open(img_path)
compressed_img = img.copy()
compressed_img.save(new_img_path, "JPEG", quality=20)

# 縮放圖像
image = cv2.imread(img_path)
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim)
cv2.imwrite('downscaling.jpg', resized_image)