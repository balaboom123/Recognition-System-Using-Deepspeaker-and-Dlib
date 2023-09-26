from iptcinfo3 import IPTCInfo

# 打开图像文件
image_path = "dog.jpg"
iptc_info = IPTCInfo(image_path)

# 添加标签
new_tags = ["cat"]
iptc_info['keywords'] = new_tags

# 保存更改
iptc_info.save(options=["overwrite"])

print("Tags added successfully.")

