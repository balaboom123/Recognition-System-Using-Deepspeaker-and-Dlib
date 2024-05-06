import cv2


def video_to_images(video_path, output_path, time):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 确认视频文件打开成功
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * time)  # 间隔为每秒一张

    # 计数器，用于保存图片
    count = 0

    # 读取视频帧并保存为图片
    while cap.isOpened():
        ret, frame = cap.read()

        # 确认成功读取到帧
        if not ret:
            break

        # 每秒一张图片
        if count % interval == 0:
            # 构造保存路径
            img_path = output_path + "/frame_" + str(count) + ".jpg"

            # 保存图片
            cv2.imwrite(img_path, frame)

            count += 1
        count += 1

    # 释放视频对象
    cap.release()