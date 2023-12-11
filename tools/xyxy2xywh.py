def xyxy2xywh(box, img_size):
    """
    將邊界框座標轉換為 YOLO 格式。
    :param box: 邊界框座標 (x_min, y_min, x_max, y_max)
    :param img_size: 圖片大小 (height, width)
    :return: 歸一化的 YOLO 格式座標 (x_center, y_center, width, height)
    """
    dw = 1. / img_size[1]
    dh = 1. / img_size[0]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)