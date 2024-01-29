def normalized_xywh(xywh: tuple, img_height: int, img_width: int):
    """
    rect to xywh
    p.s. cv2_img.shape[1] is width, cv2_img.shape[0] is height

    :param xywh:
    :param img_width:
    :param img_height:
    :return:
    """
    x_center, y_center, width, height = xywh

    # normalized
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height

    return x_center, y_center, width, height


def img_shape(img):
    import cv2
    # get img shape
    image = cv2.imread(img)
    img_height, img_width, _ = image.shape

    return img_height, img_width


def rect_to_xywh(rect):
    """
    rect to xywh

    :param rect: dlib.rectangle
    :return: x_center, y_center, width, height
    """
    # rect to xywh
    x_center = rect.left() + rect.width() // 2
    y_center = rect.top() + rect.height() // 2
    width = rect.width()
    height = rect.height()

    return x_center, y_center, width, height