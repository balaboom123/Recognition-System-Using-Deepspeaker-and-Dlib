"""Face detector based on EfficientDet model.

For more details: https://github.com/yinguobing/face_detector
"""
import cv2
import numpy as np
import tensorflow as tf


class Detector(object):
    """A face detector class implementation.

    Features:
        - box transformations like padding to square, scale, offset, clip, etc.
        - built in image preprocessing.
        - filter the results by detection score.

    The model weights should be provided in TensorFlow SavedModel format.
    """

    def __init__(self, saved_model, input_size=512):
        """Build an EfficientDet model runner.

        Args:
            saved_model: the string path to the SavedModel.
            input_size: the size of the model's input.
        """
        self.scale_width = 0
        self.scale_height = 0
        self.input_size = input_size

        # Load the SavedModel object.
        imported = tf.saved_model.load(saved_model)
        self._predict_fn = imported.signatures["serving_default"]

        # To avoid garbage collected by Python, see TensorFlow issue:37615
        self._predict_fn._backref_to_saved_model = imported

    def preprocess(self, image):
        """Preprocess the input image.

        Args:
            image: the input image.

        Returns:
            image: processed image.
        """

        # Scale the image first.
        height, width, _ = image.shape
        self.ratio = self.input_size / max(height, width)
        image = cv2.resize(
            image, (int(self.ratio * width), int(self.ratio * height)))

        # Convert to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Then pad the image to input size.
        self.padding_h = self.input_size - int(self.ratio * width)
        self.padding_v = self.input_size - int(self.ratio * height)
        image = cv2.copyMakeBorder(
            image, 0, self.padding_v, 0, self.padding_h, cv2.BORDER_CONSTANT, (0, 0, 0))

        return image

    def __filter(self, detections, threshold):
        """Filter the detection results by score threshold.

        Args:
            detections: the results of detection.
            threshold: float value to filter the results.

        Returns:
            boxes: filtered boxes.
            scores: filtered scores.
            classes: filtered classes.
        """
        # Get the detection results.
        boxes = detections['output_0'].numpy()[0]
        scores = detections['output_1'].numpy()[0]
        classes = detections['output_2'].numpy()[0]

        # Filter out the results by score threshold.
        mask = scores > threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        return boxes, scores, classes

    @tf.function
    def _predict(self, images):
        """A helper function to make prediction."""
        return self._predict_fn(images)

    def predict(self, image, threshold):
        """Run inference with image inputs.

        Args:
            image: a numpy array as an input image.

        Returns:
            boxes: a numpy array [[ymin, xmin, ymax, xmax], ...] as face 
                bounding boxes.
            scores: confidence values for face.
            classes: classification result.
        """
        frame_tensor = tf.constant(image, dtype=tf.uint8)
        frame_tensor = tf.expand_dims(frame_tensor, axis=0)
        detections = self._predict(frame_tensor)
        boxes, scores, classes = self.__filter(detections, threshold)

        # Scale the box back to the original image size.
        boxes /= self.ratio

        # Crop out the padding area.
        boxes[:, 2] = np.minimum(
            boxes[:, 2], (self.input_size - self.padding_v)/self.ratio)
        boxes[:, 1] = np.minimum(
            boxes[:, 1], (self.input_size - self.padding_h)/self.ratio)

        return boxes, scores, classes

    def transform_to_square(self, boxes, scale=1.0, offset=(0, 0)):
        """Get the square bounding boxes.

        Args:
            boxes: input boxes [[ymin, xmin, ymax, xmax], ...]
            scale: ratio to scale the boxes
            offset: a tuple of offset ratio to move the boxes (x, y)

        Returns:
            boxes: square boxes.
        """
        ymins, xmins, ymaxs, xmaxs = np.split(boxes, 4, 1)
        width = xmaxs - xmins
        height = ymaxs - ymins

        # How much to move.
        offset_x = offset[0] * width
        offset_y = offset[1] * height

        # Where is the center location.
        center_x = np.floor_divide(xmins + xmaxs, 2) + offset_x
        center_y = np.floor_divide(ymins + ymaxs, 2) + offset_y

        # Make them squares.
        margin = np.floor_divide(np.maximum(height, width) * scale, 2)
        boxes = np.concatenate((center_y-margin, center_x-margin,
                                center_y+margin, center_x+margin), axis=1)

        return boxes

    def clip_boxes(self, boxes, margins):
        """Clip the boxes to the safe margins.

        Args:
            boxes: input boxes [[ymin, xmin, ymax, xmax], ...].
            margins: a tuple of 4 int (top, left, bottom, right) as safe margins.

        Returns:
            boxes: clipped boxes.
            clip_mark: the mark of clipped sides, like [[True, False, False, False], ...]
        """
        top, left, bottom, right = margins

        clip_mark = (boxes[:, 0] < top, boxes[:, 1] < left,
                     boxes[:, 2] > bottom, boxes[:, 3] > right)

        boxes[:, 0] = np.maximum(boxes[:, 0], top)
        boxes[:, 1] = np.maximum(boxes[:, 1], left)
        boxes[:, 2] = np.minimum(boxes[:, 2], bottom)
        boxes[:, 3] = np.minimum(boxes[:, 3], right)

        return boxes, clip_mark

    def draw_boxes(self, image, boxes, scores, color=(0, 255, 0)):
        """Draw the bounding boxes.

        Args:
            image: the image to draw on.
            boxes: the face boxes.
            color: the color of the box.
            scores: detection score.
            color: a tuple of (B, G, R) 8bit int.

        """
        for box, score in zip(boxes, scores):
            y0, x0, y1, x1 = [int(b) for b in box]
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
            cv2.putText(image, "Face:{:.2f}".format(score),
                        (x0, y0-7), cv2.FONT_HERSHEY_DUPLEX, 0.5, color,
                        1, cv2.LINE_AA)
