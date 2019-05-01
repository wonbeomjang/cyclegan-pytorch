from config.config import get_config
import torch
import os
from model import build_model
import cv2
import numpy as np


def transform(image, net, size):
    image_h, image_w = image.shape[0], image.shape[1]

    down_size_image = cv2.resize(image, (size, size))
    b, g, r = cv2.split(down_size_image)
    down_size_image = cv2.merge([r,g,b])
    down_size_image = torch.from_numpy(down_size_image).float().div(255.0).unsqueeze(0)
    down_size_image = np.transpose(down_size_image, (0, 3, 1, 2)).to(device)
    transform_image = net(down_size_image)

    transform_image = transform_image.data.cpu().numpy().astype(np.uint8) * 255
    transform_image = cv2.resize(transform_image, (image_w, image_h))

    return transform_image


if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator_ab = build_model(config, config.from_style, config.to_style)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("webcam is not detected")

    while (True):
        # ret : frame capture결과(boolean)
        # frame : Capture한 frame
        ret, img = cam.read()

        if (ret):
            cv2.imshow('frame', img)
            img = transform(image=img, net=generator_ab, size=config.image_size)
            if cv2.waitKey(1) & 0xFF == ord(chr(27)):
                break

    cam.release()
    cv2.destroyAllWindows()

