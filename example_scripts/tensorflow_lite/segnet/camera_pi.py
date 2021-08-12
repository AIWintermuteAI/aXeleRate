import io
import time
import picamera
import picamera.array
import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    @staticmethod
    def frames():
        with picamera.PiCamera(resolution = (1280,720)) as camera:
            # let camera warm up
            time.sleep(2)

            with picamera.array.PiRGBArray(camera, size=(1280,720)) as stream:
                while True:
                    camera.capture(stream, format='bgr')
                    # At this point the image is available as stream.array
                    image = stream.array
                    stream.truncate(0)
                    yield image

