'''takes an image and returns a resized version of it
until the minimum constraints of width and height are reached'''
import cv2


def resize(img, scaleFactor):
  return cv2.resize(img, (int(img.shape[1] * (1 / scaleFactor)), \
                    int(img.shape[0] * (1 / scaleFactor))), \
                    interpolation=cv2.INTER_AREA)


# Generator
def pyramid(image, scale=1.5, minSize=(200, 80)):
  yield image

  while True:
    image = resize(image, scale)
    if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
      break

    yield image
