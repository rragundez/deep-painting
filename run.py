import cv2

from utils import bgr_to_rgb
import numpy as np

from utils import get_model, rgb_to_bgr
from utils import get_style_gram_matrices
from utils import ShowCombination
from keras.preprocessing import image
from keras.applications import vgg19
from keras.optimizers import Adam

IMAGE_SIZE_REDUCE_FACTOR = .5
CAMERA_NUMBER = 0

CAMERA = cv2.VideoCapture(0)
frame = bgr_to_rgb(CAMERA.read()[1])


IMG_HEIGHT, IMG_WIDTH, N_CHANNELS = frame.shape
IMG_HEIGHT = int(IMG_HEIGHT * IMAGE_SIZE_REDUCE_FACTOR)
IMG_WIDTH = int(IMG_WIDTH * IMAGE_SIZE_REDUCE_FACTOR)
print((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))


content_img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
style_img = rgb_to_bgr(np.array(image.load_img('the_scream.jpg', target_size=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))))
style_gram_matrices = get_style_gram_matrices(style_img)


LEARNING_RATE = 30
DECAY_RATE = 0.01
model = get_model(content_img,
                  learning_rate=LEARNING_RATE,
                  decay_rate=DECAY_RATE,
                  clip_weights=True)


N = 100

X = np.repeat(np.expand_dims(content_img, axis=0), N, axis=0)
y = [np.repeat(m, N, axis=0) for m in style_gram_matrices]

EPOCHS = 20
BATCH_SIZE = 1
model.fit(X, y,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          verbose=0,
          callbacks=[ShowCombination(CAMERA, style_img)]
)
CAMERA.release()
cv2.destroyAllWindows()
