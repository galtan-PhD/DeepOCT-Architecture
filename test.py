from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import cv2
import constants as CONST
from data_prep import pre_process


model = load_model('deepOCTmodel.h5')

def predict(filename):
    test_image = image.load_img(filename, target_size=(CONST.IMG_SIZE, CONST.IMG_SIZE))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == CONST.DME_LBL:
        print(CONST.DME)
    elif result[0][0] == CONST.NONDME_LBL:
        print(CONST.NONDME)
    else:
        print("None")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    name = 'OCT_withGradCAM.jpg'
    cv2.imwrite(name, frame)

    predict(name)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

model.summary()
cap.release()
cv2.destroyAllWindows()
