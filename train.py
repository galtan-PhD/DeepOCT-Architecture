import numpy as np
from data_prep import prep_and_load_data
from data_prep import pre_process
from model import DeepOCT
import constants as CONST
from keras.callbacks import TensorBoard

data = np.array(prep_and_load_data())
train_size = int(CONST.DATA_SIZE * CONST.SPLIT_RATIO)
print('dataset:', len(data), train_size)

train_data = data[:train_size]
train_images = np.array([i[0] for i in train_data]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
train_labels = np.array([i[1] for i in train_data])
print('Train OCT data fetched..')

test_data = data[train_size:]
test_images = np.array([i[0] for i in test_data]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
test_labels = np.array([i[1] for i in test_data])
print('Test OCT data fetched..')

pre_process(train_images)
pre_process(test_images)

model = DeepOCT()
print('DeepOCT model: Training started...')
history = model.fit(train_images, train_labels, batch_size = 50, epochs = 50, verbose = 1, validation_split	=0.1, callbacks=[tensorboard])
print('DeepOCT model: Training finished...')

model.save('deepOCTmodel.h5')
