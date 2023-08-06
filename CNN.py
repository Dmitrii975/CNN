import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

img_w, img_h = 300, 300
train_sml = 10
batch_size = 5

path = '/content/drive/MyDrive/Train'

gen = IDG(rescale = (1.0 / 255))
train_arr = gen.flow_from_directory(
    directory = path,
    target_size = (img_w, img_h),
    batch_size = batch_size,
    class_mode = "categorical"
    )

valid_arr = gen.flow_from_directory(
    directory = path,
    target_size = (img_w, img_h),
    batch_size = batch_size,
    class_mode = "categorical"
    )

test_arr = gen.flow_from_directory(
    directory = path,
    target_size = (img_w, img_h),
    batch_size = 10,
    class_mode = "categorical"
    )

epochs = 15
input_shape = (img_w, img_h, 3)

model = Sequential()

model.add(Conv2D(30 , (3, 3), input_shape = input_shape ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(60 , (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(30 , (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Activation("softmax"))

model.summary()

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
    )

model.fit_generator(
    train_arr,
    steps_per_epoch = train_sml // batch_size,
    epochs = epochs,
    validation_data = valid_arr,
    validation_steps = train_sml // batch_size,
    verbose=0 
    )


pred_gen = gen.flow_from_directory(
    "/content/drive/MyDrive/Test_apples",
    target_size = (300, 300),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 20
    )

for i in range(6):
    temp = next(pred_gen)

    img = temp[0][0]
    img_pred = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img_pred)
    print(prediction)

    plt.imshow(img)
    plt.show()