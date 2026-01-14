import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

x_train = x_train/255.0
x_test = x_test/255.0

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x= Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs = base_model.input, outputs=outputs)

model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

callback = EarlyStopping(patience=3, restore_best_weights=True)
model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs =10,
    batch_size=64,
    callbacks=[callback]
)


preds = model.predict(x_test)
y_pred = np.argmax(pres, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))
model.save('image_classify.h5')
print("model saved")