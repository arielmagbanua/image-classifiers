import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

project_dir = os.path.dirname(__file__)
# mask_on_path = os.path.join(project_dir, '../dataset/mask-on-mask-off/mask_on')
# train_mask_on = os.path.normpath(mask_on_path)
# mask_off_path = os.path.join(project_dir, '../dataset/mask-on-mask-off/mask_off')
# train_mask_off = os.path.normpath(mask_off_path)
dataset_path = os.path.join(project_dir, '../dataset/mask-on-mask-off')
dataset_path = os.path.normpath(dataset_path)

DESIRED_ACCURACY = 0.999


class Callbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > DESIRED_ACCURACY:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = Callbacks()

# build the model
model = tf.keras.models.Sequential([
    # input layer
    # first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(360, 360, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),

    # Only 1 output neuron.
    # It will contain a value from 0-1 where 0 for 1 class ('mask on') and 1 for the other ('mask off')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,  # This is the source directory for training images
    target_size=(360, 360),  # All images will be resized to 360x360
    batch_size=128,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(360, 360),  # All images will be resized to 360x360
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=8,
    epochs=100,
    verbose=1,
    callbacks=[callbacks],
)

print(history)
