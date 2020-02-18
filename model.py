import os
import csv
import cv2
import datetime
import transform
import argparse
import numpy as np
from network import model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

class Pipeline:
    def __init__(self, model=None, base_path='', epochs=2):
        self.data = []
        self.model = model
        self.epochs = epochs
        self.training_samples = []
        self.validation_samples = []
        self.correction_factor = 0.2
        self.base_path = base_path
        self.image_path = self.base_path + '/IMG/'
        self.driving_log_path = self.base_path + '/driving_log.csv'

    def import_data(self):
        with open(self.driving_log_path) as csvfile:
            reader = csv.reader(csvfile)
            # Skip the column names row
            next(reader)

            for line in reader:
                self.data.append(line)

        return None

    def process_batch(self, batch_sample):
        steering_angle = np.float32(batch_sample[3])
        images, steering_angles = [], []

        for image_path_index in range(3):
            image_name = batch_sample[image_path_index].split('/')[-1]

            image = cv2.imread(self.image_path + image_name)
            rgb_image = transform.bgr2rgb(image)
            resized = transform.crop_and_resize(rgb_image)

            images.append(resized)

            if image_path_index == 1:
                steering_angles.append(steering_angle + self.correction_factor)
            elif image_path_index == 2:
                steering_angles.append(steering_angle - self.correction_factor)
            else:
                steering_angles.append(steering_angle)

            if image_path_index == 0:
                flipped_center_image = transform.flipimg(resized)
                images.append(flipped_center_image)
                steering_angles.append(-steering_angle)

        return images, steering_angles

    def data_generator(self, samples, batch_size=128):
        num_samples = len(samples)

        while True:
            shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images, steering_angles = [], []

                for batch_sample in batch_samples:
                    augmented_images, augmented_angles = self.process_batch(batch_sample)
                    images.extend(augmented_images)
                    steering_angles.extend(augmented_angles)

                X_train, y_train = np.array(images), np.array(steering_angles)
                yield X_train, y_train

    def split_data(self):
        train, validation = train_test_split(self.data, test_size=0.2)
        self.training_samples, self.validation_samples = train, validation

        return None

    def train_generator(self, batch_size=128):
        return self.data_generator(samples=self.training_samples, batch_size=batch_size)

    def validation_generator(self, batch_size=128):
        return self.data_generator(samples=self.validation_samples, batch_size=batch_size)
    
    def checkpoint_callback(self):
        model_type = 'nvidia'
        save_dir = "./saved_models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = 'behavior_cloning_%s_model.{epoch:03d}.h5' % model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True)
        return checkpoint_callback
    
    def tensorboard_callback(self):
        log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        file_writer.set_as_default()

        tensorboard_callback = TensorBoard(log_dir=log_dir,
                                           update_freq='batch',
                                           histogram_freq=1)
        return tensorboard_callback
    
    def run(self):
        self.split_data()
        self.model.fit(self.train_generator(),
                       validation_data=self.validation_generator(),
                       epochs=self.epochs,
                       steps_per_epoch=len(self.training_samples) * 2,
                       validation_steps=len(self.validation_samples),
                       callbacks=[self.checkpoint_callback(), self.tensorboard_callback()])
        self.model.save('model.h5')

def main():
    parser = argparse.ArgumentParser(description='Train a car to drive itself')
    parser.add_argument(
        '--data-base-path',
        type=str,
        default='./data',
        help='Path to image directory and driving log'
    )

    args = parser.parse_args()

    # Instantiate the pipeline
    pipeline = Pipeline(model=model(), base_path=args.data_base_path, epochs=5)

    # Feed driving log data into the pipeline
    pipeline.import_data()
    # Start training
    pipeline.run()

if __name__ == '__main__':
    main()