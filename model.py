"""This file is for patches classification (step 1).

Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
"""
import numpy as np   # We recommend to use numpy arrays
from sklearn.base import BaseEstimator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
import tensorflow as tf
class Model(BaseEstimator):
    """Main class for Classification problem."""

    def __init__(self):
        """Init method.

        We define here a simple (shallow) CNN.
        """
        self.num_train_samples = 0
        self.num_feat = 1
        self.num_labels = 1
        self.is_trained = False

        self.model = Sequential()
        self.model.add(Conv2D(8, (3, 3), input_shape=(40, 40, 3)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization(axis=-1, epsilon=2e-5,momentum=0.9))#
        self.model.add(Conv2D(16, (1, 1), strides=(2,2), kernel_regularizer=l2(0.0001)))
        # self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization(axis=-1, epsilon=2e-5,momentum=0.9))
        self.model.add(Conv2D(32, (2,2),strides=(1,1), kernel_regularizer=l2(0.0001)))               
        # self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization(axis=-1, epsilon=2e-5,momentum=0.9))#
        self.model.add(Conv2D(32, (1, 1),strides=(1,1), kernel_regularizer=l2(0.0001))) 
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(GlobalMaxPooling2D())
        # self.model.add(Dense(16, activation='relu'))

        # self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])

    def fit(self, X, y):
        """Fit method.

        This function should train the model parameters.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
               An image has the following shape (40, 40, 3) then 4800 features.
            y: Training label matrix of dim num_train_samples.
        Both inputs are numpy arrays.
        """
        self.num_train_samples = X.shape[0]
        X = X.reshape((self.num_train_samples, 40, 40, 3))
        trainAug = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=20,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            horizontal_flip=False,
            fill_mode="nearest")
        # initialize the training generator
        train_generator = trainAug.flow(
            X,
            y,
            shuffle=False,
            batch_size=64)
        self.model.fit_generator(train_generator,
            steps_per_epoch=X.shape[0] // 64, epochs=15)
        self.is_trained = True

    def predict(self, X):
        """Predict method.

        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
               An image has the following shape (40, 40, 3) then 4800 features.
        This function should provide predictions of labels on (test) data.
        Make sure that the predicted values are in the correct format for the
        scoring metric. For example, binary classification problems often
        expect predictions in the form of a discriminant value (if the area
        under the ROC curve it the metric) rather that predictions of the class
        labels themselves. For multi-class or multi-labels problems, class
        probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        """
        num_test_samples = X.shape[0]
        X = X.reshape((num_test_samples, 40, 40, 3))
        # initialize the validation (and testing) data augmentation object
        testAug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
        # initialize the testing generator
        testGen = testAug.flow(
            X,
            shuffle=False,
            batch_size=32)
        return self.model.predict_generator(testGen)
