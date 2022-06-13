import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense
from tensorflow.python.keras import Model, utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import UpSampling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)
print(x_train.shape)
print(x_test.shape)

inputs = Input((28, 28, 1))
encoded_1 = Conv2D(28, (3, 3), activation='relu')(inputs)
encoded_2 = MaxPooling2D((2, 2))(encoded_1)
encoded_3 = Conv2D(56, (3, 3), activation='relu')(encoded_2)
encoded_4 = Flatten()(encoded_3)
encoded = Dense(10, activation='softmax')(encoded_4)
decoded_1 = Dense(10, activation='relu')(encoded)
decoded_2 = Dense(196, activation='sigmoid')(decoded_1)
decoded_3 = Reshape((14, 14, 1))(decoded_2)
decoded_4 = Conv2D(14, (1, 1), activation='relu')(decoded_3)
decoded_5 = UpSampling2D((2, 2))(decoded_4)
decoded = Conv2D(1, (1, 1), activation='sigmoid')(decoded_5)

auto_encoder = Model(inputs, decoded)
auto_encoder.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')
auto_encoder.summary()
utils.plot_model(auto_encoder, show_shapes=True, dpi=100)

print(x_train.shape)
auto_encoder.fit(x_train, x_train, epochs=20)

encoder = Model(inputs, encoded)
encoder.summary()

decoder = Model(decoded_1, decoded)
decoder.summary()

for i in range(10):
    # print(x_test[i])
    print(y_test[i])
    o = encoder.predict(x_test)
    print(sum(o[i]))
    print(max(o[i]))
    print(o[i])
    plt.imshow(decoder.predict(o)[i].reshape(28, 28))
    plt.show()
    print('----------------')