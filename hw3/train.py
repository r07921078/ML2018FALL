import csv
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, LeakyReLU, PReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import History ,ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image  import ImageDataGenerator
from keras import optimizers


infile = sys.argv[1]

def normalization(x):
    x = (x - x.mean()) / x.std()
    return x



x = []
y = []

with open(infile, 'r', encoding='big5') as f_train_in:
    rows = csv.reader(f_train_in)

    is_first = True

    for row in rows:
        if is_first:
            is_first = False

        else:
            y.append(float(row[0]))
            #x.append(row[1].strip().split(' '))
            x += row[1].strip().split(' ')
            
    


#x = np.array(x, dtype=float)
x = np.array(x, dtype=float).reshape(-1, 48, 48, 1)
y = np.array(y, dtype=float)
y = np_utils.to_categorical(y, num_classes=7)

x = normalization(x)


print(x.shape)
print(y.shape)


datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)



model = Sequential()


# Layer 1
model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(48, 48, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Layer 3
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Layer 4
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Layer 5
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))




model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

model.fit_generator(datagen.flow(x, y, batch_size=128), steps_per_epoch=(len(x)*10 / 128),epochs=70)
model.save("model.h5")




















