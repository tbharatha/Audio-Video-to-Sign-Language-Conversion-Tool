import os
from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from data_utils import labels_to_number, videos_to_dict
from frame_generator import VideoFrameGenerator
from modelss import model_asl
import numpy as np

# model settings
height = 224
width = 224
dim = (height, width)
batch_size = 8
frames = 20
channels = 3
output = 9

TRAIN_PATH = '/Users/tejaswinibharatha/Downloads/Human Activity Recognition using TensorFlow (CNN + LSTM) Code/data/train'
import os

if os.path.exists(TRAIN_PATH):
    print("TRAIN_PATH exists.")
else:
    print("NO")

VAL_PATH = '/Users/tejaswinibharatha/Downloads/Human Activity Recognition using TensorFlow (CNN + LSTM) Code/data/val'
TEST_PATH = '/Users/tejaswinibharatha/Downloads/Human Activity Recognition using TensorFlow (CNN + LSTM) Code/data/test'

# transform labels from string to number
labels = labels_to_number(TRAIN_PATH)
print(f'Labels: {labels}')


# load dataset as dict
y_train_dict = videos_to_dict(TRAIN_PATH, labels)
y_val_dict = videos_to_dict(VAL_PATH, labels)
y_test_dict = videos_to_dict(TEST_PATH, labels)

print(f'\nTrain set: {len(y_train_dict)} videos - with labels')
print(f'Val   set: {len(y_val_dict)} videos - with labels')
print(f'Test  set: {len(y_test_dict)} videos - with labels')
print(f'Train set samples: {list(islice(y_train_dict.items(), 3))}')
print(f'Val   set samples: {list(islice(y_val_dict.items(), 3))}')
print(f'Test  set samples: {list(islice(y_test_dict.items(), 3))}')

# get video paths (without labels)
X_train = list(y_train_dict.keys())
X_val = list(y_val_dict.keys())
X_test = list(y_test_dict.keys())

print(f'\nTrain set: {len(X_train)} videos')
print(f'Val   set: {len(X_val)} videos')
print(f'Test  set: {len(X_test)} videos')
print(f'Train set samples: {X_train[:4]}')
print(f'Val   set samples: {X_val[:4]}')
print(f'Test  set samples: {X_test[:4]}')

print(f'Number of training videos: {len(X_train)}')
print(f'Number of validation videos: {len(X_val)}')
if len(X_train) == 0 or len(X_val) == 0:
    raise ValueError("No videos to train on. Check data directories and labels.")

# instantiation of generators for train and val sets
print('\nTrain generator')
train_generator = VideoFrameGenerator(
    list_IDs=X_train,
    labels=y_train_dict,
    batch_size=batch_size,
    dim=dim,
    n_channels=3,
    n_sequence=frames,
    shuffle=True,
    type_gen='train'
)
x, y = next(iter(train_generator))
print('Sample labels:', y)

# Quick check on training labels
for X_batch, y_batch in train_generator:
    print("Batch labels:", y_batch)
    if np.any(y_batch < 0) or np.any(y_batch >= 9):
        print("Invalid labels found:", y_batch)
    break


print(train_generator.n_sequence)

import numpy as np
all_labels = np.concatenate([y for _, y in train_generator])
print("Unique labels in training set:", np.unique(all_labels))

print('\nVal generator')
val_generator = VideoFrameGenerator(
    list_IDs=X_val,
    labels=y_val_dict,
    batch_size=batch_size,
    dim=dim,
    n_channels=3,
    n_sequence=frames,
    shuffle=True,
    type_gen='val'
)

# model building
print('\nModel building and compiling . . .')

model = model_asl(frames, width, height, channels, output)
model.summary()
# model compiling
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,amsgrad=False, name="Adam")
model.compile(loss='sparse_categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

# callbacks creation
if not os.path.isdir('./saved_models/'):
    os.mkdir('./saved_models/')

# save the best model each time
path = './saved_models/'
checkpoint_cb = ModelCheckpoint(path + 'best_model_attention.keras', save_best_only=True)



# start training
print('\nStart training . . .')
learn_epochs = 500

history = model.fit(train_generator, validation_data=val_generator, epochs=learn_epochs, callbacks=checkpoint_cb)


# save learning curves
if not os.path.isdir('./plots/'):
    os.mkdir('./plots/')

print('\nSaving learning curves graph . . .')
pd.DataFrame(history.history).plot(figsize=(9, 6))
plt.grid(True)
plt.gca().set_ylim(0, 4)
plt.savefig('./plots/learning_curves.png')
