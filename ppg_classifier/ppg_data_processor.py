# %%
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
from datahandler import ppg_preprocessing,ppg_data,visualize_ppg_data, average_sequential



# %%
data_files = glob.glob('data/*.csv')
clinical_data = pd.read_excel('clinical_data.xlsx')

labels=[]
features = []
for i in clinical_data['Labels']:
    labels.append(i)
    
for data_paths in data_files:
    figure = ppg_data(data_paths,1000)
    features.append(figure)

a=0 
averaged_list = []
new_features=[]
for i in range(len(features)):
    if a<3:
        averaged_list.append(features[i])
        a+=1
    if a==3:
        averaged_array = np.mean([averaged_list[0], averaged_list[1], averaged_list[2]], axis=0)
        new_features.append(averaged_array)
        averaged_list=[]
        a=0
    
    

flattened_data = [item.flatten() for item in new_features]

features = np.array(flattened_data)
labels= np.array(labels)
labels = keras.utils.to_categorical(labels, 4)

# %%
x_train, x_test,y_train,y_test=train_test_split(features,labels,random_state=16, test_size=0.10, stratify=labels)
x_train,x_val, y_train,y_val = train_test_split(x_train, y_train,random_state=16, test_size=0.10, stratify=y_train)
# Define the model
model = Sequential()

# Add a Flatten layer if your input is 2D (e.g., a 2D array where one dimension is time and the other is features)
model.add(Input(shape=(2000,1)))  # Flatten a 2D input to 1D (1000 samples per PPG segment)
# Add a 1D convolutional layer for feature extraction
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))  # Downsample the features

# Add another Conv1D layer for deeper feature extraction
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))

# Flatten the output from the convolutional layers
model.add(Flatten())
# Add fully connected (dense) layers
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.4))  # Dropout for regularization
# Output layer for binary classification
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

val=model.fit(x=x_train, y=y_train, batch_size=8, shuffle=True, epochs=10,validation_data=(x_val,y_val))
model.evaluate(x_test,y_test)


# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(val.history['accuracy'], color='g')
ax1.plot(val.history['val_accuracy'], color='b')
ax1.legend(['Training accuracy' ,'Validation accuracy'])
ax1.set_title("Train Vs Validation Accuracy")
ax1.set_xlabel("# of Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.00,1)
ax2.plot(val.history['loss'], color='g')
ax2.plot(val.history['val_loss'], color='b')
ax2.legend(['Training loss' ,'Validation loss'])
ax2.set_title("Train Vs Validation Loss")
ax2.set_xlabel("# of Epoch")
ax2.set_ylabel("Loss")
ax2.set_ylim(0.00,2.00) 



# %% [markdown]
# Unnecessary for now 

# %%
#model.save('Data File/PPG_Prediction_Model1.keras') 


