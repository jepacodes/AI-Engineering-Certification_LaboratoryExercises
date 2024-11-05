import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense 
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

#Defining the output layer
input = Input(shape=(20,)) #assuming dataset with 20 vector length

#Adding hidden layers 
hidden_layer1 = Dense(64, activation='relu')(input)
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)
#FC layer/dense layer that takes the output of the previous layer as the input

#Defining the output layer
output = Dense(1, activation='sigmoid')(hidden_layer2) #suitable for binary classification

#Create the model by specifying the input and output models
model = Model(inputs = input, outputs = output)
model.summary()

#Compile the model 
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Train the model
x_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000,1))
model.fit(x_train,y_train,epochs=10,batch_size=32)

#Evaluating the model
x_test = np.random.rand(200,20)
y_test = np.random.randint(2,size=(200,1))
loss, accuracy = model.evaluate(x_test,y_test)
print("Test loss: ", loss)
print("Test accuracy: ", accuracy)
