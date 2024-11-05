import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

#Adding dropout layer; only added in training and not in inference
input = Input(shape=(20,))

hidden_layer1 = Dense(64, activation='relu',)(input)
dropout_layer = Dropout(rate=0.5)(hidden_layer1)
hidden_layer2 = Dense(64, activation='relu')(dropout_layer)
output = Dense(1,activation='sigmoid')(hidden_layer2)

model=Model(inputs=input, outputs=output)
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
x_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000,1))
model.fit(x_train,y_train,epochs=10,batch_size=32)

#Evaluating the model
x_test = np.random.rand(200,20)
y_test = np.random.randint(2,size=(200,1))
loss, accuracy = model.evaluate(x_test,y_test)


#Adding batch normalization; improves training and speed of neural networks

input1 = Input(shape=(20,))

hidden_layer11 = Dense(64, activation='relu',)(input1)
bn_layer = BatchNormalization()(hidden_layer11)
hidden_layer21 = Dense(64, activation='relu')(bn_layer)
output1 = Dense(1,activation='sigmoid')(hidden_layer21)

model1=Model(inputs=input1, outputs=output1)
model1.summary()

model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
x_train1 = np.random.rand(1000, 20)
y_train1 = np.random.randint(2, size=(1000,1))
model1.fit(x_train1,y_train1,epochs=10,batch_size=32)

#Evaluating the model
x_test1 = np.random.rand(200,20)
y_test1 = np.random.randint(2,size=(200,1))
loss1, accuracy1 = model1.evaluate(x_test1,y_test1)

print("DO_Test loss: ", loss)
print("DO_Test accuracy: ", accuracy)
print("BN_Test loss: ", loss1)
print("BN_Test accuracy: ", accuracy1)

