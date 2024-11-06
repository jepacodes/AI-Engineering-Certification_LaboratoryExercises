import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax, Dropout
from tensorflow.keras.models import Sequential
#from tensorflow.keras.utils import plot_model
import numpy as np

#Define the custom dense layer with 32 units and ReLU activation function
class CustomDenseLayer(Layer): #inherits the tf's Layer class
    def __init__(self, units=128): #units=neurons
        super(CustomDenseLayer, self).__init__() #initializes the parent Layer class ensuring that this custom layer behaves like tf's layer
        self.units = units #stores units as an instance so it can be accessed in other methods
        
    def build(self, input_shape): #where the trainable parameters are defined
        self.w = self.add_weight(shape=(input_shape[-1],self.units), #input_shape[-1]-> number of faetures(input size) for each input
                                 initializer='random_normal',
                                 trainable=True) #means that the weights are updated during training
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        
    def call(self,inputs): #defines the forward pass
        return tf.nn.relu(tf.matmul(inputs,self.w) + self.b)
        
#Integrating the custom layer into the model
model = Sequential([
    CustomDenseLayer(128),
    Dropout(0.5),
    CustomDenseLayer(10),
    Softmax() #used for multiclass classification
])

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model summary before building: ")
model.summary()

model.build((1000,20))
print("Model summary after building: ")
model.summary()

#Training the model
X_train = np.random.random((1000,20))
y_train = np.random.randint(10,size=(1000,1))

#Converting the layers to categorical one hot encoding
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
model.fit(X_train, y_train, epochs=10, batch_size=32)

#Evaluating the model
X_test = np.random.random((200,20))
y_test = np.random.randint(10,size=(200,1))

y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)

loss = model.evaluate(X_test,y_test)
print("Test loss: ", loss)
#plot_model(model,to_file='model_architecture.png', show_shapes=True,show_layer_names=True)
