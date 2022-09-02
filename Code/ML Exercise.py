import numpy as np  
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  
from matplotlib import pyplot as plt


(X_train, y_train), (X_test, y_test) = mnist.load_data()



model = Sequential()

model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) 

model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


y_TrainOneHot = np_utils.to_categorical(y_train) 
y_TestOneHot = np_utils.to_categorical(y_test) 


X_train_2D = X_train.reshape(60000, 28*28).astype('float32')  
X_test_2D = X_test.reshape(10000, 28*28).astype('float32')  

x_Train_norm = X_train_2D/255
x_Test_norm = X_test_2D/255


train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)  


scores = model.evaluate(x_Test_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  


X = x_Test_norm[0:10,:]
predictions = model.predict_classes(X)

print(predictions)


from keras.models import model_from_json
json_string = model.to_json()
with open("model.config", "w") as text_file:
    text_file.write(json_string)

    

model.save_weights("model.weight")
plt.imshow(X_test[1])
plt.show() 

with open("model.config", "r") as text_file:
    json_string = text_file.read()

    
model = Sequential()
model = model_from_json(json_string)
model.load_weights("model.weight", by_name=False)


for i in range(0, 10):
    X2 = np.genfromtxt(str(i)+'.csv', delimiter=',').astype('float32')  
    X1 = X2.reshape(1,28*28) / 255
    predictions = model.predict_classes(X1)

    from matplotlib import pyplot as plt
    plt.imshow(X2.reshape(28,28)*255)
    plt.show() 
    print(predictions)