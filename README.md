# Machine-Learning-Exercise
請設計多層類神經網路模型以輸出接近0~9手寫數字影像輸入到模型時所對應的數字

因為是要比出這個數字有幾分像，所以本模型的輸出是一個連續實數變化的設計，不是整數變化。


require model.config 

import numpy as np  

from keras.models import Sequential

from keras.datasets import mnist

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.utils import np_utils  

from matplotlib import pyplot as plt
