#Sieć neuronowa

#Bibioteki do obliczen tensorowych

#import tensorflow as tf
#from tensorflow import keras

import plaidml.keras
plaidml.keras.install_backend()

#Bibioteka do obsługi sieci neuronowych
import keras

#Załadowania bazy uczącej
import imageio
import numpy as np

import os

from keras.models import load_model

# returns a compiled model
# identical to the previous one
genderModel = load_model('siec.h5')
genderModel.summary() # Display summary

ImgWidth = 100
ImgHeight = 100

BazaImg = np.empty((2,ImgHeight,ImgWidth,3))

K = 0
M = 0
L = 0

filepath = ".\\wynik.txt"
file = open(filepath, "a")
file.write("+-----------------------+----------------------+\n")
file.write("|                       | Faktyczna płeć osoby |\n")
file.write("|                       | na zdjęciu           |\n")
file.write("+                       +----------+-----------+\n")
file.write("|   Odpowiedź sieci     |  Kobieta | Mężczyzna |\n")
file.write("+-----------+-----------+----------+-----------+\n")
for x in range(1,26):

    FileName = ".\\baza_testowa\\k\\"+str(x)+".jpg"
    Img = imageio.imread(FileName)
    Img = (Img / 127.5) - 1
    BazaImg[0,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

    FileName = ".\\baza_testowa\\m\\"+str(x)+".jpg"
    Img = imageio.imread(FileName)
    Img = (Img / 127.5) - 1
    BazaImg[1,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

    gender = genderModel.predict(BazaImg) # 0 - m, 1 - k
    file.write("+ "+str('%.7f' % gender[0]))
    if gender[0] > 0.5:
        file.write(" |  Kobieta  |  Kobieta |           |\n")
        K += 1
    else:
        file.write(" | Mężczyzna |  Kobieta |           |\n")
        
    file.write("+-----------+-----------+----------+-----------+\n")    
    file.write("+ "+str('%.7f' % gender[1]))  
    if gender[1] > 0.5:
        file.write(" |  Kobieta  |          | Mężczyzna |\n")
    else:
        file.write(" | Mężczyzna |          | Mężczyzna |\n")
        M += 1
    file.write("+-----------+-----------+----------+-----------+\n")
    L += 1
file.write("Poprawnie zidentyfikowanych kobiet: "+str(K)+" z "+str(L)+" jest to "+str((K/L))+" %\n")
file.write("Poprawnie zidentyfikowanych mężczyzn: "+str(M)+" z "+str(L)+" jest to "+str((M/L))+" %\n")
file.write("Skuteczność sieci: "+str((M+K))+" z "+str(2*L)+" jest to "+str((M+K)/(2*L))+" %\n")
file.close()

#50
#--------------------------------
# Prawdziwa Płeć | Odpowiedz sieci
#----------------+---------+------
#                |   M     |  K
#----------------+---------+------
#                |         |  
#----------------+---------+------
#                |         |  
#----------------+---------+------
#                |         |  
#----------------+---------+------
#                |         |  
#----------------+---------+------
#                |         |
#----------------+---------+------
#                |         |  
#----------------+---------+------
#                |         |  
#----------------+---------+------
#                |         |
#----------------+---------+------
#                |         |  
#----------------+---------+------
#                |         |  
#----------------+---------+------
#                |         |
#----------------+---------+------
#                |         |  
#----------------+---------+------
#                |         |   
#----------------+---------+------
#                |         |
#----------------+---------+------
#                |         |   
#----------------+---------+------
#                |         |   
#----------------+---------+------
#                |         |




#ImgCount = 100
#ImgWidth = 100
#ImgHeight = 100
#
#BazaImg = np.empty((ImgCount,ImgHeight,ImgWidth,3))
#BazaAns = np.empty((ImgCount)) # 0 - m, 1 - k
#dirList = os.listdir(".\\baza\\m")
#i=0
#for dir in dirList:
#  print(dir)
#  FileName = ".\\baza\\m\\{}".format(dir)
#  Img = imageio.imread(FileName)
#  Img = (Img / 127.5) - 1
#  BazaImg[i,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]
#  BazaAns[i] = 0
#  i=i+1
#  if i>=ImgCount/2:
#    break
#
#dirList = os.listdir(".\\baza\\k")
#for dir in dirList:
#  print(dir)
#  FileName = ".\\baza\\k\\{}".format(dir)
#  Img = imageio.imread(FileName)
#  Img = (Img / 127.5) - 1
#  BazaImg[i,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]
#  BazaAns[i] = 1
#  i=i+1
#  if i>=ImgCount:
#    break
#
#BazaImg = BazaImg[0:i,:,:,:]
#BazaAns = BazaAns[0:i]
#print(BazaImg.shape)
#
##Stworzenia modelu sieci
#
#input  = keras.engine.input_layer.Input(shape=(ImgHeight,ImgWidth,3),name="wejscie")
#
#FlattenLayer = keras.layers.Flatten()
#
#path = FlattenLayer(input)
#
#for i in range(0,6):
#  LayerDense1 = keras.layers.Dense(50, activation=None, use_bias=True, kernel_initializer='glorot_uniform')
#  path = LayerDense1(path)
#
#  LayerPReLU1 = keras.layers.PReLU(alpha_initializer='zeros', shared_axes=None)
#  path = LayerPReLU1(path)
#
#LayerDenseN = keras.layers.Dense(1, activation=keras.activations.sigmoid, use_bias=True, kernel_initializer='glorot_uniform')
#output = LayerDenseN(path)
#
##---------------------------------
## Creation of TensorFlow Model
##---------------------------------
#genderModel = keras.Model(input, output, name='genderEstimatior')
#
#genderModel.summary() # Display summary
#
##Włączenia procesu uczenia
#
#rmsOptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#
#genderModel.compile(optimizer=rmsOptimizer,loss=keras.losses.binary_crossentropy,metrics=['accuracy'])
#
#genderModel.fit(BazaImg, BazaAns, epochs=15, batch_size=10, shuffle=True)
#
#genderModel.save('siec.h5')
##Przetestować / użyć sieci
#
