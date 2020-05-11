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
import csv
table={}
table2={}
#przypadki={}

i=0
j=0
k=0
with open('dane.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if i>=12552 and i<=12601: #dla PL i>=12552 and i<=12601:
          table[j]=row
          j=j+1
        if i>=12602 and i<=12608: #dla PL i>=12552 and i<=12601:
          table2[k]=row
          k=k+1
        i=i+1

przypadki=np.empty(j)
BazaPred = np.empty((1,7,1))

for x in range(0,j):
  przypadki[x]=float (table[x][4])

for x in range(0,k):
  BazaPred[0,x,0]=float (table2[x][4])

BazaSize=len(przypadki)
Baza = np.empty((BazaSize-7,7,1))
BazaAns = np.empty((BazaSize-7))
for m in range(0,BazaSize-7):
  for n in range(0,7):
    Baza[m,n,:]=przypadki[m+n]
    if n==6:
      BazaAns[m]=przypadki[m+n+1]
#normalizacja  
MaxAns=max(przypadki)
MaxAns=MaxAns/2
Baza=(Baza/MaxAns)-1
BazaAns=(BazaAns/MaxAns)-1
BazaPred=(BazaPred/MaxAns)-1
#
Baza = Baza[:,:,:]
BazaAns = BazaAns[:]
BazaPred = BazaPred[:,:,:]
print(Baza.shape)

#Stworzenia modelu sieci

input  = keras.engine.input_layer.Input(shape=(7,1),name="wejscie") #rozmiar bazy 7,1

FlattenLayer = keras.layers.Flatten()

path = FlattenLayer(input)

for i in range(0,6):
  LayerDense1 = keras.layers.Dense(20, activation=None, use_bias=True, kernel_initializer='glorot_uniform')
  path = LayerDense1(path)

  LayerPReLU1 = keras.layers.PReLU(alpha_initializer='zeros', shared_axes=None)
  path = LayerPReLU1(path)

LayerDenseN = keras.layers.Dense(1, activation=None, use_bias=True, kernel_initializer='glorot_uniform')
output = LayerDenseN(path)

#---------------------------------
# Creation of TensorFlow Model
#---------------------------------
new_casesModel = keras.Model(input, output, name='COVID_Estimatior')

new_casesModel.summary() # Display summary

#Włączenia procesu uczenia

rmsOptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

new_casesModel.compile(optimizer=rmsOptimizer,loss=keras.losses.mean_absolute_error)

new_casesModel.fit(Baza, BazaAns, epochs=150, batch_size=10, shuffle=True)

new_casesModel.save('virus.h5')
#Przetestować / użyć sieci
covid = new_casesModel.predict(BazaPred)
print((covid[0]+1)*MaxAns)
filepath = ".\\wynik.txt"
file = open(filepath, "a")
file.write("Przewidywana liczba zachorowań 11.05.2020: \n")
file.write(str('%.1f' % ((covid[0]+1)*MaxAns)))
file.close()