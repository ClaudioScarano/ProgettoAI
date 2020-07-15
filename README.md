                                                                                                              
 Convolutional Neural Networks (CNN)                                                                            
 Rilevamento del volto utilizzando una Deep neural network Convolutional neural network (CNN) con OpenCV e dlib 
                                                                                                                
 Descrizione : modulo codifica_volti.py                                                                         
 Il codice principale del rilevamento del volto utilizzando la Convolutional neural network (CNN):              
 una classe delle reti neurali profonde                                                                         
 


# OpenCV
Il modulo OpenCV supporta l'inferenza di esecuzione su modelli di deep learning pre-addestrati da framework popolari come TensorFlow, Torch, Darknet e Caffe.

# Dlib
La libreria dlib è una libreria open source scritta in C++ che implementa numerose
tecniche di image processing e machine learning. Più nello specifico essa
è stata utilizzata per le sue capacità in ambito di face detection, face
alignment e feature extraction.

# Imutils
Gli Imutils sono una serie di funzioni utili per rendere più semplici le funzioni di elaborazione delle immagini di base come traduzione, rotazione, ridimensionamento, scheletro e visualizzazione di immagini Matplotlib con OpenCV e sia Python 2.7 che Python 3

## Requisiti
<ul>
<li>Python 3.6</li>
<li>OpenCV 4.2.0</li>
<li>argparse</li>
<li>pickle</li>
<li>imutils</li>
<li>dlib</li>
<li>face_recognition</li>
</ul>


## Dipendenze
<ul>
<li>from imutils import paths</li>
import face_recognition</li>
import argparse</li>
import pickle</li>
import cv2</li>
import os</li>
</ul>

## Installare dipendenze
<p><code>pip install opencv-python</code></p>
<p><code>pip pickle</code></p>
<p><code>pip install imutils</code></p>
<p><code>pip install argparse</code></p>

Si raccomanda di compilare dlib con cmake con il supporto CUDA e con il modulo CuDNN e successivamente installarlo
La GPU rende magnificamente rispetto alla CPU.
