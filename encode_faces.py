# ************************************************************************************************************** #
#                                                                                                                #
# Autore : Claudio Scarano                                                                                       #
# Email  : c.scarano1@students.uninettunouniversity.net                                                          #
# Matr   : 475HHHINGINFOR                                                                                        #
#                                                                                                                #
#                                                                                                                #
# Convolutional Neural Networks (CNN)                                                                            #
# Rilevamento del volto utilizzando una Deep neural network Convolutional neural network (CNN) con OpenCV e dlib #
#                                                                                                                #
# Descrizione : modulo codifica volti                                                                         #
# Il codice principale del rilevamento del volto utilizzando la Convolutional neural network (CNN):              #
# una classe delle reti neurali profonde                                                                         #
# ************************************************************************************************************** #


# Nel terminale lanciamo il seguente comando per consentire la codifica dei volti
# python encode_faces.py --dataset dataset --encodings encodings.pickle


# Importiamo le librerie necessarie

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# imposto argparse e analizziamo gli argomenti da utilizzare
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="Percorso della directory di volti e immagini")
ap.add_argument("-e", "--encodings", required=True,
	help="percorso al db serializzato di codifiche facciali")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="modello di rilevamento del volto da usare CNN")
args = vars(ap.parse_args())

# prende i percorsi delle immagini di input nel nostro dataset

print("[INFO] Sto quantificando i volti...")
imagePaths = list(paths.list_images(args["dataset"]))

# inizializza l'elenco di codifiche conosciute e nomi noti
knownEncodings = []
knownNames = []

# cicla sul percorso delle immagini
for (i, imagePath) in enumerate(imagePaths):
	# estrae il nome della persona dal percorso dell'immagine
	print("[INFO] processo immagine {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# carica l'immagine di input e la converte da BGR (ordinamento di OpenCV)
	# nell'ordinamento di dlib (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# rileva le coordinate (x, y) delle finestre di delimitazione
	# corrispondenti a ciascuna faccia nell'immagine di input
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# calcola l'incorporamento facciale per il viso
	encodings = face_recognition.face_encodings(rgb, boxes)

	# cicla sulle codifiche
	for encoding in encodings:
		# aggiungi ogni codifica e nome al nostro set di nomi noti e  codifiche
		knownEncodings.append(encoding)
		knownNames.append(name)

# scarica le codifiche facciali + i nomi sul disco
print("[INFO] serializzazione codifiche in corso...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
