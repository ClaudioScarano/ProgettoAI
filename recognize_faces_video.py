# ************************************************************************************************************** #
#                                                                                                                #
# Autore : Claudio Scarano                                                                                       #
# Email  : //                                                          #
# Matr   : 475HHHINGINFOR                                                                                        #
#           	                                                                                                   #
#                                                                                                                #
# Convolutional Neural Networks (CNN)                                                                            #
# Rilevamento del volto utilizzando una Deep neural network Convolutional neural network (CNN) con OpenCV e dlib #
#                                                                                                                #
# Descrizione : modulo riconoscimento facciale                                                                   #
# Il codice principale del rilevamento del volto utilizzando la Convolutional neural network (CNN):              #
# una classe delle reti neurali profonde                                                                         #
# ************************************************************************************************************** #


# Nel terminale lanciamo il seguente comando per consentire il riconoscimeno dei volti
# il primo comando non esegue la cattura del flusso video mentre il secondo salva una clip su disco

# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/cattura.avi --display 1

# Importiamo le librerie necessarie
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# imposto argparse e analizzo gli argomenti da utilizzare
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="percorso al db serializzato di codifiche facciali")
ap.add_argument("-o", "--output", type=str,
	help="percorso per l'output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="imposta se vedere l'output a schermo oppure no mettendo zero al posto di 1")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="modello di rilevamento del volto da usare CNN")
args = vars(ap.parse_args())

# carica i volti noti e gli embedding
print("[INFO] Caricamento codifiche...")
data = pickle.loads(open(args["encodings"], "rb").read())

#inizializza lo stream video  and il puntatore all'output del file video, successivamente
# consente al sensore della telecamere di avviarsi
print("[INFO] Inizio flusso video...")
#Seguono due differenti url di test di due ip camera diverse
# url = 'rtsp://192.168.1.250:8080/h264_pcm.sdp'
# url = 'rtsp://admin:123456@192.168.1.4/live/ch0'
# sostituire nel src=0 in src=url per avviare il video da ip camera anzichè da webcam utilizzata nel test iniziale

vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# cicla sui frame dal file del flusso video
while True:
	# prende il frame dallo stream video
	frame = vs.read()

	# converti il frame di input da BGR a RGB, quindi ridimensionalo per avere
	# una larghezza di 750px (per velocizzare l'elaborazione)
	bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# rileva le coordinate (x, y) delle finestre di delimitazione
	# corrispondente a ciascuna faccia nel riquadro di input, quindi calcola
	# le embeddings facciali per ogni viso
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# cicla sugli embeddings facciali
	for encoding in encodings:
		# tenta di far corrispondere ciascuna faccia nell'immagine di input alle nostre codifiche note

		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# controlla se abbiamo trovato una corrispondenza
		if True in matches:
			# trova gli indici di tutte le facce abbinate quindi inizializza un
			# dizionario per contare il numero totale di volte in cui ogni faccia è stata abbinata
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# passa in rassegna gli indici corrispondenti e mantiene un conteggio per
			# ogni volto riconosciuto
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determina il volto riconosciuto con il maggior numero di voti
			# (nota: in caso di pareggio improbabile Python selezionerà la prima voce nel dizionario)
			name = max(counts, key=counts.get)
		
		# aggiorna l'elenco dei nomi
		names.append(name)

	# cicla sui volti riconosciuti
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# ridimensiona le coordinate del viso
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		#  disegna il nome del viso previsto sull'immagine
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	#  Se writer è uguale a 'None' scriviamo il video di output su disco e inizializzo writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)

	# se writer non è uguale a 'None' scriviamo il frame con i volti riconosciuti su disco
	if writer is not None:
		writer.write(frame)

	# controlla se dovremmo visualizzare il frame di output sullo schermo
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# se è stato premuto il tasto `q`, interrompi il loop
		if key == ord("q"):
			break


cv2.destroyAllWindows()
vs.stop()

# controlla se è necessario rilasciare il writer video
if writer is not None:
	writer.release()
