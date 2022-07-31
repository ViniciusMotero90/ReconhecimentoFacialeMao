import cv2
import mediapipe as mp 
import time

camera = cv2.VideoCapture(0)
mpMaos = mp.solutions.hands
maos   = mpMaos.Hands()
mpDesenho = mp.solutions.drawing_utils


reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rosto = reconhecimento_rosto.FaceDetection()

tic = 0
tac = 0

while True:
    sucesso, imagem = camera.read()
    imagemRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    resultados = maos.process(imagemRGB)
    lista_rostos = reconhecedor_rosto.process(imagem)
    
    if lista_rostos.detections: # caso algum rosto tenha sido reconhecido
        for rosto in lista_rostos.detections: # para cada rosto que foi reconhecido
            mpDesenho.draw_detection(imagem, rosto)
    
    if resultados.multi_hand_landmarks:
        for maosPntRef in resultados.multi_hand_landmarks:
            mpDesenho.draw_landmarks(imagem, maosPntRef, mpMaos.HAND_CONNECTIONS)
    
    tac = time.time()
    fps = 1/(tac-tic)
    tic = tac

    cv2.putText(imagem, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Câmera", imagem)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break