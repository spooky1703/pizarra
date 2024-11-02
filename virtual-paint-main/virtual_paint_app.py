import mediapipe as mp
import cv2
import numpy as np
import time

# Constantes para la interfaz de herramientas y dibujo
ml = 150
max_x, max_y = 250 + ml, 50
curr_tool = "Selecciona herramienta"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0

# Función para obtener la herramienta según la posición del dedo índice (x)
def getTool(x):
    if x < 50 + ml:
        return "Recta"
    elif x < 100 + ml:
        return "Rectangulo"
    elif x < 150 + ml:
        return "Trazo_libre"
    elif x < 200 + ml:
        return "Circulo"
    else:
        return "Borrar_trazo"

# Función para verificar si el dedo índice está levantado
def index_raised(yi, y9):
    return (y9 - yi) > 40

# Inicialización de MediaPipe
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Cargar la imagen de las herramientas
tools = cv2.imread("tools.png").astype('uint8')

# Crear una máscara en blanco
mask = np.ones((480, 640), dtype='uint8') * 255

# Inicializar la captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Función para mostrar la ventana de instrucciones
def show_instructions():
    instructions_img = cv2.imread("instrucciones.png")
    if instructions_img is not None:
        # Redimensiona la imagen a 400 x 500 para que no se vea borrosa
        instructions_img = cv2.resize(instructions_img, (400, 500), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Instrucciones", instructions_img)
        cv2.waitKey(0)  # Espera hasta que se cierre manualmente la ventana
        cv2.destroyWindow("Instrucciones")
    else:
        print("Error: No se pudo cargar la imagen de instrucciones.")

# Bucle principal del programa
while True:
    ret, frm = cap.read()
    if not ret:
        print("Error: No se pudo leer el fotograma de la cámara.")
        break

    frm = cv2.flip(frm, 1)
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    # Detectar manos y herramientas
    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)
            cv2.circle(frm, (x, y), 5, (0, 0, 255), -1)

            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Tu herramienta actual es: ", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            # Lógica de herramientas
            if curr_tool == "Trazo_libre":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                    prevx, prevy = x, y
                else:
                    prevx = x
                    prevy = y

            elif curr_tool == "Recta":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True

                    cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)
                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "Rectangulo":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True

                    cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)
                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "Circulo":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True

                    cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (255, 255, 0), thick)
                else:
                    if var_inits:
                        cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), 0, thick)
                        var_inits = False

            elif curr_tool == "Borrar_trazo":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                    cv2.circle(mask, (x, y), 30, 255, -1)

    # Mostrar botón de "Instrucciones" en la esquina inferior derecha
    height, width = frm.shape[:2]
    button_text = "Instrucciones"
    button_position = (width - 200, height - 30)
    button_size = (width - 10, height - 10)
    cv2.rectangle(frm, button_position, button_size, (0, 0, 255), -1)
    cv2.putText(frm, button_text, (width - 180, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Detección de clic en el botón "Instrucciones"
    if cv2.waitKey(1) & 0xFF == ord('i'):
        show_instructions()

    # Mostrar el fotograma procesado
    frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)
    cv2.putText(frm, curr_tool, (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Dibujo version beta", frm)

    # Salir con la tecla 'ESC'
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
