import cv2

# Forzamos el uso de V4L2 (Video for Linux 2)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    # Intento 2: Ruta directa
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

if cap.isOpened():
    print("¡CONSEGUIDO! La cámara está abierta.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Prueba", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("ERROR: El sistema sigue bloqueando el acceso incluso con V4L2.")