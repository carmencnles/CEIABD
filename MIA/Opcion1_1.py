import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import time

# ── Configuración ────────────────────────────────────────────────────────────
EMOTION_SKIP = 3          # Analizar emoción cada N frames para fluidez
CONF_THRESH  = 0.5        # Confianza mínima YOLO

EMOTION_COLORS = {
    'happy': (0, 220, 90), 'sad': (200, 80, 40), 'angry': (30, 30, 220),
    'surprise': (0, 200, 255), 'fear': (160, 30, 160), 'disgust': (30, 160, 80),
    'neutral': (160, 160, 160)
}

# ── Helpers de Dibujo ────────────────────────────────────────────────────────
def draw_ui(img, x1, y1, x2, y2, label, color, conf):
    # Rectángulo elegante
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # Etiqueta superior
    cv2.rectangle(img, (x1, y1-30), (x2, y1), color, -1)
    cv2.putText(img, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    # Barra de confianza inferior
    cv2.rectangle(img, (x1, y2+5), (x2, y2+10), (40,40,40), -1)
    w = int((x2 - x1) * conf)
    cv2.rectangle(img, (x1, y2+5), (x1+w, y2+10), color, -1)

# ── Buscador de Cámara Robusto ───────────────────────────────────────────────
def get_camera():
    # En Linux, podemos abrir el archivo del dispositivo como si fuera un video
    dispositivos = ["/dev/video0", "/dev/video1", "/dev/video2"]
    
    for dev in dispositivos:
        print(f"[INFO] Intentando abrir acceso directo a {dev}...")
        # Pasamos la cadena de texto directamente
        cap = cv2.VideoCapture(dev) 
        
        if cap.isOpened():
            # Hacemos una prueba de lectura real
            ret, frame = cap.read()
            if ret:
                print(f"[¡LOGRADO!] Conectado con éxito a {dev}")
                return cap
            else:
                print(f"[ADVERTENCIA] {dev} se abrió pero no devuelve imagen.")
            cap.release()
    return None

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("[INFO] Cargando YOLOv8 Face...")
    # Usamos el modelo nano de caras para máxima velocidad en CPU
    face_model = YOLO("yolov8n.pt") 
    
    cap = get_camera()
    if not cap:
        print("[ERROR] No se pudo acceder a ninguna cámara. Revisa permisos sudo.")
        return

    # Ajustar resolución para mejor rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    emotion_cache = {}
    
    print("[INFO] Iniciando bucle. Presiona 'Q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        results = face_model(frame, conf=CONF_THRESH, verbose=False)
        boxes = results[0].boxes

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Recorte para DeepFace
            face_roi = frame[max(0,y1):y2, max(0,x1):x2]
            
            emotion_label = "Detectando..."
            color = (200, 200, 200)

            # Analizar emoción solo cada N frames para no congelar el video
            if face_roi.size > 0 and (frame_count % EMOTION_SKIP == 0 or idx not in emotion_cache):
                try:
                    res = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                    emotion_label = res[0]['dominant_emotion']
                    emotion_cache[idx] = emotion_label
                except:
                    emotion_label = emotion_cache.get(idx, "neutral")
            else:
                emotion_label = emotion_cache.get(idx, "neutral")

            color = EMOTION_COLORS.get(emotion_label, (200, 200, 200))
            draw_ui(frame, x1, y1, x2, y2, emotion_label.upper(), color, conf)

        # Info HUD
        cv2.putText(frame, f"Caras: {len(boxes)} | FPS: {int(cap.get(cv2.CAP_PROP_FPS))}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Deteccion IABD - Presiona Q", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()