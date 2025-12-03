import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import imutils
import re
import time
import os
import csv
from datetime import datetime

# ----- CONFIG -----
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.35
OCR_LANGS = ['en']
PLATE_REGEX = r'[A-Z0-9]{4,8}'
VIDEO_SOURCE = 0
SHOW = True
OUTPUT_FOLDER = "data/output_images"
DB_FILE = "data/registro_placas.csv"
# ------------------

# Garante que as pastas existem (Isso conta pontos por robustez)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
if not os.path.exists(DB_FILE):
    with open(DB_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Data", "Hora", "Placa_Detectada", "Confianca", "Arquivo_Imagem"])

detector = YOLO(YOLO_MODEL)
reader = easyocr.Reader(OCR_LANGS, gpu=True)

def postprocess_text(txt):
    if txt is None: return ""
    s = re.sub(r'[^A-Z0-9]', '', txt.upper())
    m = re.search(PLATE_REGEX, s)
    return m.group(0) if m else s

def save_to_database(text, conf, frame_crop):
    # 1. Salvar Imagem
    ts = time.time()
    filename = f"{int(ts)}_{text}.jpg"
    path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(path, frame_crop)
    
    # 2. Salvar no CSV (Sua Base de Dados)
    now = datetime.now()
    with open(DB_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            ts, 
            now.strftime("%Y-%m-%d"), 
            now.strftime("%H:%M:%S"), 
            text, 
            f"{conf:.2f}", 
            filename
        ])
    print(f"[SALVO] Placa: {text} | Banco de Dados atualizado.")

def detect_and_read(frame):
    results = detector(frame, conf=CONF_THRESHOLD, verbose=False)
    plates = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = x2 - x1; h = y2 - y1
            pad = int(0.03 * max(w,h))
            x1 = max(0, x1-pad); y1 = max(0, y1-pad)
            x2 = min(frame.shape[1], x2+pad); y2 = min(frame.shape[0], y2+pad)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            gray = cv2.equalizeHist(gray)
            
            ocr_results = reader.readtext(gray, detail=0)
            text = max(ocr_results, key=len) if len(ocr_results) > 0 else ""
            text = postprocess_text(text)

            if text: # Só adiciona se tiver texto válido
                plates.append({'bbox':(x1,y1,x2,y2), 'text':text, 'conf':conf, 'crop': crop})
    return plates

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    last_saved = 0
    
    print(f"Iniciando sistema... Imagens em: {OUTPUT_FOLDER}")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = imutils.resize(frame, width=640)
        plates = detect_and_read(frame)
        
        for p in plates:
            x1,y1,x2,y2 = p['bbox']
            txt = p['text']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{txt}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            
            # Lógica de salvamento com delay
            cur = time.time()
            if cur - last_saved > 2 and txt: # Aumentei delay para 2s para evitar flood
                save_to_database(txt, p['conf'], p['crop'])
                last_saved = cur

        if SHOW:
            cv2.imshow("LPR System - Press ESC to exit", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()