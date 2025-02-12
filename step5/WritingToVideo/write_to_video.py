import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')  
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640 * 2, 480 * 2))  

if not cap.isOpened():
    print("Web kamerası açılamadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare okunamadı!")
        break

    zeros = np.zeros(frame.shape[:2], dtype="uint8")
    
    (B, G, R) = cv2.split(frame)
    R = cv2.merge([zeros, zeros, R])
    G = cv2.merge([zeros, G, zeros])
    B = cv2.merge([B, zeros, zeros])

    h, w = frame.shape[:2]
    output = np.zeros((h * 2, w * 2, 3), dtype="uint8")
    output[0:h, 0:w] = frame    # Sol üst - Orijinal
    output[0:h, w:w*2] = R      # Sağ üst - Kırmızı kanal
    output[h:h*2, 0:w] = G      # Sol alt - Yeşil kanal
    output[h:h*2, w:w*2] = B    # Sağ alt - Mavi kanal

    out.write(output)

    cv2.imshow("Renk Kanalları ile Çıkış", output)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  
        break

cap.release()
out.release()
cv2.destroyAllWindows()
