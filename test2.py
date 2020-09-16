from create_video import best_frame, scale_image
import cv2


out_path = 'data/test.mp4'
video_path = 'data/1.mp4'

cascade = cv2.CascadeClassifier('data/haarcascade.xml')
coord, max_long = best_frame(video_path, cascade)
print(max_long)
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
codec = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(out_path, codec, fps, (256, 256))
print(max_long)

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame = scale_image(frame, coord, max_long)
    frame = cv2.resize(frame, (256, 256))
    out.write(frame)
print(1)
out.release()
video.release()
