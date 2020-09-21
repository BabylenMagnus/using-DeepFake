import cv2
import argparse


def compute_long(a, n, long, max_long):
    high_a = int(a - n) if a > n else 0
    high_b = high_a + 2 * n + long
    high_b = int(high_b) if high_b < max_long else max_long

    return high_a, high_b


def search_face(frame, cascade):
    faces = cascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=6
    )

    n = int(0.1 * frame.shape[0])

    if len(faces) == 0:
        return frame

    max_area_face = faces[0]

    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    face = max_area_face

    a, b = compute_long(face[1], n, face[2], frame.shape[0])

    c, d = compute_long(face[0], n, face[2], frame.shape[1])

    return frame[a:b, c:d]


def best_frame(video_path, cascade, initial=True):
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()

        faces = cascade.detectMultiScale(
            frame,
            scaleFactor=1.2,
            minNeighbors=6
        )

        if len(faces) == 0:
            continue

        max_area_face = faces[0]

        for face in faces:
            if face[2] > max_area_face[2]:
                max_area_face = face

        if initial:
            return tuple(max_area_face), frame
        return tuple(max_area_face)


def scale_image(image, coord, long, shape, n=0.3):
    n = int(n * shape[1])

    a = coord[1] - n if n < coord[1] else 0
    b = coord[1] + n + long
    b = b if b < shape[0] else shape[0]

    c = coord[0] - n if n < coord[0] else 0
    d = coord[0] + long + n
    d = d if d < shape[1] else shape[1]

    return image[a: b, c: d]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Use Deepfake")
    parser.add_argument('video', type=str, help="path to video")

    args = parser.parse_args()

    cascade = cv2.CascadeClassifier(r'data/haarcascade.xml')
    coord = best_frame(args.video, cascade)
    video = cv2.VideoCapture(args.video)

    while True:

        ret, frame = video.read()

        if not ret:
            break

        frame = scale_image(frame, coord, coord[2], frame.shape, n=0.1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
