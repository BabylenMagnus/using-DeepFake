from create_video import search_face
import cv2

image_path = 'data/3.jpg'
image = cv2.imread(image_path)
cascade = cv2.CascadeClassifier('data/haarcascade.xml')


def show_cv2(image):
    while True:
        cv2.imshow('Image', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


image = search_face(image, cascade, image=True)
show_cv2(image)
