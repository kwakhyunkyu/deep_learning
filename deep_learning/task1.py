import cv2
import dlib

# 캡처 객체 선언. 0번 웹캠으로부터 영상을 가져옴.
cap = cv2.VideoCapture(0)

# dlib의 기본 얼굴 검출기 불러오기.
detector = dlib.get_frontal_face_detector()
# 사전 학습된 랜드마크 예측기 불러오기
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 무한루프
while True:
    # 캡처 객체로부터 frame 받아옴
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        print(landmarks)

        # 얼굴 랜드마크를 시각화. 모든 64개의 랜드마크에 대하여 반복
        for i in range(0,64):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

    # 해당 프레임에 얼굴 랜드마크 시각화
    # Landmarks라는 창에 띄움
    cv2.imshow("Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
