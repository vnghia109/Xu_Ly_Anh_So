import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import tensorflow as tf
from PIL import Image
import numpy as np


model = tf.keras.applications.InceptionV3(input_shape=(299, 299, 3), include_top=True, weights='imagenet')


def predict(image):
    
    image1 = np.array(Image.open(image))
    img = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    ##img = cv2.imread(image)
    img = cv.resize(img, (299, 299))
    img = np.reshape(img, [1, 299, 299, 3])
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    prediction = model.predict(img)
    predicted_class = tf.keras.applications.imagenet_utils.decode_predictions(prediction)
    animal = predicted_class[0][0][1]

    return animal



def NhanDienDongVat():
    st.subheader("Nhận diện động vật")

    uploaded_file = st.file_uploader("Chọn một file ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))

        st.image(
            image, caption=f"Ảnh được tải lên", use_column_width=True
        )

        with st.spinner("Đang phân loại động vật..."):
            animal = predict(uploaded_file)

        st.success(f"Loài động vật dự đoán: {animal.title()}")


def phatHien():
    st.subheader('Phát hiện khuôn mặt')
    FRAME_WINDOW = st.image([])
    deviceId = 0
    cap = cv.VideoCapture(deviceId)


    if 'stop' not in st.session_state:
        st.session_state.stop = False
        stop = False

    press = st.button('Stop', key = "btn1")
    if press:
        if st.session_state.stop == False:
            st.session_state.stop = True
            cap.release()
        else:
            st.session_state.stop = False

    print('Trang thai nhan Stop', st.session_state.stop)

    if 'frame_stop' not in st.session_state:
        frame_stop = cv.imread('stop.jpg')
        st.session_state.frame_stop = frame_stop
        print('Đã load stop.jpg')

    if st.session_state.stop == True:
        FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')


    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    detector = cv.FaceDetectorYN.create(
        'face_detection_yunet_2022mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    tm = cv.TickMeter()
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        frame = cv.resize(frame, (frameWidth, frameHeight))

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()
    
def NhanDien():
    st.subheader('Nhận dạng khuôn mặt')
    FRAME_WINDOW = st.image([])
    cap = cv.VideoCapture(0)

    if 'stop' not in st.session_state:
        st.session_state.stop = False
        stop = False

    press = st.button('Stop', key = "btn2")
    if press:
        if st.session_state.stop == False:
            st.session_state.stop = True
            cap.release()
        else:
            st.session_state.stop = False

    print('Trang thai nhan Stop', st.session_state.stop)

    if 'frame_stop' not in st.session_state:
        frame_stop = cv.imread('stop.jpg')
        st.session_state.frame_stop = frame_stop
        print('Đã load stop.jpg')

    if st.session_state.stop == True:
        FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')


    svc = joblib.load('svc.pkl')
    mydict = ['BanKiet', 'BanNghia', 'BanNguyen', 'BanThanh', 'SangSang', 'ThayDuc']

    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    detector = cv.FaceDetectorYN.create(
        'face_detection_yunet_2022mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    
    recognizer = cv.FaceRecognizerSF.create(
    'face_recognition_sface_2021dec.onnx',"")

    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        
        if faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            cv.putText(frame,result,(1,50),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()

def main():
    st.title("Đồ án cuối kỳ xử lý ảnh số")
    menu = ["Phát hiện khuôn mặt", "Nhận diện khuôn mặt", "Nhận diện động vật"]
    page = st.sidebar.selectbox("Menu",menu)


    # Hiển thị nội dung của từng tab khi người dùng chọn
    if page == "Phát hiện khuôn mặt":
        phatHien()
    elif page == "Nhận diện khuôn mặt":
        NhanDien()
    elif page == "Nhận diện động vật":
        NhanDienDongVat()
        
if __name__ == '__main__':
    main()
