import os.path

import cv2
import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# 경로 설정
file_path = os.path.dirname(__file__)

# 모델파일 폴더 생성
save_dir = os.path.join(file_path, 'model')

# 학습된 모델 불러오기
model_file = 'minist_model.h5'
# model = load_model(os.path.join(save_dir, model_file))
model = load_model(model_file)    # 모델 파일이 같은 폴더 안에 있으므로

# 헤더 출력
st.subheader('손글씨 숫자 인식')

SIZE = 192

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',             # 글자색을 흰색으로
    background_color='#000000',         # 백그라운드 칼라를 검정색
    width=SIZE,
    height=SIZE,
    drawing_mode='freedraw',
    update_streamlit=False,
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))            # 결과 이미지 파일을 28X28로 리사이즈
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)       # 리사이즈 한 것을 다시 192X192로 키움
    st.write('모델 입력 형태')
    st.image(rescaled)                                                              # 줄여놓고 키운 것을 화면에 보여줌(실제 학습에 사용할 데이터)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              # 채널 3개인 이미지를 흑백 이미지로 바꿈
    res = model.predict(np.reshape(test_x, (1, 28 * 28)))
    st.success(np.argmax(res[0]))
    st.bar_chart(res[0])