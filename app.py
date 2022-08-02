import streamlit as st
import sqlite3
#import os.path

# # 현재 파일의 절대경로
# file_path = os.path.dirname(__file__)
# db_file_path = file_path + '/db.db'
#
# #DB연결
# file_exists = os.path.exists(db_file_path)
# con = None
# cur = None
#
# if file_exists:
#   con = sqlite3.connect(db_file_path)
#   cur = con.cusor()
#else:
#   st.error('db.db 파일이 존재하지 않습니다.')


con = sqlite3.connect('db.db')
cur = con.cursor()

st.subheader('회원가입 폼')

with st.form('my_form', clear_on_submit=True):
    st.info('다음 양식을 모두 입력 후 제출합니다.')
    uid = st.text_input('아이디', max_chars=12)
    uname = st.text_input('성명', max_chars=10)
    uemail = st.text_input('이메일')
    upw = st.text_input('비밀번호', type='password')
    upw_chk = st.text_input('비밀번호 확인', type='password')
    ubd = st.date_input('생년월일')
    ugender = st.radio('성별', options=['남', '여'], horizontal=True)

    submitted = st.form_submit_button('제출')
    if submitted:
        st.success(f'{uid} {uname} {upw} {ubd} {ugender}')
        cur.execute(f"INSERT INTO users VALUES ("
                    f"'{uid}', '{uname}', '{upw}',"
                    f"'{ubd}', '{ugender}', CURRENT_DATE)")
        con.commit()


#st.title("데이터과학 실습 과정")
#st.header("1일차")
#st.subheader("파이참 설정 및 사용법")
#st.sidebar.selectbox('메뉴', ['Home', '입력', '조회'])