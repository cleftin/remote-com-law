import streamlit as st
from dotenv import load_dotenv
from llm-law import get_ai_message

load_dotenv()


st.set_page_config(page_title="통신관련 법률 챗봇", page_icon = "🤖")

st.title("🤖 통신관련 법률 챗봇")
st.caption("통신관련 법률 검색!")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])    

if user_question := st.chat_input(placeholder="통신관련 법률을 검색하세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})   

    with st.spinner("답변을 생성하는 중입니다."):
        ai_message = get_ai_message(user_question)
       
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})   