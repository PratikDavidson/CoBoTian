import streamlit as st
import backend as K

st.set_page_config(page_title="CoboTian", layout="centered")


model = K.init_model()

st.markdown("<h1 style='text-align: center;'>CoboTian</h1>", unsafe_allow_html=True)
st.subheader('', divider='gray')

prompt = st.chat_input("Give a phrase")
if prompt:
    
    with st.chat_message("user"):
        st.write(prompt)

    output = model.process(prompt)

    with st.chat_message("ai"):
        st.write_stream(K.stream_data(output))