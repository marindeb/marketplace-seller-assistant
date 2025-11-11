import requests
import streamlit as st


st.title("🧠 Mistral E-Commerce Agent")

question = st.text_input("Ask a question about the products:")
mode = st.selectbox("Mode", ("Auto", "RAG", "Agent"), index=0)

mode_param = None if mode == "Auto" else mode.lower()

if st.button("Ask"):
    with st.spinner("Thinking..."):
        st.write("if and with")
        response = requests.post(
            "http://127.0.0.1:8000/query",
            json={"question": question, "mode": mode_param},
        ).json()

st.write(
    f"**Answer :** {response.get('answer') if response.get('mode') == 'agent' else response.get('answer').get('result')}"
)
st.write(f"**Used mode :** {response.get('mode')}")
