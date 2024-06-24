import streamlit as st
import pandas as pd
import sys
import os

import altair as alt
from openai import OpenAI
# .streamlit/secrets.toml


# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将上一级目录添加到sys.path
sys.path.append(parent_dir)
# 现在可以导入predict模块

from predict import predict
from predict_analyse import predict_binary_images




uploaded_files = st.file_uploader("Choose pictures for detection", accept_multiple_files=True)

col = st.columns(6)
idx=0
tab = st.tabs(["Chart", "Data","Summery"])






uploaded_files_data = []
for uploaded_file1 in uploaded_files:
    bytes_data = uploaded_file1.read()
    uploaded_files_data.append(bytes_data)

    with col[int(idx)]:
        idx = idx + 1
        idx = idx % 6
        #st.image(bytes_data, width=100)



#上传字节码


#预测
weights_path = './weights/model-172.pth'
predicted_classes, class_count = predict_binary_images(uploaded_files_data, weights_path)

#for idx, class_name in enumerate(predicted_classes):
#    #st.image(class_name, width=100)
#    st.write(f"Image {idx + 1}: Predicted class - {class_name}")




with tab[0]:
    # 创建DataFrame
    class_count_df = pd.DataFrame(list(class_count.items()), columns=['Class', 'Count'])
    # 使用Altair创建柱形图
    chart = (alt.Chart(class_count_df)
    .mark_bar(
        color='red',
        size=20
    ).encode(
        x=alt.X('Class', sort=None),
        y='Count'
    ).properties(
        title='distribution of disease',
        width=500,  # 设置图表宽度
        height=600  # 设置图表高度
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=16
    ).configure_title(
        fontSize=20,
        anchor='start',
        color='black'
    ))

    # 显示柱形图
    st.altair_chart(chart, use_container_width=True)

with tab[1]:
    # 显示类别计数
    for class_name, count in class_count.items():
        st.write(f"{class_name}: {count}")
    #st.write(final_result)

    messages = st.container(height=300)

        #messages.chat_message("user").write(prompt)
        #messages.chat_message("assistant").write(f"Echo: {prompt}")

with tab[2]:
    st.title("ChatGPT-like clone")

    client = OpenAI(api_key="d1b5cbeb-9e40-4303-9a38-911abe7fc550",base_url = "https://ark.cn-beijing.volces.com/api/v3",)

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "ep-20240615141335-ljtxc"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})