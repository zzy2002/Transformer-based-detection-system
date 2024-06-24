

import streamlit as st
import pandas as pd
from predict import predict

uploaded_files = st.file_uploader("Choose a picture for detection", accept_multiple_files=True)
col = st.columns(2)
idx=0

with st.sidebar:
    st.write("made by")
    st.header("ZZY")

for uploaded_file in uploaded_files:
    #上传字节码
    with col[int(idx)]:
        idx=idx+1
        idx=idx%2
        bytes_data = uploaded_file.read()
        #st.write("filename:", uploaded_file.name)
        st.image(bytes_data,caption=uploaded_file.name,width=300)

        #预测并转成int的dataframe
        predict_results,final_result,labels=predict(bytes_data)
        #st.write(predict_results)
        st.write(" ")
        st.header(final_result)
        st.write(" ")
        results_int = predict_results.numpy()
        #st.write(labels)
        data_df = pd.DataFrame({"kind": labels,"propotion": results_int})

    with col[int(idx)]:
        idx = idx + 1
        idx = idx % 2
        #展示bar
        st.data_editor(
            data_df,
            column_config={
                "propotion": st.column_config.ProgressColumn(
                    "probability of each kind",
                    help="probability",
                    format="%%",
                    min_value=0,
                    max_value=1,
                    width="medium",
                    #sort_order = "descending"
                ),
            },
            hide_index=True,
        )



