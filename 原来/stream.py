import streamlit as st
import base64
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

# 设置标题和描述
st.markdown('<h1 style="color:black;">Vgg 19 Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>',
            unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> street,  buildings, forest, sea, mountain, glacier</h3>', unsafe_allow_html=True)


# 背景图片background image to streamlit
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# 设置背景图片,颜色等
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg('background.webp')

# 上传png/jpg的照片
upload = st.file_uploader('Insert image for classification', type=['png', 'jpg'])
c1, c2 = st.columns(2)
if upload is not None:
    im = Image.open(upload)
    img = np.asarray(im)
    image = cv2.resize(img, (224, 224))
    img = preprocess_input(image)
    img = np.expand_dims(img, 0)
    c1.header('Input Image')
    c1.image(im)
    c1.write(img.shape)

    # 下载预训练模型
    input_shape = (224, 224, 3)
    optim_1 = Adam(learning_rate=0.0001)
    n_classes = 6
    vgg_model = model(input_shape, n_classes, optim_1, fine_tune=2)
    vgg_model.load_weights('tune_model19.weights.best.hdf5')

    # 预测
    vgg_preds = vgg_model.predict(img)
    vgg_pred_classes = np.argmax(vgg_preds, axis=1)
    c2.header('Output')
    c2.subheader('Predicted class :')
    c2.write(classes[vgg_pred_classes[0]])


def model(input_shape, n_classes, optimizer, fine_tune=0):
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    if fine_tune > 0:
        for layer in base_model.layers[:-fine_tune]:
            layer.trainable = False
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


classes = ['street', 'buildings', 'forest', 'sea', 'mountain', 'glacier']
