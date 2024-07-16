# import streamlit as st
# from keras.models import load_model
# from keras.layers import Dropout
# from keras.utils import custom_object_scope
# from PIL import Image, ImageOps
# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# import pywt
# import cv2
# from io import BytesIO
# import os

# # FixedDropout - Kerasning maxsus dropout qatlami
# class FixedDropout(Dropout):
#     def __init__(self, rate, seed=None, **kwargs):
#         super(FixedDropout, self).__init__(rate, **kwargs)
#         self.seed = seed

#     def call(self, inputs, training=None):
#         return super(FixedDropout, self).call(inputs, training=training)

# # ImmProcessor sinfi - tasvirni qayta ishlash uchun
# class ImmProcessor:
#     def __init__(self, img, img_size=640, num_clusters=2):
#         self.img_size = img_size  # Tasvir o'lchami
#         self.num_clusters = num_clusters  # K-means klasterlar soni
#         self.img = cv2.resize(img, (self.img_size, self.img_size))  # Tasvirni qayta o'lchash

#     # Tasvirni oldindan qayta ishlash (kulrangga aylantirish va histogramni tenglashtirish)
#     def preprocess_image(self):
#         img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
#         equ = cv2.equalizeHist(img_gray)
#         return equ

#     # Tasvirga DWT (Diskret Wavelet Transform) qo'llash
#     def apply_dwt(self, img):
#         coeffs = pywt.dwt2(img, 'haar')
#         equ2 = pywt.idwt2(coeffs, 'haar')
#         return np.array(equ2)

#     # Mos keluvchi filtr bankini qo'llash
#     def apply_matched_filter_bank(self, img, bank):
#         equ3 = self._apply_filters(img, bank)
#         return np.array(equ3)

#     # K-means clustering qo'llash
#     def apply_kmeans(self, img):
#         Z = img.flatten().reshape((-1, 1))
#         Z = np.float32(Z)

#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#         _, label, center = cv2.kmeans(Z, self.num_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

#         center = np.uint8(center)
#         res = center[label.flatten()]
#         res2 = res.reshape((img.shape))

#         return res2

#     # Oddiy qayta ishlash (oldindan qayta ishlash va DWT)
#     def process_image(self):
#         processed_img = self.preprocess_image()
#         dwt_img = self.apply_dwt(processed_img)
#         return dwt_img

#     # Filtrlar bilan qayta ishlash
#     def process_image_with_filters(self):
#         processed_img = self.preprocess_image()
#         dwt_img = self.apply_dwt(processed_img)
#         filter_bank = self._create_matched_filter_bank()
#         filtered_img = self.apply_matched_filter_bank(dwt_img, filter_bank)
#         return filtered_img

#     # K-means bilan qayta ishlash
#     def process_image_with_kmeans(self):
#         processed_img = self.preprocess_image()
#         dwt_img = self.apply_dwt(processed_img)
#         kmeans_img = self.apply_kmeans(dwt_img)
#         return kmeans_img

#     # Filtrlarni qo'llash (ichki yordamchi funksiya)
#     def _apply_filters(self, img, kernels):
#         images = np.array([cv2.filter2D(img, -1, k) for k in kernels])
#         return np.max(images, 0)

#     # Natijalarni vizualizatsiya qilish
#     def visualize_results(self):
#         processed_img = self.preprocess_image()
#         dwt_img = self.apply_dwt(processed_img)
#         filter_bank = self._create_matched_filter_bank()
#         filtered_img = self.apply_matched_filter_bank(dwt_img, filter_bank)
#         kmeans_img = self.apply_kmeans(dwt_img)

#         images = [self.img, dwt_img, filtered_img, kmeans_img]
#         titles = ['Original', 'Diskret Wavelet Transform', 'Filtered (Matched Filter Bank)', 'K-means Clustering']

#         return self._get_images_as_buffers(images, titles)

#     # Natijalarni buferlarga olish (vizualizatsiya uchun yordamchi funksiya)
#     def _get_images_as_buffers(self, images, titles, scale=1.3):
#         buffers = []
#         fig, axs = plt.subplots(1, len(images), figsize=(20, 8))
#         for i, (image, title) in enumerate(zip(images, titles)):
#             axs[i].imshow(image, cmap='gray')
#             axs[i].set_title(title, fontsize=10)
#             axs[i].axis('off')
#         plt.tight_layout()

#         buf = BytesIO()
#         plt.savefig(buf, format='png')
#         plt.close(fig)
#         buf.seek(0)
#         buffers.append(buf)
#         return buffers

#     # Mos keluvchi filtr bankini yaratish
#     def _create_matched_filter_bank(self):
#         filters = []
#         ksize = 31
#         for theta in np.arange(0, np.pi, np.pi / 16):
#             kern = cv2.getGaborKernel((ksize, ksize), 6, theta, 12, 0.37, 0, ktype=cv2.CV_32F)
#             kern /= 1.5 * kern.sum()
#             filters.append(kern)
#         return filters

# # Logo rasmini ko'rsatish
# logo_path = './TATU-01.png'
# if os.path.exists(logo_path):
#     st.image(logo_path, width=600)

# # Streamlit ilovasi uchun sarlavha va bosh sarlavha
# st.title("Diabetic Retinopathy Detection")
# st.header("Iltimos, ko'z tasvirini yuklang!")

# # Fayl yuklagich (tasvirlar uchun)
# file = st.file_uploader('', type=['jpeg', 'png', 'jfif', 'jpg'])

# # Oldindan o'qitilgan modelni yuklash (maxsus qatlam bilan)
# with custom_object_scope({'FixedDropout': FixedDropout}):
#     model = load_model("./model.h5")

# # Labels faylidan sinf nomlarini yuklash
# with open("./labels.txt", 'r') as f:
#     class_names = [a.strip() for a in f.readlines()]

# # Tasvirni oldindan qayta ishlash funksiyasi
# def preprocess_image(image):
#     image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
#     image = np.asarray(image)
#     image = image / 255.0  # Normalizatsiya [0, 1] oralig'iga
#     image = np.expand_dims(image, axis=0)  # Batch o'lchamini qo'shish
#     return image

# if file is not None:
#     try:
#         # Tasvirni ochish va ko'rsatish
#         image = Image.open(file).convert('RGB')
#         st.image(image, use_column_width=True)

#         # Tasvirni oldindan qayta ishlash
#         processed_image = preprocess_image(image)

#         # Bashorat qilish
#         prediction = model.predict(processed_image)
#         predicted_class = np.argmax(prediction)
#         predicted_prob = np.max(prediction)

#         # Bashorat natijalarini ko'rsatish
#         st.success(f"Prediction: {class_names[predicted_class]}")
#         st.info(f"Probability: {predicted_prob * 100:.2f}%")

#         # Ehtimolliklar uchun Plotly bar chart yaratish va ko'rsatish
#         fig = px.bar(x=class_names, y=prediction[0] * 100, labels={'x': 'Class', 'y': 'Probability (%)'},
#                      title="Class Probabilities", text=prediction[0] * 100)
#         fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
#         fig.update_layout(xaxis_title="Class", yaxis_title="Probability (%)")
#         st.plotly_chart(fig)

#         # PIL tasvirini OpenCV formatiga o'zgartirish
#         open_cv_image = np.array(image)
#         open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB dan BGR ga o'zgartirish

#         # Tasvirni ImmProcessor bilan qayta ishlash
#         processor = ImmProcessor(open_cv_image)
#         buffers = processor.visualize_results()
#         for buf in buffers:
#             st.image(buf)
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")

import streamlit as st
from keras.models import load_model
from keras.layers import Dropout
from keras.utils import custom_object_scope
from PIL import Image, ImageOps
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pywt
import cv2
from io import BytesIO
import os

# FixedDropout - Keras custom dropout layer
class FixedDropout(Dropout):
    def __init__(self, rate, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)
        self.seed = seed

    def call(self, inputs, training=None):
        return super(FixedDropout, self).call(inputs, training=training)

# ImmProcessor class - for image processing
class ImmProcessor:
    def __init__(self, img, img_size=640, num_clusters=2):
        self.img_size = img_size
        self.num_clusters = num_clusters
        self.img = cv2.resize(img, (self.img_size, self.img_size))

    def preprocess_image(self):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(img_gray)
        return equ

    def apply_dwt(self, img):
        coeffs = pywt.dwt2(img, 'haar')
        equ2 = pywt.idwt2(coeffs, 'haar')
        return np.array(equ2)

    def apply_matched_filter_bank(self, img, bank):
        equ3 = self._apply_filters(img, bank)
        return np.array(equ3)

    def apply_kmeans(self, img):
        Z = img.flatten().reshape((-1, 1))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(Z, self.num_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        return res2

    def process_image(self):
        processed_img = self.preprocess_image()
        dwt_img = self.apply_dwt(processed_img)
        return dwt_img

    def process_image_with_filters(self):
        processed_img = self.preprocess_image()
        dwt_img = self.apply_dwt(processed_img)
        filter_bank = self._create_matched_filter_bank()
        filtered_img = self.apply_matched_filter_bank(dwt_img, filter_bank)
        return filtered_img

    def process_image_with_kmeans(self):
        processed_img = self.preprocess_image()
        dwt_img = self.apply_dwt(processed_img)
        kmeans_img = self.apply_kmeans(dwt_img)
        return kmeans_img

    def _apply_filters(self, img, kernels):
        images = np.array([cv2.filter2D(img, -1, k) for k in kernels])
        return np.max(images, 0)

    def visualize_results(self):
        processed_img = self.preprocess_image()
        dwt_img = self.apply_dwt(processed_img)
        filter_bank = self._create_matched_filter_bank()
        filtered_img = self.apply_matched_filter_bank(dwt_img, filter_bank)
        kmeans_img = self.apply_kmeans(dwt_img)

        images = [self.img, dwt_img, filtered_img, kmeans_img]
        titles = ['Original', 'Discrete Wavelet Transform', 'Filtered (Matched Filter Bank)', 'K-means Clustering']

        return self._get_images_as_buffers(images, titles)

    def _get_images_as_buffers(self, images, titles, scale=1.3):
        buffers = []
        fig, axs = plt.subplots(1, len(images), figsize=(20, 8))
        for i, (image, title) in enumerate(zip(images, titles)):
            axs[i].imshow(image, cmap='gray')
            axs[i].set_title(title, fontsize=10)
            axs[i].axis('off')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        buffers.append(buf)
        return buffers

    def _create_matched_filter_bank(self):
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 6, theta, 12, 0.37, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
        return filters

# Display logo image
logo_path = './TATU-01.png'
if os.path.exists(logo_path):
    st.image(logo_path, width=600)

# Streamlit app title and header
st.title("Diabetic Retinopathyni aniqlash!")
st.header("Iltimosa ko'z taavirini yuklang!")

# File uploader for image
file = st.file_uploader('', type=['jpeg', 'png', 'jfif', 'jpg'])

# Load pre-trained model with custom layer
with custom_object_scope({'FixedDropout': FixedDropout}):
    model = load_model("./diabetic_retinopathy_model.h5")

# Load class names from labels file
with open("./labels.txt", 'r') as f:
    class_names = [a.strip() for a in f.readlines()]

# Image preprocessing function
def preprocess_image(image):
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

if file is not None:
    try:
        # Tasvirni ochish va ko'rsatish
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # PIL tasvirini OpenCV formatiga o'zgartirish
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB dan BGR ga o'zgartirish

        # Tasvirni ImmProcessor bilan qayta ishlash
        processor = ImmProcessor(open_cv_image)
        processed_image_with_filters = processor.process_image_with_filters()

        # Filterlangan tasvirni modelga uzatish uchun oldindan qayta ishlash
        processed_image_with_filters = Image.fromarray(processed_image_with_filters).convert('RGB')
        processed_image_with_filters = preprocess_image(processed_image_with_filters)

        # Bashorat qilish
        prediction = model.predict(processed_image_with_filters)
        predicted_class = np.argmax(prediction)
        print(prediction)
        predicted_prob = np.max(prediction)

        # Bashorat natijalarini ko'rsatish
        st.success(f"Prediction: {class_names[predicted_class]}")
        st.info(f"Probability: {predicted_prob * 100:.2f}%")

        # Ehtimolliklar uchun Plotly bar chart yaratish va ko'rsatish
        fig = px.bar(x=class_names, y=prediction[0] * 100, labels={'x': 'Class', 'y': 'Probability (%)'},
                     title="Class Probabilities", text=prediction[0] * 100)
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(xaxis_title="Class", yaxis_title="Probability (%)")
        st.plotly_chart(fig)
        
        # Vizualizatsiya natijalarini ko'rsatish
        buffers = processor.visualize_results()
        for buf in buffers:
            st.image(buf)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
