import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import math
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import joblib
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import SGD, RMSprop




def build_model(input_shape_image):
    input_image = Input(shape=input_shape_image)
    

    x1 = Dense(128, activation='relu')(input_image)
    x4 = Dense(64, activation='relu')(x1)

    output_quality = Dense(1, name='quality_output')(x4)
    
  
    quality_mapped_output = output_quality


    model = Model(inputs=[input_image], outputs=[quality_mapped_output])
    return model

data1 = pd.read_csv('AGIQA-3k-Database\data.csv')
data1.fillna("none", inplace=True)  # 用0填充缺失值

quality_scores = data1['mos_quality']   # 替换为实际的列名
print(len(quality_scores))
data2 = pd.read_excel('output.xlsx')
quality_scores2 = data2['mos_quality']
print(len(quality_scores2))


# 计算原数据得分范围
a, b = quality_scores.min(), quality_scores.max()

# 计算新数据得分范围
c, d = quality_scores2.min(), quality_scores2.max()

# 缩放新数据得分
quality_scores2 = ((b - a) * (quality_scores2 - c) / (d - c)) + a


quality_scores = quality_scores.to_numpy()
quality_scores2 = quality_scores2.to_numpy()
quality_scores = np.concatenate([quality_scores, quality_scores2])
print(len(quality_scores))





image_features = np.load('fetures_numpy/AGIQA_3k/image_features.npy')
augmented_image_features = np.load('fetures_numpy/AGIQA_3k/augmented_image_features.npy')
augmented_image_features = augmented_image_features.reshape(augmented_image_features.shape[0], -1)
image_features = image_features.reshape(image_features.shape[0], -1)

image_features2 = np.load('fetures_numpy/AGIQA_2023/image_features.npy')
augmented_image_features2 = np.load('fetures_numpy/AGIQA_2023/augmented_image_features.npy')
augmented_image_features2 = augmented_image_features2.reshape(augmented_image_features2.shape[0], -1)
image_features2 = image_features2.reshape(image_features2.shape[0], -1)

print(len(image_features[0]))
print(len(image_features2[0]))

image_features = np.concatenate((image_features, image_features2), axis=0)
augmented_image_features = np.concatenate((augmented_image_features, augmented_image_features2), axis=0)
# print(len(image_features))

k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)



for fold, (train_index, test_index) in enumerate(kf.split(image_features)):
    # 分割数据
    X_image_train, X_image_test = image_features[train_index], image_features[test_index]
    

    y_quality_train, y_quality_test = quality_scores[train_index], quality_scores[test_index]
    

    # 构建模型
    model  = build_model(input_shape_image=image_features.shape[1:])
    # 0.0015不错
    optimizer = Adam(learning_rate=0.001)

    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
    rmsprop_optimizer = RMSprop(learning_rate=0.0015, rho=0.9)

    model.compile(optimizer=optimizer, loss='mse')

    # 训练模型
    model.fit([X_image_train], [y_quality_train], epochs=20, batch_size=32)

    # 性能评估
    quality_pred = model.predict([X_image_test])
    srcc_quality = spearmanr(y_quality_test, quality_pred.ravel())[0]
    plcc_quality = pearsonr(y_quality_test, quality_pred.ravel())[0]
    
    # print(f"Fold {fold + 1}: Optimized parameters:", mapping_layer.get_params())
    print(f"Fold {fold + 1}: Quality - SRCC: {srcc_quality}, PLCC: {plcc_quality}")
