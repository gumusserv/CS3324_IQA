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


huber_loss = Huber(delta=1.5)  # delta是Huber损失的阈值，可以调整


class MappingLayer(Layer):
    def __init__(self, **kwargs):
        super(MappingLayer, self).__init__(**kwargs)
        # 初始化映射函数参数
        self.param1 = tf.Variable(1.0, trainable=True)
        self.param2 = tf.Variable(0.0, trainable=True)
        self.param3 = tf.Variable(0.0, trainable=True)
        self.param4 = tf.Variable(0.0, trainable=True)
        self.param5 = tf.Variable(0.0, trainable=True)

    def call(self, x):
        # 映射函数的实现
        return self.param1 * (0.5 - 1 / (1 + tf.exp(self.param2 * (x - self.param3)))) + self.param4 * x + self.param5

    def get_params(self):
        return [self.param1.numpy(), self.param2.numpy(), self.param3.numpy(), self.param4.numpy(), self.param5.numpy()]


def build_model(input_shape_image, input_shape_augmentation_image, input_shape_prompt, input_shape_name):
    input_image = Input(shape=input_shape_image)
    input_augmentation_image = Input(shape = input_shape_augmentation_image)
    input_prompt = Input(shape = input_shape_prompt)
    input_name = Input(shape = input_shape_name)
    merged = concatenate([input_image, input_augmentation_image, input_prompt, input_name])

    x1 = Dense(128, activation='relu')(merged)
    x2 = Dropout(0.2)(x1)
    x3 = Dense(64, activation='sigmoid')(x2)
    x4 = Dropout(0.2)(x3)

    output_quality = Dense(1, name='quality_output')(x4)
    
    # 应用自定义映射层
    mapping_layer = MappingLayer()
    quality_mapped_output = mapping_layer(output_quality)


    model = Model(inputs=[input_image, input_augmentation_image, input_prompt, input_name], outputs=[quality_mapped_output])
    return model, mapping_layer



# 文本特征提取
def extract_text_features(prompts):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(prompts)
    sequences = tokenizer.texts_to_sequences(prompts)
    text_features = pad_sequences(sequences, maxlen=100)
    return text_features

if __name__ == '__main__':
    data1 = pd.read_csv('AGIQA-3k-Database\data.csv')
    data1.fillna("none", inplace=True)  # 用0填充缺失值

    ##############################################################################
    index1 = data1['style'] == 'none'
    # index1 = data1['style'].str.contains('baroque', case=False, na=False)
    # index1 = data1['style'].str.contains('anime', case=False, na=False) | data1['style'].str.contains('realistic', case=False, na=False)
    # index1 = data1['style'].str.contains('abstract', case=False, na=False) | data1['style'].str.contains('sci-fi', case=False, na=False)
    ##############################################################################
    # # 计算每行中"none"的出现次数，并将其存储在新列"none_count"中
    # data1['none_count'] = data1[['adj1', 'adj2', 'style']].apply(lambda row: row.str.count('none')).sum(axis=1)

    # # 根据"none_count"列进行筛选
    # index1 = data1['none_count'] == 0
    # # index1 = data1['none_count'] == 1
    # # index1 = data1['none_count'] == 2
    # # index1 = data1['none_count'] == 3
    # filtered_data = data1[index1]

    # # 移除"none_count"列（如果不需要）
    # filtered_data = filtered_data.drop(columns=['none_count'])
    #######################################################################################

    # index1 = data1['name'].str.contains('glide', case=False, na=False) | data1['name'].str.contains('AttnGAN', case=False, na=False)
    # index1 = data1['name'].str.contains('DALLE', case=False, na=False) | data1['name'].str.contains('sd1.5', case=False, na=False)
    # index1 = data1['name'].str.contains('midjourney', case=False, na=False) | data1['name'].str.contains('xl2.2_normal', case=False, na=False)


    filtered_data1 = data1[index1]
    quality_scores = filtered_data1['mos_align']   # 替换为实际的列名
    # quality_scores = data1['mos_align']
    print(len(quality_scores))
    data2 = pd.read_csv('AIGCIQA2023/pic-index.csv')
    data2.fillna("none", inplace=True)  # 用0填充缺失值
    index2 = ~data2['name_one_model'].str.contains('51', case=False, na=False)
    # index2 = data2['name_one_model'].str.contains('51', case=False, na=False)

    # index2 = data2['model'].str.contains('Glide', case=False, na=False)
    # index2 = data2['model'].str.contains('DALLE', case=False, na=False) | data2['model'].str.contains('stable-diffusion', case=False, na=False)
    
    filtered_data2 = data2[index2]

    quality_scores2 = filtered_data2['mos_align']
    # quality_scores2 = data2['mos_align']
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
    augmented_image_features = augmented_image_features.reshape(augmented_image_features.shape[0], -1)[index1]
    image_features = image_features.reshape(image_features.shape[0], -1)[index1]

    image_features2 = np.load('fetures_numpy/AGIQA_2023/image_features.npy')
    augmented_image_features2 = np.load('fetures_numpy/AGIQA_2023/augmented_image_features.npy')
    augmented_image_features2 = augmented_image_features2.reshape(augmented_image_features2.shape[0], -1)[index2]
    image_features2 = image_features2.reshape(image_features2.shape[0], -1)[index2]
    # augmented_image_features2 = augmented_image_features2.reshape(augmented_image_features2.shape[0], -1)
    # image_features2 = image_features2.reshape(image_features2.shape[0], -1)

    name1 = data1['name']
    name2 = data2['model']

    for i in range(len(name1)):
        name1[i] = str(name1[i])[0: str(name1[i]).find('_')].lower()
        if name1[i] == 'sd1.5':
            name1[i] = 'stable-diffusion'
            # print(name1[i])
        elif name1[i] == 'xl2.2':
            name1[i] = 'stable-diffusionXL'
            # print(name1[i])

    for i in range(len(name2)):
        name2[i] = str(name2[i]).lower()
    name_features1 = extract_text_features(data1['name'])[index1]
    name_features2 = extract_text_features(data2['model'])[index2]
    # name_features2 = extract_text_features(data2['model'])

    prompt_features1 = extract_text_features(data1['prompt'])[index1]
    prompt_features2 = extract_text_features(data2['prompt'])[index2]
    # prompt_features2 = extract_text_features(data2['prompt'])

    # print(len(image_features[0]))
    # print(len(image_features2[0]))

    image_features = np.concatenate((image_features, image_features2), axis=0)
    augmented_image_features = np.concatenate((augmented_image_features, augmented_image_features2), axis=0)
    name_features = np.concatenate((name_features1, name_features2), axis=0)
    prompt_features = np.concatenate((prompt_features1, prompt_features2), axis=0)

    
    # name_features = name_features1
    # prompt_features = prompt_features1
    print(len(image_features))

    k = 3
    kf = KFold(n_splits=k, shuffle=True, random_state=42)



    for fold, (train_index, test_index) in enumerate(kf.split(image_features)):
        # 分割数据
        X_image_train, X_image_test = image_features[train_index], image_features[test_index]
        X_augment_train, X_augment_test = augmented_image_features[train_index], augmented_image_features[test_index]
        X_name_train, X_name_test = name_features[train_index], name_features[test_index]
        X_prompt_train, X_prompt_test = prompt_features[train_index], prompt_features[test_index]


        y_quality_train, y_quality_test = quality_scores[train_index], quality_scores[test_index]
        

        # 构建模型
        model, mapping_layer  = build_model(input_shape_image=image_features.shape[1:], \
                                             input_shape_augmentation_image = augmented_image_features.shape[1:], \
                                                input_shape_prompt = prompt_features.shape[1:], \
                                                    input_shape_name = name_features.shape[1:])
        # 0.0003不错
        optimizer = Adam(learning_rate=0.0005)

        sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
        rmsprop_optimizer = RMSprop(learning_rate=0.0005, rho=0.9)

        model.compile(optimizer=optimizer, loss='mse')

        # 训练模型
        model.fit([X_image_train, X_augment_train, X_prompt_train, X_name_train], [y_quality_train], epochs=100, batch_size=32)

        # 性能评估
        quality_pred = model.predict([X_image_test, X_augment_test, X_prompt_test, X_name_test])
        srcc_quality = spearmanr(y_quality_test, quality_pred.ravel())[0]
        plcc_quality = pearsonr(y_quality_test, quality_pred.ravel())[0]
        
        print(f"Fold {fold + 1}: Optimized parameters:", mapping_layer.get_params())
        print(f"Fold {fold + 1}: Quality - SRCC: {srcc_quality}, PLCC: {plcc_quality}")



    model.save("model_nostyle", save_format='tf')
