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


def build_model(input_shape_image, input_shape_image_0_5, input_shape_prompt, input_shape_name, \
                input_shape_adj1, input_shape_adj2, input_shape_style, input_shape_image_0_7_5):
    input_image = Input(shape=input_shape_image)
    input_image_0_5 = Input(shape = input_shape_image_0_5)
    input_prompt = Input(shape = input_shape_prompt)
    input_name = Input(shape = input_shape_name)
    input_adj1 = Input(shape = input_shape_adj1)
    input_adj2 = Input(shape = input_shape_adj2)
    input_style = Input(shape = input_shape_style)
    input_image_0_7_5 = Input(shape = input_shape_image_0_7_5)
    merged0 = concatenate([input_image, input_prompt, input_name])
    merged1 = concatenate([input_image_0_5, input_adj1, input_name])
    merged2 = concatenate([input_image_0_7_5, input_adj2, input_name])
    merged3 = concatenate([input_image, input_style, input_name])

    x01 = Dense(128, activation='relu')(merged0)
    x02 = Dropout(0.2)(x01)
    x03 = Dense(64, activation='sigmoid')(x02)
    x04 = Dropout(0.2)(x03)
    output0 = Dense(1, name = 'output0')(x04)

    x11 = Dense(128, activation='relu')(merged1)
    x12 = Dropout(0.2)(x11)
    x13 = Dense(64, activation='sigmoid')(x12)
    x14 = Dropout(0.2)(x13)
    output1 = Dense(1, name = 'output1')(x14)

    x21 = Dense(128, activation='relu')(merged2)
    x22 = Dropout(0.2)(x21)
    x23 = Dense(64, activation='sigmoid')(x22)
    x24 = Dropout(0.2)(x23)
    output2 = Dense(1, name = 'output2')(x24)

    x31 = Dense(128, activation='relu')(merged3)
    x32 = Dropout(0.2)(x31)
    x33 = Dense(64, activation='sigmoid')(x32)
    x34 = Dropout(0.2)(x33)
    output3 = Dense(1, name = 'output3')(x34)

    # output_quality = Dense(1, name='quality_output')(x4)

    

    output_score = output0 + 8 / 7 * (output1 / 2 + output2 / 4 + output3 / 8)
    
    # 应用自定义映射层
    mapping_layer = MappingLayer()
    quality_mapped_output = mapping_layer(output_score)


    model = Model(inputs=[input_image, input_image_0_5, input_prompt, input_name, input_adj1, input_adj2, input_style, input_image_0_7_5], outputs=[quality_mapped_output])
    return model, mapping_layer



# 文本特征提取
def extract_text_features(prompts):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(prompts)
    sequences = tokenizer.texts_to_sequences(prompts)
    text_features = pad_sequences(sequences, maxlen=100)
    return text_features

if __name__ == '__main__':
    data1 = pd.read_csv(r'C:\Users\Lenovo\Desktop\数字图像处理\AGIQA-3k-Database\data.csv')
    data1.fillna("none", inplace=True)  # 用0填充缺失值

    quality_scores = data1['mos_align']
    print(len(quality_scores))
    



    quality_scores = quality_scores.to_numpy()
    
    print(len(quality_scores))





    image_features = np.load(r'C:\Users\Lenovo\Desktop\数字图像处理\HW1\fetures_numpy\AGIQA_3k\image_features.npy')
    image_features = image_features.reshape(image_features.shape[0], -1)

    image_0_5_features = np.load(r'C:\Users\Lenovo\Desktop\数字图像处理\HW1\fetures_numpy\AGIQA_3k\image_features_0_5.npy')
    image_0_5_features = image_0_5_features.reshape(image_0_5_features.shape[0], -1)

    image_0_7_5_features = np.load(r'C:\Users\Lenovo\Desktop\数字图像处理\HW1\fetures_numpy\AGIQA_3k\image_features_0_7_5.npy')
    image_0_7_5_features = image_0_7_5_features.reshape(image_0_7_5_features.shape[0], -1)


    name1 = data1['name']
    

    for i in range(len(name1)):
        name1[i] = str(name1[i])[0: str(name1[i]).find('_')].lower()
        if name1[i] == 'sd1.5':
            name1[i] = 'stable-diffusion'
            # print(name1[i])
        elif name1[i] == 'xl2.2':
            name1[i] = 'stable-diffusionXL'
            # print(name1[i])

    
    name_features = extract_text_features(name1)
    
    prompts = data1['prompt']
    for i in range(len(prompts)):
        prompts[i] = str(prompts[i]).split(',')[0]
        # print(prompts[i])
    prompt_features = extract_text_features(prompts)

    adj1_features = extract_text_features(data1['adj1'])
    adj2_features = extract_text_features(data1['adj2'])
    style_features = extract_text_features(data1['style'])
    
    

    

    
    # name_features = name_features1
    # prompt_features = prompt_features1
    print(len(image_features))

    k = 3
    kf = KFold(n_splits=k, shuffle=True, random_state=42)



    for fold, (train_index, test_index) in enumerate(kf.split(image_features)):
        # 分割数据
        X_image_train, X_image_test = image_features[train_index], image_features[test_index]
        X_image_0_5_train, X_image_0_5_test = image_0_5_features[train_index], image_0_5_features[test_index]
        X_image_0_7_5_train, X_image_0_7_5_test = image_0_7_5_features[train_index], image_0_7_5_features[test_index]
        X_name_train, X_name_test = name_features[train_index], name_features[test_index]
        X_prompt_train, X_prompt_test = prompt_features[train_index], prompt_features[test_index]
        X_adj1_train, X_adj1_test = adj1_features[train_index], adj1_features[test_index]
        X_adj2_train, X_adj2_test = adj2_features[train_index], adj2_features[test_index]
        X_style_train, X_style_test = style_features[train_index], style_features[test_index]
        


        y_quality_train, y_quality_test = quality_scores[train_index], quality_scores[test_index]
        

        # 构建模型
        model, mapping_layer  = build_model(input_shape_image=image_features.shape[1:], \
                                             input_shape_image_0_5 = image_0_5_features.shape[1:], \
                                                input_shape_prompt = prompt_features.shape[1:], \
                                                    input_shape_name = name_features.shape[1:], \
                                                        input_shape_adj1 = adj1_features.shape[1:], \
                                                            input_shape_adj2 = adj2_features.shape[1:], \
                                                            input_shape_style = style_features.shape[1:], \
                                                                input_shape_image_0_7_5 = image_0_7_5_features.shape[1:])
        # 0.0003不错
        optimizer = Adam(learning_rate=0.0005)

        sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
        rmsprop_optimizer = RMSprop(learning_rate=0.0005, rho=0.9)

        model.compile(optimizer=optimizer, loss='mse')

        # 训练模型
        model.fit([X_image_train, X_image_0_5_train, X_prompt_train, X_name_train, \
                   X_adj1_train, X_adj2_train, X_style_train, X_image_0_7_5_train], \
                    [y_quality_train], epochs=50, batch_size=32)

        # 性能评估
        quality_pred = model.predict([X_image_test, X_image_0_5_test, X_prompt_test,\
                                       X_name_test, X_adj1_test, X_adj2_test, X_style_test, X_image_0_7_5_test])
        srcc_quality = spearmanr(y_quality_test, quality_pred.ravel())[0]
        plcc_quality = pearsonr(y_quality_test, quality_pred.ravel())[0]
        
        print(f"Fold {fold + 1}: Optimized parameters:", mapping_layer.get_params())
        print(f"Fold {fold + 1}: Quality - SRCC: {srcc_quality}, PLCC: {plcc_quality}")

    model.save("model_withStair", save_format='tf')



    