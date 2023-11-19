import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os




def image_data_augmentation(image_path):
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return datagen.flow(image, batch_size=1)

def extract_image_features(image_paths, augment=False, augment_times=1):
    model = ResNet152V2(weights='imagenet', include_top=False, pooling='avg')
    features = []
    for path in image_paths:
        if augment:
            augmented_images = image_data_augmentation(path)
            for _ in range(augment_times):
                augmented_image = next(augmented_images)
                features.append(model.predict(augmented_image))
        else:
            image = load_img(path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            features.append(model.predict(image))
    return np.array(features)


# 文本特征提取
def extract_text_features(prompts):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(prompts)
    sequences = tokenizer.texts_to_sequences(prompts)
    text_features = pad_sequences(sequences, maxlen=100)
    return text_features

if __name__ == '__main__':

    # 指定要遍历的文件夹路径
    folder_path = r'C:\Users\Lenovo\Desktop\数字图像处理\AIGCIQA2023\data\AIGCIQA2023\Image\allimg\allimg'

    # 使用os.listdir()获取文件夹下的所有文件和子文件夹的名称
    file_names = os.listdir(folder_path)

    # 将文件名按数字从小到大的顺序排序
    sorted_file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

    image_paths = []
    # 遍历排序后的文件名列表并打印每个文件名
    for file_name in sorted_file_names:
        image_paths.append('AIGCIQA2023/data/AIGCIQA2023/Image/allimg/allimg/' + file_name)

    # 特征提取
    image_features = extract_image_features(image_paths)
    np.save('image_features.npy', image_features)
    
    augmented_image_features = extract_image_features(image_paths, augment= True, augment_times = 1)
    np.save('augmented_image_features.npy', augmented_image_features)




    # input_file = 'AGIQA-3k-Database\data.csv'
    # data = pd.read(input_file)
    # data.fillna("none", inplace=True)  # 用none填充缺失值
    # image_paths = data['name']  # 替换为实际的列名
    # prompts = data['prompt']

    # adj1 = data['adj1']
    # adj2 = data['adj2']
    # style = data['style']

    # for i in range(len(image_paths)):
    #     image_paths[i] = "AGIQA-3k-Database/image/" + image_paths[i]    

    # # 特征提取
    # image_features = extract_image_features(image_paths)
    # np.save('fetures_numpy/AGIQA_3k/image_features.npy', image_features)
    # text_features = extract_text_features(prompts)
    # np.save('fetures_numpy/AGIQA_3k/text_features.npy', text_features)
    # augmented_image_features = extract_image_features(image_paths, augment= True, augment_times = 1)
    # np.save('augmented_image_features.npy', augmented_image_features)

    # pure_prompts = []
    # for i in range(len(prompts)):
    #     pure_prompts.append(str(prompts[i]).split(',')[0])

    # pure_prompts_features = extract_text_features(pure_prompts)
    # np.save('fetures_numpy/AGIQA_3k/pure_prompts_features.npy', text_features)

    # adj1_features = extract_text_features(adj1)
    # np.save('fetures_numpy/AGIQA_3k/adj1_features.npy', adj1_features)
    # adj2_features = extract_text_features(adj2)
    # np.save('fetures_numpy/AGIQA_3k/adj2_features.npy', adj2_features)
    # style_features = extract_text_features(style)
    # np.save('fetures_numpy/AGIQA_3k/style_features.npy', style_features)