import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image

def crop_center(image_path, crop_rate):
    image = Image.open(image_path)
    width, height = image.size

    new_width = width // crop_rate
    new_height = height // crop_rate

    left = (width - new_width) // crop_rate
    top = (height - new_height) // crop_rate
    right = (width + new_width) // crop_rate
    bottom = (height + new_height) // crop_rate

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image




def extract_image_features(image_paths, crop_rate):
    count = 1
    model = ResNet152V2(weights='imagenet', include_top=False, pooling='avg')
    features = []
    for path in image_paths:
        cropped_image = crop_center(path, crop_rate)
        # image = load_img(path, target_size=(224, 224))
        image = img_to_array(cropped_image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features.append(model.predict(image))
        print(count)
        count += 1
    return np.array(features)



if __name__ == '__main__':
    input_file = 'AGIQA-3k-Database\data.csv'
    data = pd.read_csv(input_file)
    data.fillna("none", inplace=True)  # 用none填充缺失值
    image_paths = data['name']  # 替换为实际的列名
    prompts = data['prompt']

    adj1 = data['adj1']
    adj2 = data['adj2']
    style = data['style']

    for i in range(len(image_paths)):
        image_paths[i] = "AGIQA-3k-Database/image/" + image_paths[i]    

    # 特征提取
    image_features = extract_image_features(image_paths, 4/3)
    np.save('image_features_0_7_5.npy', image_features)
    