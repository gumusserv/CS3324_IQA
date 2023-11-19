from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.stats import spearmanr, pearsonr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Lambda


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

def extract_image_features_stair(image_paths, crop_rate):
    model = ResNet152V2(weights='imagenet', include_top=False, pooling='avg')
    features = []
    for path in image_paths:
        cropped_image = crop_center(path, crop_rate)
        image = img_to_array(cropped_image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features.append(model.predict(image))
    return np.array(features)

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