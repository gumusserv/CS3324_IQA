from tensorflow.keras.models import load_model
from utils import *


if __name__ == '__main__':
    image = 'midjourney_normal_279.jpg'
    T2I_model_name = image[0 : image.find('_')]
    prompt = "table with everything ready for 5 o'clock at flower garden, elevation view, long-shot, baroque style"
    prompt0 = "table with everything ready for 5 o'clock at flower garden"
    prompt1 = 'elevation view'
    prompt2 = 'long-shot'
    prompt3 = 'baroque style'


    image_features = extract_image_features([image])
    image_features = image_features.reshape(image_features.shape[0], -1)

    image_5_features = extract_image_features_stair([image], 2)
    image_5_features = image_5_features.reshape(image_5_features.shape[0], -1)

    image_75_features = extract_image_features_stair([image], 4 / 3)
    image_75_features = image_75_features.reshape(image_75_features.shape[0], -1)


    augmented_image_features = extract_image_features([image], augment=True, augment_times=1)
    augmented_image_features = augmented_image_features.reshape(augmented_image_features.shape[0], -1)
    
    prompt_features = extract_text_features([prompt])
    model_name_features = extract_text_features([T2I_model_name])
    prompt0_features = extract_text_features([prompt0])
    prompt1_features = extract_text_features([prompt1])
    prompt2_features = extract_text_features([prompt2])
    prompt3_features = extract_text_features([prompt3])

    model_perception = load_model("model2_total")
    model_alignment_without_stair = load_model("model_total")
    model_alignment_with_stair = load_model("model_withStair")

    # 使用加载的模型进行预测
    perception_pred = model_perception.predict([image_features, augmented_image_features])

    alignment_pred_withoutStair = model_alignment_without_stair.predict(
        [image_features, augmented_image_features, prompt_features, model_name_features]
    )

    alignment_pred_withStair = model_alignment_with_stair.predict(
        [image_features, image_5_features, prompt0_features, model_name_features, \
         prompt1_features, prompt2_features, prompt3_features, image_75_features]
    )

    #########################
    ##  Real Score         ##
    ##  Perception: 3.637  ##
    ##  Alignment: 2.247   ##
    #########################


    print("Perception Model Prediction Score: {}.".format(perception_pred[0]))
    print("Alignment Model(no StairReward) Score: {}.".format(alignment_pred_withoutStair[0]))
    print("Alignment Model(with StairReward) Score: {}.".format(alignment_pred_withStair[0]))


    ###############################################################
    ##  Perception Model Prediction Score: [3.6725333].          ##
    ##  Alignment Model(no StairReward) Score: [2.9300935].      ##
    ##  Alignment Model(with StairReward) Score: [2.5936632].    ##     
    ###############################################################
