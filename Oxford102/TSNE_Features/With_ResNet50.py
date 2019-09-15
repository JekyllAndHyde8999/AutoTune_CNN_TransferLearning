import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

# select random images from training dataset
SELECTED_IMAGES = []
SELECTED_IMAGES_LABELS = []
DATA_FOLDER = "Oxford102Flowers"
TRAIN_DIR = "training"
FULL_DIR = os.path.abspath(os.path.join(DATA_FOLDER, TRAIN_DIR))
RANDOM_CLASSES = np.random.choice(os.listdir(FULL_DIR), size=4, replace=False)

for _class in RANDOM_CLASSES:
    CLASS_DIR = os.path.join(FULL_DIR, _class)
    IMAGES = np.random.choice(os.listdir(CLASS_DIR), size=15, replace=False)
    SELECTED_IMAGES.extend([os.path.join(CLASS_DIR, img) for img in IMAGES])
    SELECTED_IMAGES_LABELS.extend([int(_class)] * 15)
# selecting random images ends here


# predicting features
IMAGE_FEATURES = []
base_model = ResNet50(weights="imagenet")
X = base_model.layers[-2].output
X = layers.Dense(102)(X)
model = models.Model(inputs=base_model.inputs, outputs=X)

for img_path in SELECTED_IMAGES:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    img_features = model.predict(x)
    IMAGE_FEATURES.append(img_features.tolist()[0])
# predicting features ends here


# plotting tsne
tsne = TSNE(n_components=2, perplexity=100, learning_rate=500, n_iter=5000, random_state=0)

# ASSUMING DATA FINALLY IS NOT SHUFFLED
FEATURES_2D = tsne.fit_transform(IMAGE_FEATURES)

plt.scatter(FEATURES_2D[:, 0], FEATURES_2D[:, 1], c=SELECTED_IMAGES_LABELS)
plt.savefig("TSNE_Features_ResNet.png")
