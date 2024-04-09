import matplotlib.pyplot as plt
import tensorflow as tf

DATA_DIR = r"./dataset"
IMG_SIZE = (240, 320)

# ローカルからDatasetを準備
dataset: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    shuffle=True,
    image_size=IMG_SIZE,
    color_mode='grayscale',
)
class_names = dataset.class_names

# data_augmentationを定義
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),
])

# データ増強のテスト表示
__image_batch, __label_batch = next(iter(dataset))
plt.figure(figsize=(8, 8))
rows, cols = (5, 5)
for i in range(rows):
    original = __image_batch[i]
    rgb = tf.image.grayscale_to_rgb(original)
    expandit = tf.expand_dims(rgb, 0)
    for n in range(cols):
        ax = plt.subplot(rows, cols, i*cols + n+1)
        augmented_image = data_augmentation(expandit)
        plt.imshow(augmented_image[0] / 255)
        plt.title(class_names[__label_batch[i]])
        plt.axis("off")
plt.show()
