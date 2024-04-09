import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_DIR = r"./dataset"

IMG_SIZE = (240, 320)
LEARNING_RATE = 0.00001
EPOCHS = 50
FINE_TUNE_AT = 100  # max:154
TITLE = "mymodel"

print("=>", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

# ======== ローカルからDatasetを準備 ========
# ラベル名はサブディレクトリで推測されます
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
dataset: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    shuffle=True,
    image_size=IMG_SIZE,
    color_mode='grayscale',
)
labels = dataset.class_names
cardinality = tf.data.experimental.cardinality(dataset).numpy()
# datasetの情報を見ておく
print("=>", "dataset.element_spec:", dataset.element_spec)
print("=>", "dataset.class_names(labels):", labels)
print("=>", "cardinality:", cardinality)
# => dataset.element_spec: (TensorSpec(shape=(None, 240, 320, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))
# => dataset.class_names(labels): ['bad', 'good', 'none', 'ok', 'paper', 'rock', 'scissors']
# => cardinality: 46

# train:validation:test を 7:3:1 くらいに分ける
# NOTE TFのバージョンが新しければ、split_datasetでスマートにできる
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/split_dataset
vali_cardinality = 3*cardinality // 11
test_cardinality = 1*cardinality // 12
train_cardinality = cardinality - vali_cardinality - test_cardinality
vali_dataset = dataset.take(vali_cardinality)
__dataset = dataset.skip(vali_cardinality)
test_dataset = __dataset.take(test_cardinality)
__dataset = __dataset.skip(test_cardinality)
train_dataset = __dataset

# Batchの数 cardinality
cardinalities = {
    "training": tf.data.experimental.cardinality(train_dataset).numpy(),
    "validation": tf.data.experimental.cardinality(vali_dataset).numpy(),
    "test": tf.data.experimental.cardinality(test_dataset).numpy(),
}
print("=>", "cardinalities:", cardinalities)

# バッファ付きプリフェッチを使用して、I/O のブロッキングなしでディスクから画像を読み込み
# https://www.tensorflow.org/guide/data_performance?hl=ja
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
vali_dataset = vali_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# ======== Model作成 ========
# ベースモデルとして事前トレーニング済みモデル`MobileNetV2`を使う
# 最上部の全結合層を抜いて取得
# https://keras.io/api/applications/
mobileNetV2 = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet')
# 指定の深さから上部を FineTuning するとして、それ以下をフリーズする
mobileNetV2.trainable = True
for layer in mobileNetV2.layers[:FINE_TUNE_AT]:
    layer.trainable = False
# mobileNetV2.summary()

# NOTE 一つのバッチをつまんで調べてみる...
__image_batch, __label_batch = next(iter(train_dataset))
__image_batch = tf.image.grayscale_to_rgb(__image_batch)
__feature = mobileNetV2(__image_batch)
print("=>", f"{__image_batch.shape} -> MobileNetV2 -> {__feature.shape}")
# => (32, 240, 320, 3) -> MobileNetV2 -> (32, 8, 10, 1280)


# Keras Functional API でビルドしていく
# https://www.tensorflow.org/guide/keras/functional?hl=ja
# 入力のTensor GrayScale
# `name`はつけておくと後でいいことがある
inputs = tf.keras.Input(name="input", shape=(IMG_SIZE+(1,)))
print("=>", "inputs", inputs.shape)
# => inputs (None, 240, 320, 1)

# ピクセル値正規化 [0, 255]->[-1, 1]
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
print("=>", "ピクセル値正規化", x.shape)
# => ピクセル値正規化 (None, 240, 320, 1)

# データ増強層 data_augmentation (学習時のみ有効になり、推論実行時にはなにもしないレイヤになる)
# https://www.tensorflow.org/tutorials/images/data_augmentation?hl=ja
x = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),
])(x)
print("=>", "data_augmentation", x.shape)
# => データ増強 (None, 240, 320, 1)

# grayscale_to_rgb
x = tf.image.grayscale_to_rgb(x)
print("=>", "toRGB", x.shape)
# => RGB化 (None, 240, 320, 3)

# CNN層 取得した事前学習済みモデル
x = mobileNetV2(x)
print("=>", "mobileNetV2", x.shape)
# => mobileNetV2 (None, 8, 10, 1280)

# プーリング層
x = tf.keras.layers.GlobalAveragePooling2D()(x)
print("=>", "pooling", x.shape)
# => pooling (None, 1280)

# ドロップアウト
x = tf.keras.layers.Dropout(0.03)(x)
print("=>", "Dropout", x.shape)
# => Dropout (None, 1280)

# output 全結合層 ラベルの数でDenseする
# `name`はつけておくと後でいいことがある
outputs = tf.keras.layers.Dense(len(labels), "softmax", name="output")(x)
print("=>", "outputs", outputs.shape)
# => outputs (None, 7)


# model生成, コンパイル
model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
)

model.summary()
# input, output のShapeを見ると
# 240*320のGrayScaleを入力として、6要素(クラス毎の確立)を出力することが分かる
# > input (InputLayer) [(None, 240, 320, 1)]
# > output (Dense) (None, 7)

# NOTE 一つのバッチをつまんで調べてみる...
__image_batch, __label_batch = next(iter(train_dataset))
__feature = model(__image_batch)
print("=>", f"{__image_batch.shape} -> model -> {__feature.shape}")
# => (32, 240, 320, 1) -> model -> (32, 7)

# ======== FineTuning ========
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=vali_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3
        ),
    ]
)

# fitの結果をグラフ表示
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
FITTING_IMG = "history_plot.png"
plt.savefig(FITTING_IMG)
print("=>", "plot:", FITTING_IMG)
# plt.show()

# ======== FineTuning後のmodelを試す ========
# evaluate
loss, accuracy = model.evaluate(test_dataset)
print("=>", "model.evaluate(test_dataset)")
print("=>", f"* loss: {loss:.6f}")
print("=>", f"* accuracy: {accuracy:.6f}")

# 手作業で確かめてみる
for n in range(5):
    __image_batch, __label_batch = next(iter(test_dataset))
    plt.figure(figsize=(10, 8)).suptitle("fine tuned model")
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        img = __image_batch[i]
        idx = __label_batch[i]
        plt.imshow(img)
        input_img = np.expand_dims(img, axis=0)
        ret = model(input_img).numpy()[0]
        max_val = np.amax(ret)
        max_idx = np.argmax(ret)
        plt.title(
            f"{labels[idx]} {labels[max_idx]} {max_val:.2f}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"tryModel_{n}.png")
    # plt.show()

# ======== Export for TensorFlow (saved_model) ========
print("=>", f"savedmodel: {TITLE}")
model.save(filepath=TITLE, save_format="tf")
