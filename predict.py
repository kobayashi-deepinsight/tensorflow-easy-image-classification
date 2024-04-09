import time
import cv2
import numpy as np
import tensorflow as tf
import statistics

CAMERA_IDX = 0
# 推論に使用する画像サイズ (height, width)
DATA_IMG_SIZE = (240, 320)
# DATA_IMG_SIZEを切り出す左上の座標 (top, left)
P = (40, 280)
# DATA_IMG_SIZE, P から座標を計算
top, left = P
bottom, right = tuple(x+y for (x, y) in zip(P, DATA_IMG_SIZE))

# SavedModelディレクトリ名
SAVED_MODEL = "mymodel"
# ラベル一覧
SAVED_MODEL_LABELS = ['bad', 'good', 'none', 'ok', 'paper', 'rock', 'scissors']
# モデル設計時にname指定したinputとoutputの名前
SAVED_MODEL_INPUT_KEYWORD = "input"
SAVED_MODEL_OUTPUT_KEYWORD = "output"
# NOTE マジックナンバーに感じるなら動的に確認することもできる
# `SAVED_MODEL_INPUT_KEYWORD`, `SAVED_MODEL_OUTPUT_KEYWORD` は、traing時のTensor定義で`name`指定したもの
# 今回はハードコーディングしているが、training時にテキストファイルに出力してもいいし、predict時には次の属性を使って動的に決定することもできる
# * infer.structured_input_signature
# * infer.structured_outputs


class MyTFModel:
    # ラベル一覧 SavedModelはラベル情報を持たず、クラス分け結果はsortした順番に保持します
    LABELS = np.sort(SAVED_MODEL_LABELS)

    def __init__(self, saved_model):
        loaded_model = tf.saved_model.load(saved_model)
        self.infer = loaded_model.signatures[
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]
        self.input_key = SAVED_MODEL_INPUT_KEYWORD
        self.output_key = SAVED_MODEL_OUTPUT_KEYWORD

    def predict(self, img: np.ndarray) -> list[tuple[str, float]]:
        """画像を MyTFModel.LABELS にクラス分類する

        Args:
            img (np.ndarray): shape (240, 320) であるGrayScale画像

        Returns:
            tuple[str, float]: (ラベル名, その確率) を要素に持つリスト
        """
        # ## input.shapeに次元と形式を合わせる
        data = img.reshape(1, *DATA_IMG_SIZE, 1)
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        # ## predict
        feed = {self.input_key: data}
        res = self.infer(**feed)
        # ## outputを解釈する
        # 簡単なlistにする
        res_list = res[self.output_key][0].numpy()
        # ラベルと合わせてtupleのリストにする [(ラベル, 確立), (ラベル, 確立), ...]
        res_pairs = zip(MyTFModel.LABELS, res_list)
        return res_pairs

    def predict_label(self, img: np.ndarray) -> str:
        """画像を MyTFModel.LABELS にクラス分類して"ラベル名"か"unkwon"を返す

        Args:
            img (np.ndarray): shape (240, 320) であるGrayScale画像

        Returns:
            str: 80%以上の確立を持つラベル名 / "unkwon"
        """
        # 推論
        res = self.predict(img)
        # 確立の降順でソートする
        res = sorted(res, key=lambda x: -x[1])
        # もっとも確立の高い結果を返す
        label, rate = res[0]
        if 0.8 < rate:
            return label
        return "unknown"


def myPutText(frame, str, height_handle, scale=1.5):
    """左上の方に次々描いていく"""
    p = (20, height_handle[0])
    height_handle[0] += 26
    cv2.putText(
        frame, str, p, cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 255), 2)


if __name__ == "__main__":
    # TensorFlowモデルから推論用のインスタンス化
    infer = MyTFModel(SAVED_MODEL)

    # カメラループ
    cap = cv2.VideoCapture(CAMERA_IDX)
    que = []
    perf_counters = []
    fps = 0
    while True:
        # キャプチャ
        _, frame = cap.read()
        # 鏡映しにする
        frame = cv2.flip(frame, 1)

        # 使用する部分を切り出し
        # あとで描いたものが入らないようにコピーを控えておく
        data = frame[top:bottom, left:right].copy()
        # dataをGrayScaleにする
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        cv2.imshow('data', data)
        # 切り出す部分にrectを描く
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        # 操作方法を描く
        __height_handle = [30]
        myPutText(frame, "Esc: exit", __height_handle)

        # fps計測して描く
        perf_counters.append(time.perf_counter())
        if 30 == len(perf_counters):
            fps = 30 / (perf_counters[29] - perf_counters[0])
            perf_counters = []
        myPutText(frame, f"fps: {fps:.2f}", __height_handle)

        # 推論実行
        res = infer.predict_label(data)
        # フレーム単位で文字が変わるのが嫌なので直近10フレームで最頻値を結果とする
        que = que[1:10]+[res]
        mode = statistics.mode(que)

        # 結果を描く
        __height_handle = [140]
        myPutText(frame, mode, __height_handle, 4.0)

        # show
        cv2.imshow('camera', frame)
        key = cv2.waitKey(2)
        if key == 27:  # ESCで終了
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
