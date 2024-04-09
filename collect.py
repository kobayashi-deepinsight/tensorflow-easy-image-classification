"""WEBカメラ映像からデータセットを集める

`./dataset/`配下にラベル名のディレクトリを作り、
そこへ画像を集めます

"""

import os
import cv2
import datetime

CAMERA_IDX = 0
# 保存する画像サイズ (height, width)
DATA_IMG_SIZE = (240, 320)
# DATA_IMG_SIZEを切り出す左上の座標 (top, left)
P = (40, 280)
# DATA_IMG_SIZE, P から座標を計算
top, left = P
bottom, right = tuple(x+y for (x, y) in zip(P, DATA_IMG_SIZE))

# ラベル一覧
LABELS = [
    "ok",
    "good",
    "bad",
    "scissors",
    "rock",
    "paper",
    "none",
]
# ラベルに対応する保存先の辞書 {ラベル名, 保存先ディレクトリ}
DIRS = {label: f"./dataset/{label}" for label in LABELS}
# 画像保存先ディレクトリを作る
for dir in DIRS.values():
    os.makedirs(dir, exist_ok=True)


def myPutText(frame, str, height_handle):
    """左上の方に次々描いていく"""
    p = (20, height_handle[0])
    height_handle[0] += 26
    cv2.putText(
        frame, str, p, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)


# カメラループ
cap = cv2.VideoCapture(CAMERA_IDX)
while True:
    # キャプチャ
    _, frame = cap.read()
    # 鏡映しにする
    frame = cv2.flip(frame, 1)

    # 保存する部分を切り出し
    # あとで描いたものが入らないようにコピーを控えておく
    data = frame[top:bottom, left:right].copy()
    # dataをGrayScaleにする
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    cv2.imshow('data', data)
    # 切り出した部分にrectを描く
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    # 操作方法を描く
    __height_handle = [30]
    myPutText(frame, "Esc: exit", __height_handle)
    myPutText(frame, "o: ok", __height_handle)
    myPutText(frame, "g: good", __height_handle)
    myPutText(frame, "b: bad", __height_handle)
    myPutText(frame, "s: scissors", __height_handle)
    myPutText(frame, "r: rock", __height_handle)
    myPutText(frame, "p: paper", __height_handle)
    myPutText(frame, "n: none", __height_handle)

    cv2.imshow('camera', frame)

    # キーボード入力
    key = cv2.waitKey(2)
    if key == 27:  # ESCで終了
        cv2.destroyAllWindows()
        break
    if not key:
        continue

    # ラベル名ディレクトリで保存
    # 保存ファイル名が被らないようにdatetimeのmsまで使う
    now = datetime.datetime.now().strftime('%m%d%H%M%S%f')
    # NOTE キーを押しっぱなしにすると画像を保存しすぎてしまうので、適当に保存タイミングを減らす
    if 0 != (int(now) % 3):
        continue
    elif key == ord('o'):  # ok
        filepath = f"{DIRS['ok']}/ok_{now}.png"
        cv2.imwrite(filepath, data)
    elif key == ord('g'):  # good
        filepath = f"{DIRS['good']}/good_{now}.png"
        cv2.imwrite(filepath, data)
    elif key == ord('b'):  # bad
        filepath = f"{DIRS['bad']}/bad_{now}.png"
        cv2.imwrite(filepath, data)
    elif key == ord('s'):  # scissors
        filepath = f"{DIRS['scissors']}/scissors_{now}.png"
        cv2.imwrite(filepath, data)
    elif key == ord('r'):  # rock
        filepath = f"{DIRS['rock']}/rock_{now}.png"
        cv2.imwrite(filepath, data)
    elif key == ord('p'):  # paper
        filepath = f"{DIRS['paper']}/paper_{now}.png"
        cv2.imwrite(filepath, data)
    elif key == ord('n'):  # none
        filepath = f"{DIRS['none']}/none_{now}.png"
        cv2.imwrite(filepath, data)

cap.release()
cv2.destroyAllWindows()

# 集めた画像の数を表示
total = 0
for key, dir in DIRS.items():
    num = sum(
        os.path.isfile(os.path.join(dir, name)) for name in os.listdir(dir))
    print(key, num, dir)
    total += num

print(f"total: {total}")
