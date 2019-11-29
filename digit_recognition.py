import tkinter

class Pen:
    def pressed(self, event):
        """マウスが押されたとき"""
        #キャンバスの保存
        if scribble.flag == True:
            scribble.save()
        elif scribble.flag == False:
            scribble.flag = True
        #始点を描く
        color = self.get_color()
        self.x = event.x
        self.y = event.y
        canvas.create_oval(self.x, self.y, event.x, event.y, outline=color, width=self.width.get())

    def dragged(self, event):
        """マウスが動かされたとき"""
        color = self.get_color()
        canvas.create_oval(self.x, self.y, event.x, event.y, outline=color, width=self.width.get())
        self.x = event.x
        self.y = event.y

    def set_color(self):
        """線の色を設定"""
        things = ["Pen", "Eraser"]
        self.thing = tkinter.StringVar()
        self.thing.set(things[0])
        menu = tkinter.OptionMenu(root, self.thing, *things)
        menu.pack(anchor=tkinter.S, side=tkinter.LEFT)

    def get_color(self):
        """線の色を取得"""
        if self.thing.get() == "Pen":
            return "black"
        elif self.thing.get() == "Eraser":
            return "white"

    def width(self):
        """線の太さを選択"""
        self.width = tkinter.Scale(root, from_=10, to=30, orient=tkinter.HORIZONTAL)
        self.width.set(20)
        self.width.pack(anchor=tkinter.S, side=tkinter.LEFT)


class Scribble:
    def __init__(self):
        self.images = []
        self.flag = True

    def get(self):
        """キャンバスを画像として取得"""
        from PIL import ImageGrab
        x1 = root.winfo_rootx() + canvas.winfo_x()
        y1 = root.winfo_rooty() + canvas.winfo_y()
        x2 = x1 + canvas.winfo_width()
        y2 = y1 + canvas.winfo_height()
        img = ImageGrab.grab(bbox=(x1,y1,x2,y2)) #(x1,y1)から(x2,y2)の範囲をキャプチャする
        return img

    def save(self):
        """取得した画像を保存"""
        img = self.get()
        self.images.append(img)

    def clear(self):
        """キャンバスを元の状態に戻す"""
        self.save()
        self.flag = False
        canvas.delete("all")

    def back(self):
        """キャンバスを1つ前の状態に戻す"""
        from PIL import ImageTk
        global imgtk #ガベージコレクションにより削除されることを防ぐ
        if len(self.images) > 0:
            img = self.images.pop() #末尾の要素を削除し取得
            imgtk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, image=imgtk, anchor=tkinter.NW)

    def judge(self):
        """取得した画像を予測"""
        img = self.get()
        data = predict.image_processing(img)
        predicted_num[0] = predict.k_nearest_neighbors(data)
        k_nearest_neighbors_label["text"] = "k_nearest_neighbors:{}".format(str(predicted_num[0]))
        predicted_num[1] = predict.Logistic_regression(data)
        Logistic_regression_label["text"] = "Logistic_regression:{}".format(str(predicted_num[1]))
        predicted_num[2] = predict.SVM(data)
        SVM_label["text"] = "SVM:{}".format(str(predicted_num[2]))


class Predict:
    def image_cropping(self, img):
        """余白の部分を切り出す"""
        from PIL import Image, ImageChops
        #キャンバスの縁(灰色)の部分をトリミング
        w, h = img.size
        box = (2, 2, w-2, h-2)
        img = img.crop(box)
        #背景色画像(白色)を作成
        bg = Image.new("RGB", img.size, img.getpixel((0, 0)))
        #元画像と背景色画像の差分画像を作成
        img_diff = ImageChops.difference(img, bg)
        #背景色との境界を求めて画像を切り抜く
        croprange = img_diff.convert("RGB").getbbox()
        img = img.crop(croprange)
        return img

    def image_square(self, img):
        """長方形の画像を正方形にする"""
        from PIL import Image
        import numpy as np
        w, h = img.size
        if w > h:
            diff = w - h
            diff_half = int(diff / 2)
            for i in range(diff_half):
                img = np.insert(img, h+i, 255, axis=0) #末尾の行を追加
                img = np.insert(img, 0, 255, axis=0) #先頭の行を追加
            if diff % 2 == 1:
                img = np.insert(img, 0, 255, axis=0) #先頭の行を追加
        else:
            diff = h - w
            diff_half = int(diff / 2)
            for i in range(diff_half):
                img = np.insert(img, w+i, 255, axis=1) #末尾の列を追加
                img = np.insert(img, 0, 255, axis=1) #先頭の列を追加
            if diff % 2 == 1:
                img = np.insert(img, 0, 255, axis=1) #先頭の列を追加
        #引数の画像が既に正方形のとき
        if type(img) == Image.Image:
            img = np.array(img, dtype=np.uint8) #PIL型からOpenCV型に変換
        return img

    def image_processing(self, img_src):
        """画像処理"""
        import cv2
        from sklearn.preprocessing import MinMaxScaler
        img_cropped = self.image_cropping(img_src)
        img_squared = self.image_square(img_cropped)
        img_gry = cv2.cvtColor(img_squared, cv2.COLOR_BGR2GRAY) #グレースケールに変換
        img_resized = cv2.resize(img_gry, (8,8), interpolation=cv2.INTER_AREA) #平均画素法で縮小
        img_inv = 15 - img_resized // 16 #色を反転
        scaler = MinMaxScaler()
        img_scaled = scaler.fit_transform(img_inv) #訓練用のデータを正規化
        data = img_scaled.reshape(-1,64) #2次元から1次元に変換
        return data

    def k_nearest_neighbors(self, data):
        """k近傍法"""
        import joblib
        clf = joblib.load("learning_data/k_nearest_neighbors_digit.pkl") #digit_learningで作成した学習データの読み込み
        res = clf.predict(data) #データ予測
        return res[0]

    def Logistic_regression(self, data):
        """ロジスティック回帰"""
        import joblib
        clf = joblib.load("learning_data/Logistic_regression_digit.pkl")
        res = clf.predict(data)
        return res[0]

    def SVM(self, data):
        """サポートベクトルマシン"""
        import joblib
        clf = joblib.load("learning_data/SVM_digit.pkl")
        res = clf.predict(data)
        return res[0]


pen = Pen()
scribble = Scribble()
predict = Predict()
predicted_num = [None, None, None]

root = tkinter.Tk()
root.title("digit recognition")
canvas = tkinter.Canvas(root, bg="white", width=360, height=360)
canvas.pack() #Canvasの配置

canvas.bind("<ButtonPress-1>", pen.pressed) #マウスの右ボタンを押したとき
canvas.bind("<B1-Motion>", pen.dragged) #マウスの右ボタンを押しながら動かすとき

quit_button = tkinter.Button(root, text="Quit", command=root.quit) #終了ボタン
quit_button.pack(anchor=tkinter.N, side=tkinter.RIGHT)

clear_button = tkinter.Button(root, text="Clear", command=scribble.clear) #クリアボタン
clear_button.pack(anchor=tkinter.N, side=tkinter.RIGHT)

back_button = tkinter.Button(root, text="Back", command=scribble.back) #戻るボタン
back_button.pack(anchor=tkinter.N, side=tkinter.RIGHT)

judge_button = tkinter.Button(root, text="Judge", command=scribble.judge) #判定ボタン
judge_button.pack(anchor=tkinter.N, side=tkinter.RIGHT)

k_nearest_neighbors_label = tkinter.Label(root, text="k_nearest_neighbors:{}".format(str(predicted_num[0])))
k_nearest_neighbors_label.pack(anchor=tkinter.W, side=tkinter.TOP)

Logistic_regression_label = tkinter.Label(root, text="Logistic_regression:{}".format(str(predicted_num[1])))
Logistic_regression_label.pack(anchor=tkinter.W, side=tkinter.TOP)

SVM_label = tkinter.Label(root, text="SVM:{}".format(str(predicted_num[2])))
SVM_label.pack(anchor=tkinter.W, side=tkinter.TOP)

pen.set_color()
pen.width()

root.mainloop()