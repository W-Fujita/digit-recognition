import tkinter

class Mouse:
    def pressed(self, event):
        self.x = event.x
        self.y = event.y
        canvas.create_oval(self.x, self.y, event.x, event.y, outline='black', width=20)
        
    def dragged(self, event):
        canvas.create_oval(self.x, self.y, event.x, event.y, outline='black', width=20)
        self.x = event.x
        self.y = event.y


class Scribble:
    def clear(self):
        canvas.delete('all')

    def judge(self):
        #キャンバスを画像として保存
        from PIL import ImageGrab
        x1 = root.winfo_rootx() + canvas.winfo_x()
        y1 = root.winfo_rooty() + canvas.winfo_y()
        x2 = x1 + canvas.winfo_width()
        y2 = y1 + canvas.winfo_height()
        img = ImageGrab.grab(bbox=(x1,y1,x2,y2)) #(x1,y1)から(x2,y2)の範囲をキャプチャする
        img.save("scribble.jpg")
        #保存した画像を予測
        img = predict.image_processing('scribble.jpg')
        predicted_num = predict.k_nearest_neighbors(img)
        k_nearest_neighbors_label['text'] = 'k_nearest_neighbors:{}'.format(str(predicted_num))
        predicted_num = predict.Logistic_regression(img)
        Logistic_regression_label['text'] = 'Logistic_regression:{}'.format(str(predicted_num))        
        predicted_num = predict.SVM(img)
        SVM_label['text'] = 'SVM:{}'.format(str(predicted_num))


class Predict: 
    #画像処理をするメソッド
    def image_processing(self, filename):
        import cv2
        img = cv2.imread(filename) #手書きデータの読み込み
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケールに変換
        img = cv2.resize(img, (8,8), interpolation=cv2.INTER_AREA) #平均画素法で縮小
        img = 15 - img // 16 #色を反転
        img = img.reshape(-1,64) #2次元から1次元に変換
        return img
        
    def k_nearest_neighbors(self, img):
        from sklearn.externals import joblib
        clf = joblib.load('learning_data/k_nearest_neighbors_digit.pkl') #digit_learningで作成した学習データの読み込み
        res = clf.predict(img) #データ予測
        return res[0]

    def Logistic_regression(self, img):
        from sklearn.externals import joblib
        clf = joblib.load('learning_data/Logistic_regression_digit.pkl')
        res = clf.predict(img)
        return res[0]
        
    def SVM(self, img):
        from sklearn.externals import joblib
        clf = joblib.load('learning_data/SVM_digit.pkl')
        res = clf.predict(img)
        return res[0]


mouse = Mouse()   
scribble = Scribble()
predict = Predict()
predicted_num = 0

root = tkinter.Tk()
root.title('digit recognition')
canvas = tkinter.Canvas(root, bg='white', width=320, height=320)
canvas.pack() #Canvasの配置

canvas.bind('<ButtonPress-1>', mouse.pressed) #マウスの右ボタンを押したとき
canvas.bind('<B1-Motion>', mouse.dragged) #マウスの右ボタンを押しながら動かすとき

quit_button = tkinter.Button(root, text='Quit', command=root.quit) #終了ボタン
quit_button.pack(anchor=tkinter.NE, side=tkinter.RIGHT)

clear_button = tkinter.Button(root, text='Clear', command=scribble.clear) #クリアボタン
clear_button.pack(anchor=tkinter.NE, side=tkinter.RIGHT)

judge_button = tkinter.Button(root, text='Judge', command=scribble.judge) #判定ボタン
judge_button.pack(anchor=tkinter.NE, side=tkinter.RIGHT)

k_nearest_neighbors_label = tkinter.Label(root, text='k_nearest_neighbors:{}'.format(str(predicted_num)))
k_nearest_neighbors_label.pack(anchor=tkinter.NW, side=tkinter.TOP)

Logistic_regression_label = tkinter.Label(root, text='Logistic_regression:{}'.format(str(predicted_num)))
Logistic_regression_label.pack(anchor=tkinter.NW, side=tkinter.TOP)

SVM_label = tkinter.Label(root, text='SVM:{}'.format(str(predicted_num)))
SVM_label.pack(anchor=tkinter.NW, side=tkinter.TOP)

root.mainloop()