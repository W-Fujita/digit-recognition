# 数字判定プログラム

## 実行環境
Python 3.7
### 追加パッケージ
* scikit-learn==0.21.3
* joblib==0.14.0
* Pillow==6.2.1
* numpy==1.17.4
* opencv-python==4.1.2.30

## 説明
手書き数字(0~9)の判定プログラムです。  
digit_learning.pyでは学習データを作成します。
scikit-learnに付属している[データセット](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)を使用します。
k近傍法、ロジスティック回帰、サポートベクトルマシンのアルゴリズムで学習します。  
digit_recognition.pyでは数字を予測します。
キャンバスを画像として読み込み処理をすることで、数字の大きさや位置を修正して判定することができます。
最終的に64個の画素に縮小し、digit_learning.pyで作成した学習データを元に予測します。

## 使い方
最初にdigit_learning.pyを実行してください。
3つのアルゴリズムの正解率と混合行列が表示されます。
learning_dataフォルダ内にはそれぞれのpklファイルが作成されます。  
次にdigit_recognition.pyを実行してください。
ウィンドウが表示されるのでキャンバス内にマウスを押しながら数字を書いてください。
Jugdeボタンを押すと予測された数字を表示します。

## 実行結果
### digit_learning
~~~
Accuracy of k_nearest_neighbors:99.2%
Confusion matrix:
[[28  0  0  0  0  0  0  0  0  0]
 [ 0 39  0  0  0  0  0  0  0  0]
 [ 0  0 37  0  0  0  0  0  0  0]
 [ 0  0  0 46  0  0  0  0  0  0]
 [ 0  0  0  0 34  0  0  0  0  0]
 [ 0  0  0  0  0 31  0  0  0  1]
 [ 0  0  0  0  0  0 39  0  0  0]
 [ 0  0  0  0  0  0  0 45  0  0]
 [ 0  0  0  0  0  0  0  0 24  0]
 [ 0  0  0  1  0  1  0  0  0 34]]

Accuracy of Logistic_regression:96.4%
Confusion matrix:
[[27  0  0  0  1  0  0  0  0  0]
 [ 0 34  0  1  0  0  0  0  0  4]
 [ 0  1 36  0  0  0  0  0  0  0]
 [ 0  0  0 45  0  0  0  0  1  0]
 [ 0  0  0  0 34  0  0  0  0  0]
 [ 0  0  0  0  0 31  0  0  0  1]
 [ 0  1  0  0  0  0 38  0  0  0]
 [ 0  0  0  0  0  0  0 45  0  0]
 [ 0  1  0  0  0  0  0  0 23  0]
 [ 0  0  0  0  0  2  0  0  0 34]]

Accuracy of SVM:97.8%
Confusion matrix:
[[27  0  0  0  1  0  0  0  0  0]
 [ 0 39  0  0  0  0  0  0  0  0]
 [ 0  0 37  0  0  0  0  0  0  0]
 [ 0  0  0 44  0  0  0  0  2  0]
 [ 0  0  0  0 34  0  0  0  0  0]
 [ 0  0  0  0  0 31  0  0  0  1]
 [ 0  1  0  0  0  0 38  0  0  0]
 [ 0  0  0  0  0  0  0 45  0  0]
 [ 0  1  0  0  0  0  0  0 23  0]
 [ 0  0  0  0  0  2  0  0  0 34]]
~~~
### digit_recognition
![execution_result](/screenshot/7.jpg)  
[もっと見る](/screenshot)