from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

digits = datasets.load_digits() #データの読み込み
x = digits.images
y = digits.target
x = x.reshape((-1,64)) #2次元から1次元に変換
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #訓練用と試験用に分割

#k近傍法
def k_nearest_neighbors():
    from sklearn import neighbors
    clf = neighbors.KNeighborsClassifier(n_neighbors=3) #アルゴリズムの指定
    clf.fit(x_train, y_train) #データ学習

    y_pred = clf.predict(x_test)
    print('Accuracy of k_nearest_neighbors:{:.1f}%'.format((accuracy_score(y_test, y_pred))*100)) #正解率の表示
    print('Confusion matrix:\n{}'.format(metrics.confusion_matrix(y_test, y_pred))) #混同行列の表示
    
    joblib.dump(clf, 'learning_data/k_nearest_neighbors_digit.pkl') #学習データの保存
    
#ロジスティック回帰
def Logistic_regression():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=100, max_iter=5000, solver='lbfgs', multi_class='auto')
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print('\nAccuracy of Logistic_regression:{:.1f}%'.format((accuracy_score(y_test, y_pred))*100))
    print('Confusion matrix:\n{}'.format(metrics.confusion_matrix(y_test, y_pred)))

    joblib.dump(clf, 'learning_data/Logistic_regression_digit.pkl')
    
#サポートベクトルマシン
def SVM():
    from sklearn import svm   
    clf = svm.SVC(C=100, gamma=0.001)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print('\nAccuracy of SVM:{:.1f}%'.format((accuracy_score(y_test, y_pred))*100))
    print('Confusion matrix:\n{}'.format(metrics.confusion_matrix(y_test, y_pred)))

    joblib.dump(clf, 'learning_data/SVM_digit.pkl')
    

k_nearest_neighbors()
Logistic_regression()
SVM()