from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mel = np.load('DataSet/mel.npy')
label = np.load('DataSet/label.npy')

x_train, x_test, y_train, y_test = train_test_split(mel, label, test_size=0.2, random_state=42)

#clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(32, 32, 16), random_state=1, max_iter=1000)
#clf = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=0)
clf.fit(x_train, y_train)

print(f"Train Accuracy: {clf.score(x_train, y_train)}")
print(f"Test Accuracy: {clf.score(x_test, y_test)}")
