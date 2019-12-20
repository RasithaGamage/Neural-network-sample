import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

numpy.random.seed(7)
dataset = numpy.loadtxt("E:/UNIVERSITY/YEAR 4/RESEARCH\DATA/top-bat - last 20 matches - Avg.csv", delimiter=",")
# position | acc_matches | avg_runs | avg_balls | avg_4s | avg_6s | avg_strike_rate | avg_minutes
x = dataset[:, 0:8]
y = dataset[:, 0]
# print(x)
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
    model = Sequential()

    model.add(Dense(8, input_dim=8, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=1000, batch_size=10, verbose=0)
# model.fit(x, y, epochs=1000, batch_size=10)

kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, x, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
