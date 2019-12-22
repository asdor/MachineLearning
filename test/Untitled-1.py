import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/asdor45/education/ML/lab2/2_features/9.csv')
X = df.iloc[:, 0:2].to_numpy()
y = df.iloc[:, 2].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.shape)



from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=20, alpha=0.0,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1, momentum=0.0)

mlp.fit(X_train, y_train)
mlp.predict(X_test)
print(mlp.predict(X_test))
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

for coef in mlp.coefs_:
    print(coef.shape)

