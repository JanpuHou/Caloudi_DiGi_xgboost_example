import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb

x = np.load('x.npy') 
y = np.load('y.npy')
print(x.shape)
print(y.shape)


# Create a test/train split.  25% test
# Split into train/test


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Create a test/train split.  25% test
# Split into train/test


model = xgb.XGBClassifier(max_depth=7, n_estimators=1000)
# model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.01)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=20)
#print(model.predict(X_test))

scores = cross_val_score(x,y,cv=5)
print('Accuracy =', np.round(scores,2))
print('Accuracy Mean: %0.2f' %(scores.mean())) 


model.save_model("my_xgboost_model.txt")