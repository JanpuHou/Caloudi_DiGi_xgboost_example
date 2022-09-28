import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,cross_val_score
import xgboost as xgb
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns


x = np.load('x.npy') 
y = np.load('y.npy')
print(x.shape)
print(y.shape)


# Create a test/train split.  25% test
# Split into train/test


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Create a test/train split.  25% test
# Split into train/test

model = xgb.XGBClassifier()
model.load_model("my_xgboost_model.txt")

y_pred = model.predict(X_test)
print(y_pred.shape)


labels=['back.' ,'buffer_overflow.' ,'ftp_write.', 'guess_passwd.', 'imap.','ipsweep.' ,'land.', 'loadmodule.', 'multihop.' ,'neptune.', 'nmap.' ,'normal.','perl.' ,'phf.' ,'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.','teardrop.' ,'warezclient.', 'warezmaster.']
n=y_pred.shape[0]
print (n)

#n=1130000
ypred=np.empty(n)
ytest=np.empty(n)


for i in range(n):
    a_list=list(y_pred[i,])
    b_list=list(y_test[i,])
    #print(a_list)
    max_value = max(a_list)
    max_index_a = a_list.index(max_value)
    max_value = max(b_list)
    max_index_b = b_list.index(max_value)
    ypred[i]=max_index_a
    ytest[i]=max_index_b
    
print(ypred.shape)
print(ytest.shape)



# print(np.array(np.unique(ypred, return_counts=True)).T)
# print(np.array(np.unique(ytest, return_counts=True)).T)





y_target =  ytest
y_predicted = ypred

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted, 
                      binary=False)
print(cm)


fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.show()



# Normalizing the matrix

cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]










ax = sns.heatmap(cmn, annot=True,fmt='.2', cmap="Blues",vmax=0.8)

ax.set_title('Intrution Detection System XGBoost Model Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Attack Category')
ax.set_ylabel('Actual Attack Category ');


labels=['back.' ,'buffer_overflow.' ,'ftp_write.', 'guess_passwd.', 'imap.','ipsweep.' ,'land.', 'loadmodule.', 'multihop.' ,'neptune.', 'nmap.' ,'normal.','perl.' ,'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'teardrop.' ,'warezclient.', 'warezmaster.']


## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.xticks(rotation = 30) 
## Display the visualization of the Confusion Matrix.
plt.show()