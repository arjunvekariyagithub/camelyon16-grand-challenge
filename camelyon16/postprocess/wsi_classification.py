import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from camelyon16 import utils as utils

df_train = pd.read_csv(utils.HEATMAP_FEATURE_CSV_TRAIN)
df_validation = pd.read_csv(utils.HEATMAP_FEATURE_CSV_VALIDATION)

# print(df_train, end='\n**************************************\n\n')
#
# print(df_validation)

n_columns = len(df_train.columns)

feature_column_names = df_train.columns[:n_columns - 1]
label_column_name = df_train.columns[n_columns - 1]
print(feature_column_names)
print(label_column_name)

train_x = df_train[feature_column_names]
train_y = df_train[label_column_name]
validation_x = df_validation[feature_column_names]
validation_y = df_validation[label_column_name]

clf = RandomForestClassifier(n_estimators=50, n_jobs=2)
clf.fit(train_x, train_y)

predict_y_validation = clf.predict(validation_x)
# print(predict_y_validation)
prob_predict_y_validation = clf.predict_proba(validation_x)
# print(prob_predict_y_validation)

predictions_validation = prob_predict_y_validation[:, 1]
fpr, tpr, _ = roc_curve(validation_y, predictions_validation)
roc_auc = auc(fpr, tpr)

plt.title('ROC Validation')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print(pd.crosstab(validation_y, predict_y_validation, rownames=['Actual'], colnames=['Predicted']))

predict_y_train = clf.predict(train_x)
# print(predict_y_train)
prob_predict_y_train = clf.predict_proba(train_x)
# print(prob_predict_y_train)

predictions_train = prob_predict_y_train[:, 1]
fpr, tpr, _ = roc_curve(train_y, predictions_train)
roc_auc = auc(fpr, tpr)

plt.title('ROC Train')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print(pd.crosstab(train_y, predict_y_train, rownames=['Actual'], colnames=['Predicted']))
