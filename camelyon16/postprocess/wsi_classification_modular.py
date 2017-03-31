import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from camelyon16 import utils as utils


def plot_roc(gt_y, prob_predicted_y, subset):
    predictions = prob_predicted_y[:, 1]
    fpr, tpr, _ = roc_curve(gt_y, predictions)
    roc_auc = auc(fpr, tpr)

    plt.title('ROC %s' % subset)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def validate(x, gt_y, clf):
    predicted_y = clf.predict(x)
    prob_predicted_y = clf.predict_proba(x)
    print(pd.crosstab(gt_y, predicted_y, rownames=['Actual'], colnames=['Predicted']))

    return predicted_y, prob_predicted_y


def train(x, y):
    rf_clf = RandomForestClassifier(n_estimators=50, n_jobs=2)
    rf_clf.fit(x, y)
    return rf_clf


def load_data():
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

    return df_train[feature_column_names], df_train[label_column_name], df_validation[feature_column_names], \
           df_validation[label_column_name]


if __name__ == '__main__':
    train_x, train_y, validation_x, validation_y = load_data()
    model = train(train_x, train_y)
    predict_y, prob_predict_y = validate(validation_x, validation_y, model)
    plot_roc(validation_y, prob_predict_y, 'Validation')
    predict_y, prob_predict_y = validate(train_x, train_y, model)
    plot_roc(train_y, prob_predict_y, 'Train')
