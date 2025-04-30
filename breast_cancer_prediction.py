import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
pd.set_option('display.max_rows',20)
pd.set_option('display.float_format',lambda x: '%.3f' % x)

df=pd.read_csv(r"C:\Users\Mustafa\PycharmProject2\pythonProject2\Datasets\breast_canser_dataset.csv")
df.head()
#print(df)
dataframe=df
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B':0})
dataframe["Unnamed: 32"]=1
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(dataframe)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "diagnosis")

num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    """if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()"""

for col in num_cols:
    num_summary(df, col, plot=True)


imputer = KNNImputer(n_neighbors=5, weights="uniform")
numerical_cols = dataframe.select_dtypes(include=["float64", "int64"]).columns
imputed_data = imputer.fit_transform(dataframe[numerical_cols])
dataframe[numerical_cols] = pd.DataFrame(imputed_data, columns=numerical_cols)
print(dataframe.head(50))

X = dataframe.drop("diagnosis", axis=1)
y = dataframe["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

y_train = y_train.round().astype(int)
y_test = y_test.round().astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)


print("KNN Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# Visualization of Evaluation Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
values = [accuracy, precision, recall, f1, roc_auc]

plt.figure(figsize=(8, 5))
sns.barplot(x=metrics, y=values, palette="viridis")
plt.title("KNN Model Evaluation Metrics", fontsize=14)
plt.ylabel("Scores", fontsize=12)
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
plt.show()







