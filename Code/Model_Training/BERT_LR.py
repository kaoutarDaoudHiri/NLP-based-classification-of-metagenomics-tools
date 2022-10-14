# imports
from sklearn import preprocessing
import sklearn.metrics as metrics
from tqdm import tqdm
import pickle
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
#from lime import lime_text

# load data
path_df = "/home/khiri/local/MetagenomicToolsClassifier/Berts/AbstractsBert_S2L.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)

# tarain test split
y = df["Category_Code"]
X = df.drop(["Category_Code","Category", "Content_Parsed", "Unnamed: 0"] , axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify= y)

# Saving the train/test splits
X_test.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/BERTS2Labs/BERTS2L-LR-X_test.pkl")
y_test.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/BERTS2Labs/BERTS2L-LR-y_test.pkl")

X_train.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/BERTS2Labs/BERTS2L-LR-X_train.pkl")
y_train.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/BERTS2Labs/BERTS2L-LR-y_train.pkl")

### Model & Gridsearch setup
lrc = LogisticRegression(max_iter = 1000)

# Setting params for Logistic Regression gridsearch
lr_params = {
            'C':[float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)],
            'solver':['newton-cg', 'sag', 'saga', 'lbfgs'], 
            'class_weight':['balanced', None],
             }


### nested cross validation for model optimization
d = {}
d2 = {}
best_auc = 0
NUM_TRIALS = 5
for i in range(NUM_TRIALS):
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    for train_idx, val_idx in tqdm(cv_outer.split(X_train, y_train)):

        train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
        train_target, val_target = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = lrc
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True , random_state=i)

        gd_search = GridSearchCV(model, lr_params, scoring='balanced_accuracy', n_jobs=-1, cv=cv_inner)
        gd_search.fit(train_data, train_target)
        best_model = gd_search.best_estimator_

        classifier = best_model.fit(train_data, train_target)
        y_pred_proba = classifier.predict_proba(val_data)
        y_pred = classifier.predict(val_data)

        val_target = val_target.to_numpy()

        auc = metrics.roc_auc_score(val_target, y_pred_proba, multi_class='ovr', average = 'weighted')

        d['Best GS Acc'] = gd_search.best_score_
        d['Val Acc'] = auc
        d['Best Params'] = gd_search.best_params_
        print(d)

        if auc >= best_auc:
            d2=d.copy()
            best_auc = auc

print("### Best Combination :", best_auc, d2)

### pipeline with best parameters from nested CV
lrc = LogisticRegression(C =d2['Best Params']['C'],
            solver = d2['Best Params']['solver'],
            class_weight = d2['Best Params']['class_weight'])

# fit model and save it to disk
model = lrc.fit(X_train, y_train)
with open('/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/BERTS2Labs/best_BERTS2L_lrc.pickle', 'wb') as output:
    pickle.dump(model, output)

#predictions
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Classification report
print("##### Classification report #####")
print(metrics.classification_report(y_test, y_pred))

### calculate model scores and save it to disk
d = {
     'Model': 'BERTS2L + Logistic Regression',
     'Training Set Accuracy': metrics.accuracy_score(y_train, model.predict(X_train)),
     'Test Set Accuracy': metrics.accuracy_score(y_test, y_pred),
     'Area under the curve': metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
     'Precision': metrics.precision_score(y_test, y_pred, average='weighted', zero_division =0),
     'Recall': metrics.recall_score(y_test, y_pred, average='weighted', zero_division =0),
     'Fscore': metrics.f1_score(y_test, y_pred, average='weighted', zero_division =0)
}

df_models_lrc = pd.DataFrame(d, index=[0])
df_models_lrc

with open('/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/BERTS2Labs/df_models_BERTS2L_lrc.pickle', 'wb') as output:
    pickle.dump(df_models_lrc, output)
