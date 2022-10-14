#imports
from sklearn import preprocessing
import sklearn.metrics as metrics
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

# Load Data
path_df = "/home/khiri/local/MetagenomicToolsClassifier/Berts/elmo_dfAbs_18022021.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)

# tarain test split
y = df["Category_Code"]
X = df.drop(["Category_Code","File_Name", "Content_Parsed"] , axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify= y)

# Saving the train/test splits
X_test.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/ELMOabs/ELMO-NB-X_test.pkl")
y_test.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/ELMOabs/ELMO-NB-y_test.pkl")

X_train.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/ELMOabs/ELMO-NB-X_train.pkl")
y_train.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/ELMOabs/ELMO-NB-y_train.pkl")

### Model & Gridsearch setup
mnbc = GaussianNB()

# Setting params for Logistic Regression gridsearch

nb_params = {'var_smoothing': [0.00000001, 0.000000001, 0.00000001]}


### nested cross validation for model optimization
d = {}
d2 = {}
best_auc = 0
NUM_TRIALS = 5
for i in range(NUM_TRIALS):
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    for train_idx, val_idx in tqdm(cv_outer.split(X_train, y_train)):

        train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
        train_target, val_target = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = mnbc
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True , random_state=7)

        gd_search = GridSearchCV(model, nb_params, scoring='balanced_accuracy', n_jobs=-1, cv=cv_inner)
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


print("### Best Combination :",best_auc, d2)

### pipeline with best parameters from nested CV

mnbc = GaussianNB(var_smoothing=d2['Best Params']['var_smoothing'])

# fit model and save it to disk
model = mnbc.fit(X_train, y_train)
with open('/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/ELMOabs/best_ELMO_mnbc.pickle', 'wb') as output:
    pickle.dump(model, output)

#predictions
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Classification report
print("##### Classification report #####")
print(metrics.classification_report(y_test, y_pred))

### calculate model scores and save it to disk
d = {
    'Model': 'ELMO + Naive Bayes',
    'Training Set Accuracy': metrics.accuracy_score(y_train, model.predict(X_train)),
    'Test Set Accuracy': metrics.accuracy_score(y_test, y_pred),
    'Area under the curve': metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
    'Precision': metrics.precision_score(y_test, y_pred, average='weighted', zero_division =0),
    'Recall': metrics.recall_score(y_test, y_pred, average='weighted', zero_division =0),
    'Fscore': metrics.f1_score(y_test, y_pred, average='weighted', zero_division =0)
                                           }

df_models_mnbc = pd.DataFrame(d, index=[0])
df_models_mnbc

with open('/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/ELMOabs/df_models_ELMO_mnbc.pickle', 'wb') as output:
        pickle.dump(df_models_mnbc, output)
