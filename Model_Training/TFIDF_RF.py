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
from sklearn.ensemble import RandomForestClassifier

# Dataframe
path_df = "/home/khiri/local/MetagenomicToolsClassifier/Methods/ML/BagOfWords/Pickles/FE-df.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)


y = df["Category_Code"]
X = df['Content_Parsed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify= y)

# Saving the train/test splits
X_test.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/06-02-21_run/RF-X_test.pkl")
y_test.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/06-02-21_run/RF-y_test.pkl")

X_train.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/06-02-21_run/RF-X_train.pkl")
y_train.to_pickle("/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/06-02-21_run/RF-y_train.pkl")

# Randomforest pipeline setup
rf_pipe = Pipeline([
 ('tvec', TfidfVectorizer()),
 ('rf', RandomForestClassifier())
])
# Fit
rf_pipe.fit(X_train, y_train)
# Setting up randomforest params
rf_params = {'tvec__max_features':[40, 140, 180, 220],
            'tvec__ngram_range': [(1, 1), (1, 2), (2, 2)],
            'tvec__min_df':[0.001, 0.01, 0.1],
            'tvec__max_df':[0.5, 0.6, 0.7, 0.8, 0.9],
            'tvec__stop_words': [None, 'english'],
            'rf__max_depth': [100],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf':[1, 2, 4],
            'rf__max_leaf_nodes': [None],
            'rf__max_features':['auto', 'sqrt'],
            'rf__bootstrap':[True, False]
}


d = {}
d2 = {}
best_auc = 0
NUM_TRIALS = 5
for i in range(NUM_TRIALS):
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    for train_idx, val_idx in tqdm(cv_outer.split(X_train, y_train)):

        train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
        train_target, val_target = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = rf_pipe
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True , random_state=i)

        gd_search = GridSearchCV(model, rf_params, scoring='balanced_accuracy', n_jobs=-1, cv=cv_inner)
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



rf_pipe = Pipeline([
 ('tvec', TfidfVectorizer(max_features = d2['Best Params']['tvec__max_features'],
            ngram_range = d2['Best Params']['tvec__ngram_range'],
            min_df = d2['Best Params']['tvec__min_df'],
            max_df = d2['Best Params']['tvec__max_df'],
            stop_words = d2['Best Params']['tvec__stop_words'])),
 ('rf', RandomForestClassifier(max_depth = d2['Best Params']['rf__max_depth'],
            min_samples_split = d2['Best Params']['rf__min_samples_split'],
            min_samples_leaf = d2['Best Params']['rf__min_samples_leaf'],
            max_leaf_nodes = d2['Best Params']['rf__max_leaf_nodes'],
            max_features = d2['Best Params']['rf__max_features'],
            bootstrap = d2['Best Params']['rf__bootstrap'] ))
])

model = rf_pipe.fit(X_train, y_train)
with open('/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/06-02-21_run/best_rfc.pickle', 'wb') as output:
    pickle.dump(model, output)

y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Classification report
print("##### Classification report #####")
print(metrics.classification_report(y_test, y_pred))

# Confusion matrix
#print("##### Confision Matrix #####", metrics.multilabel_confusion_matrix(y_test, y_pred, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

d = {
     'Model': 'Random Forest',
     'Training Set Accuracy': metrics.accuracy_score(y_train, model.predict(X_train)),
     'Test Set Accuracy': metrics.accuracy_score(y_test, y_pred),
     'Area under the curve': metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
     'Precision': metrics.precision_score(y_test, y_pred, average='weighted', zero_division =0),
     'Recall': metrics.recall_score(y_test, y_pred, average='weighted', zero_division =0),
     'Fscore': metrics.f1_score(y_test, y_pred, average='weighted', zero_division =0)
}

df_models_rfc = pd.DataFrame(d, index=[0])
df_models_rfc

with open('/home/khiri/local/MetagenomicToolsClassifier/Abstracts/ML/BagOfWords/06-02-21_run/df_models_rfc.pickle', 'wb') as output:
    pickle.dump(df_models_rfc, output)
