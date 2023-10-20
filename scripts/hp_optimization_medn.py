# Performs hyperparamter optimization of all downstream models for the medium-N condition

# Imports
import torch
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.utils import resample
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import resample
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import ndcg_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load sequences from csv
df = pd.read_csv('./LazBF_sequences.csv')
LazBF_sequences = df['sequences'].tolist()
LazBF_labels = df['labels'].tolist()

df = pd.read_csv('./LazBF_sample.csv')
LazBF_sample = df['sequences'].tolist()
LazBF_sample_labels = df['labels'].tolist()

df = pd.read_csv('./LazDEF_sequences.csv')
LazDEF_sequences = df['sequences'].tolist()
LazDEF_labels = df['labels'].tolist()

df = pd.read_csv('./LazDEF_sample.csv')
LazDEF_sample = df['sequences'].tolist()
LazDEF_sample_labels = df['labels'].tolist()

# Load Embs
lazbf_mlm_none = np.load("./LazBF_mlm_none.npy")
lazdef_mlm_none = np.load("./LazDEF_mlm_none.npy")

lazbf_mlm_lazbf = np.load("./LazBF_mlm_LazBF.npy")
lazdef_mlm_lazbf = np.load("./LazDEF_mlm_LazBF.npy")

lazbf_mlm_lazdef = np.load("./LazBF_mlm_LazDEF.npy")
lazdef_mlm_lazdef = np.load("./LazDEF_mlm_LazDEF.npy")

optX = lazbf_mlm_none
opty = LazBF_sample_labels
print("Optimization for Models trained on LazBF embs from Vanilla-ESM")

model_list = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, SVC, MLPClassifier]
model_names = ['Logistic regression', 'KNN classif', 'RF', 'AdaBoost', 'SVC', 'MLP']
parameters_list = [
    {'classifier__penalty': ("l1", "l2", "elasticnet", None), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__n_neighbors': [5, 10, 25, 50], 'classifier__weights': ('uniform', 'distance')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__criterion': ('gini', 'entropy', 'log_loss')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__learning_rate': [0.1, 1, 5, 10]},
    {'classifier__kernel':('linear', 'rbf'), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__hidden_layer_sizes': [50, 100, 500, 750, 1000], 'classifier__activation':('tanh', 'relu')},
]

for model, name, parameters in zip(model_list, model_names, parameters_list):

    random_subset_indices = random.sample(range(len(optX)), 1000)
    random_optX = [optX[i] for i in random_subset_indices]
    random_opty = [opty[i] for i in random_subset_indices]

    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('classifier', model())
    ]
    pipeline = Pipeline(steps)
    grid_search = GridSearchCV(pipeline, parameters, scoring='accuracy')
    grid_search.fit(random_optX, random_opty)
    print(f'Best paramters for {name} {grid_search.best_params_}')
    print(f'Best model for {name} {grid_search.best_estimator_}')

optX = lazdef_mlm_none
opty = LazDEF_sample_labels
print("Optimization for Models trained on LazDEF embs from Vanilla-ESM")

model_list = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, SVC, MLPClassifier]
model_names = ['Logistic regression', 'KNN classif', 'RF', 'AdaBoost', 'SVC', 'MLP']
parameters_list = [
    {'classifier__penalty': ("l1", "l2", "elasticnet", None), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__n_neighbors': [5, 10, 25, 50], 'classifier__weights': ('uniform', 'distance')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__criterion': ('gini', 'entropy', 'log_loss')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__learning_rate': [0.1, 1, 5, 10]},
    {'classifier__kernel':('linear', 'rbf'), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__hidden_layer_sizes': [50, 100, 500, 750, 1000], 'classifier__activation':('tanh', 'relu')},
]

for model, name, parameters in zip(model_list, model_names, parameters_list):

    random_subset_indices = random.sample(range(len(optX)), 1000)
    random_optX = [optX[i] for i in random_subset_indices]
    random_opty = [opty[i] for i in random_subset_indices]

    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('classifier', model())
    ]
    pipeline = Pipeline(steps)
    grid_search = GridSearchCV(pipeline, parameters, scoring='accuracy')
    grid_search.fit(random_optX, random_opty)
    print(f'Best paramters for {name} {grid_search.best_params_}')
    print(f'Best model for {name} {grid_search.best_estimator_}')

optX = lazbf_mlm_lazbf
opty = LazBF_sample_labels
print("Optimization for Models trained on LazBF embs from LazBF-ESM")

model_list = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, SVC, MLPClassifier]
model_names = ['Logistic regression', 'KNN classif', 'RF', 'AdaBoost', 'SVC', 'MLP']
parameters_list = [
    {'classifier__penalty': ("l1", "l2", "elasticnet", None), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__n_neighbors': [5, 10, 25, 50], 'classifier__weights': ('uniform', 'distance')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__criterion': ('gini', 'entropy', 'log_loss')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__learning_rate': [0.1, 1, 5, 10]},
    {'classifier__kernel':('linear', 'rbf'), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__hidden_layer_sizes': [50, 100, 500, 750, 1000], 'classifier__activation':('tanh', 'relu')},
]

for model, name, parameters in zip(model_list, model_names, parameters_list):

    random_subset_indices = random.sample(range(len(optX)), 1000)
    random_optX = [optX[i] for i in random_subset_indices]
    random_opty = [opty[i] for i in random_subset_indices]

    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('classifier', model())
    ]
    pipeline = Pipeline(steps)
    grid_search = GridSearchCV(pipeline, parameters, scoring='accuracy')
    grid_search.fit(random_optX, random_opty)
    print(f'Best paramters for {name} {grid_search.best_params_}')
    print(f'Best model for {name} {grid_search.best_estimator_}')

optX = lazdef_mlm_lazbf
opty = LazDEF_sample_labels
print("Optimization for Models trained on LazDEF embs from LazBF-ESM")

model_list = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, SVC, MLPClassifier]
model_names = ['Logistic regression', 'KNN classif', 'RF', 'AdaBoost', 'SVC', 'MLP']
parameters_list = [
    {'classifier__penalty': ("l1", "l2", "elasticnet", None), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__n_neighbors': [5, 10, 25, 50], 'classifier__weights': ('uniform', 'distance')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__criterion': ('gini', 'entropy', 'log_loss')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__learning_rate': [0.1, 1, 5, 10]},
    {'classifier__kernel':('linear', 'rbf'), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__hidden_layer_sizes': [50, 100, 500, 750, 1000], 'classifier__activation':('tanh', 'relu')},
]

for model, name, parameters in zip(model_list, model_names, parameters_list):


    random_subset_indices = random.sample(range(len(optX)), 1000)
    random_optX = [optX[i] for i in random_subset_indices]
    random_opty = [opty[i] for i in random_subset_indices]
    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('classifier', model())
    ]
    pipeline = Pipeline(steps)
    grid_search = GridSearchCV(pipeline, parameters, scoring='accuracy')
    grid_search.fit(random_optX, random_opty)
    print(f'Best paramters for {name} {grid_search.best_params_}')
    print(f'Best model for {name} {grid_search.best_estimator_}')

optX = lazbf_mlm_lazdef
opty = LazBF_sample_labels
print("Optimization for Models trained on LazBF embs from LazDEF-ESM")

model_list = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, SVC, MLPClassifier]
model_names = ['Logistic regression', 'KNN classif', 'RF', 'AdaBoost', 'SVC', 'MLP']
parameters_list = [
    {'classifier__penalty': ("l1", "l2", "elasticnet", None), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__n_neighbors': [5, 10, 25, 50], 'classifier__weights': ('uniform', 'distance')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__criterion': ('gini', 'entropy', 'log_loss')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__learning_rate': [0.1, 1, 5, 10]},
    {'classifier__kernel':('linear', 'rbf'), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__hidden_layer_sizes': [50, 100, 500, 750, 1000], 'classifier__activation':('tanh', 'relu')},
]

for model, name, parameters in zip(model_list, model_names, parameters_list):

    random_subset_indices = random.sample(range(len(optX)), 1000)
    random_optX = [optX[i] for i in random_subset_indices]
    random_opty = [opty[i] for i in random_subset_indices]
    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('classifier', model())
    ]
    pipeline = Pipeline(steps)
    grid_search = GridSearchCV(pipeline, parameters, scoring='accuracy')
    grid_search.fit(random_optX, random_opty)
    print(f'Best paramters for {name} {grid_search.best_params_}')
    print(f'Best model for {name} {grid_search.best_estimator_}')

optX = lazdef_mlm_lazdef
opty = LazDEF_sample_labels
print("Optimization for Models trained on LazDEF embs from LAZDEF-ESM")

model_list = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, SVC, MLPClassifier]
model_names = ['Logistic regression', 'KNN classif', 'RF', 'AdaBoost', 'SVC', 'MLP']
parameters_list = [
    {'classifier__penalty': ("l1", "l2", "elasticnet", None), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__n_neighbors': [5, 10, 25, 50], 'classifier__weights': ('uniform', 'distance')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__criterion': ('gini', 'entropy', 'log_loss')},
    {'classifier__n_estimators': [50, 100, 200, 500], 'classifier__learning_rate': [0.1, 1, 5, 10]},
    {'classifier__kernel':('linear', 'rbf'), 'classifier__C':[0.1, 1, 5, 10]},
    {'classifier__hidden_layer_sizes': [50, 100, 500, 750, 1000], 'classifier__activation':('tanh', 'relu')},
]

for model, name, parameters in zip(model_list, model_names, parameters_list):
    random_subset_indices = random.sample(range(len(optX)), 1000)
    random_optX = [optX[i] for i in random_subset_indices]
    random_opty = [opty[i] for i in random_subset_indices]

    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('classifier', model())
    ]
    pipeline = Pipeline(steps)
    grid_search = GridSearchCV(pipeline, parameters, scoring='accuracy')
    grid_search.fit(random_optX, random_opty)
    print(f'Best paramters for {name} {grid_search.best_params_}')
    print(f'Best model for {name} {grid_search.best_estimator_}')