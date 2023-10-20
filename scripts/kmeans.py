# Trains k-means clustering models on each set of embeddings

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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score
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

# Helper functions
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Auroc']

def print_metrics(y_true, y_pred):
  metrics = [accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred), roc_auc_score(y_true, y_pred)]
  for i in range(len(metrics)):
    if metrics[i] < 0.5:
      metrics[i] = 1-metrics[i]
  for met, name in zip(metrics, metric_names):
    print(f'{name}: {met}')
  return metrics

def print_avg_performance(performances):
  performances = np.array(performances)
  performances_mean = np.mean(performances, axis=0)
  performances_std = np.std(performances, axis=0)
  for met, std, name in zip(performances_mean, performances_std, metric_names):
    print(f'Avgerage {name}: {met} +- {std}')

# k-Means classification of embeddings from each model

# Vanilla-ESM embeddings
LazBF_embs = lazbf_mlm_none 
LazDEF_embs = lazdef_mlm_none

print("----------")
print("LazBF Vanilla-ESM Embeddings")
performance_list = []
for i in range(5):
  kmeans = KMeans(n_clusters=2, random_state=i+1, n_init='auto')
  kmeans.fit(LazBF_embs)
  LazBF_pred = kmeans.labels_
  centroids = kmeans.cluster_centers_
  performance_list.append(print_metrics(LazBF_sample_labels, LazBF_pred))
print_avg_performance(performance_list)

print("----------")
print("LazDEF Vanilla-ESM Embeddings")
performance_list = []
for i in range(5):
  kmeans = KMeans(n_clusters=2, random_state=i+1, n_init='auto')
  kmeans.fit(LazDEF_embs)
  LazDEF_pred = kmeans.labels_
  centroids = kmeans.cluster_centers_
  performance_list.append(print_metrics(LazDEF_sample_labels, LazDEF_pred))
print_avg_performance(performance_list)

# LazBF-ESM embeddings
LazBF_embs = lazbf_mlm_lazbf
LazDEF_embs = lazdef_mlm_lazbf

print("----------")
print("LazBF LazBF-ESM Embeddings")
performance_list = []
for i in range(5):
  kmeans = KMeans(n_clusters=2, random_state=i+1, n_init='auto')
  kmeans.fit(LazBF_embs)
  LazBF_pred = kmeans.labels_
  centroids = kmeans.cluster_centers_
  performance_list.append(print_metrics(LazBF_sample_labels, LazBF_pred))
print_avg_performance(performance_list)

print("----------")
print("LazDEF LazBF-ESM Embeddings")
performance_list = []
for i in range(5):
  kmeans = KMeans(n_clusters=2, random_state=i+1, n_init='auto')
  kmeans.fit(LazDEF_embs)
  LazDEF_pred = kmeans.labels_
  centroids = kmeans.cluster_centers_
  performance_list.append(print_metrics(LazDEF_sample_labels, LazDEF_pred))
print_avg_performance(performance_list)

# LazDEF-ESM embeddings
LazBF_embs = lazbf_mlm_lazdef
LazDEF_embs = lazdef_mlm_lazdef

print("----------")
print("LazBF LazDEF-ESM Embeddings")
performance_list = []
for i in range(5):
  kmeans = KMeans(n_clusters=2, random_state=i+1, n_init='auto')
  kmeans.fit(LazBF_embs)
  LazBF_pred = kmeans.labels_
  centroids = kmeans.cluster_centers_
  performance_list.append(print_metrics(LazBF_sample_labels, LazBF_pred))
print_avg_performance(performance_list)

print("----------")
print("LazDEF LazDEF-ESM Embeddings")
performance_list = []
for i in range(5):
  kmeans = KMeans(n_clusters=2, random_state=i+1, n_init='auto')
  kmeans.fit(LazDEF_embs)
  LazDEF_pred = kmeans.labels_
  centroids = kmeans.cluster_centers_
  performance_list.append(print_metrics(LazDEF_sample_labels, LazDEF_pred))
print_avg_performance(performance_list)