# Preprocesses the LazBF/DEF data as described in the Data Preprocessing section
# Creates the LazBF/DEF MLM and held-out  data sets

#Imports
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

# Load all LazBF/DEF sequences as .npy files
LazBF_neg = np.load('./55_r6_anti_P.npy')
LazBF_pos = np.load('./55_r6_sele_P.npy')
LazDEF_neg = np.load('./66_r5_anti_P.npy')
LazDEF_pos = np.load('./66_r5_sele_P.npy')

# Convert all LazBF sequences to strings
LazBF_sequences_pos = []
LazBF_sequences_neg = []

for i in tqdm(range(LazBF_pos.shape[0])):
  peptide = ''.join([str(aa)[2] for aa in LazBF_pos[i]])
  LazBF_sequences_pos.append(peptide)

for i in tqdm(range(LazBF_neg.shape[0])):
  peptide = ''.join([str(aa)[2] for aa in LazBF_neg[i]])
  LazBF_sequences_neg.append(peptide)

# Remove duplicates
LazBF_sequences_pos = list(set(LazBF_sequences_pos))
LazBF_sequences_neg = list(set(LazBF_sequences_neg))

# Remove sequences found in both the selection and antiselection data
common_seqs = set(LazBF_sequences_pos) & set(LazBF_sequences_neg)
LazBF_sequences_pos = [seq for seq in LazBF_sequences_pos if seq not in common_seqs]
LazBF_sequences_neg = [seq for seq in LazBF_sequences_neg if seq not in common_seqs]

# Create the list of labels
LazBF_pos_labels = [1] * len(LazBF_sequences_pos)
LazBF_neg_labels = [0] * len(LazBF_sequences_neg)

# Balanced sample of 1300000 sequences
LazBF_pos, _, LazBF_pos_labels, _ = train_test_split(LazBF_sequences_pos, LazBF_pos_labels, train_size=int(1300000/2), random_state=42)
LazBF_neg, _, LazBF_neg_labels, _ = train_test_split(LazBF_sequences_neg, LazBF_neg_labels, train_size=int(1300000/2), random_state=42)
LazBF_sequences = LazBF_pos + LazBF_neg
LazBF_labels = LazBF_pos_labels + LazBF_neg_labels
print(len(LazBF_sequences) == len(set(LazBF_sequences)))

# Convert all LazDEF sequences to strings
LazDEF_sequences_pos = []
LazDEF_sequences_neg = []

for i in tqdm(range(LazDEF_pos.shape[0])):
  peptide = ''.join([str(aa)[2] for aa in LazDEF_pos[i]])
  LazDEF_sequences_pos.append(peptide)

for i in tqdm(range(LazDEF_neg.shape[0])):
  peptide = ''.join([str(aa)[2] for aa in LazDEF_neg[i]])
  LazDEF_sequences_neg.append(peptide)

# Remove duplicates
LazDEF_sequences_pos = list(set(LazDEF_sequences_pos))
LazDEF_sequences_neg = list(set(LazDEF_sequences_neg))

# Remove sequences found in both the selection and antiselection data
common_seqs = set(LazDEF_sequences_pos) & set(LazDEF_sequences_neg)
LazDEF_sequences_pos = [seq for seq in LazDEF_sequences_pos if seq not in common_seqs]
LazDEF_sequences_neg = [seq for seq in LazDEF_sequences_neg if seq not in common_seqs]

# Create the list of labels
LazDEF_pos_labels = [1] * len(LazDEF_sequences_pos)
LazDEF_neg_labels = [0] * len(LazDEF_sequences_neg)

# Balanced sample of 1300000 sequences
LazDEF_pos, _, LazDEF_pos_labels, _ = train_test_split(LazDEF_sequences_pos, LazDEF_pos_labels, train_size=int(1300000/2), random_state=42)
LazDEF_neg, _, LazDEF_neg_labels, _ = train_test_split(LazDEF_sequences_neg, LazDEF_neg_labels, train_size=int(1300000/2), random_state=42)
LazDEF_sequences = LazDEF_pos + LazDEF_neg
LazDEF_labels = LazDEF_pos_labels + LazDEF_neg_labels
print(len(LazDEF_sequences) == len(set(LazDEF_sequences)))

# Create MLM and held-out sets
LazBF_sequences, LazBF_sample, LazBF_labels, LazBF_sample_labels = train_test_split(LazBF_sequences, LazBF_labels, train_size=len(LazBF_sequences)-50000, test_size=50000, stratify=LazBF_labels, random_state=42)
LazDEF_sequences, LazDEF_sample, LazDEF_labels, LazDEF_sample_labels = train_test_split(LazDEF_sequences, LazDEF_labels, train_size=len(LazDEF_sequences)-50000, test_size=50000, stratify=LazDEF_labels, random_state=42)

# Save each data set to csv
df = pd.DataFrame({'sequences': LazBF_sequences, 'labels': LazBF_labels})
df.to_csv('./drive/MyDrive/LazBF_sequences.csv', index=False)

df = pd.DataFrame({'sequences': LazBF_sample, 'labels': LazBF_sample_labels})
df.to_csv('./drive/MyDrive/LazBF_sample.csv', index=False)

df = pd.DataFrame({'sequences': LazDEF_sequences, 'labels': LazDEF_labels})
df.to_csv('./drive/MyDrive/LazDEF_sequences.csv', index=False)

df = pd.DataFrame({'sequences': LazDEF_sample, 'labels': LazDEF_sample_labels})
df.to_csv('./drive/MyDrive/LazDEF_sample.csv', index=False)