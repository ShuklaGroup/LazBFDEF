# Trains LazBF-ESM through MLM on LazBF substrate data

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
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
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

# Define model, tokenizer, and MLM data set object
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
train = Dataset.from_dict(tokenizer(LazBF_sequences)).shuffle(seed=42)
test = Dataset.from_dict(tokenizer(LazBF_sample))

args = TrainingArguments(
    output_dir='./LazBF_ESM',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=3e-6,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Do MLM for one epoch on LazBF MLM data set
trainer.train()

# Returns mean embedding from LazBF-ESM
def get_mean_rep(sequence):
  token_ids = tokenizer(sequence, return_tensors='pt').to(device)
  with torch.no_grad():
      results = model(token_ids.input_ids, output_hidden_states=True)
  representations = results.hidden_states[12][0]
  mean_embedding = representations.mean(dim=0)
  return mean_embedding.cpu().numpy()

model.eval()

# Get LazBF embeddings from LazBF-ESM
LazBF_embs = []
for seq in tqdm(LazBF_sample):
  LazBF_embs.append(get_mean_rep(seq))
LazBF_embs = np.array(LazBF_embs)

# Get LazDEF embeddings from LazBF-ESM
LazDEF_embs = []
for seq in tqdm(LazDEF_sample):
  LazDEF_embs.append(get_mean_rep(seq))
LazDEF_embs = np.array(LazDEF_embs)

# SAVE LazBF-ESM embs
np.save('./LazBF_mlm_LazBF.npy', LazBF_embs)
np.save('./LazDEF_mlm_LazBF.npy', LazDEF_embs)