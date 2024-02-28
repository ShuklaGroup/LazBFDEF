from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import random

# Preprocesses the LazBCDEF data as described in the Data Preprocessing section
# Creates the LazBCDEF MLM and held-out data sets

LazBCDEF_neg = np.load('./content/LazBCDEF_r6_anti_merged.npy')
LazBCDEF_pos = np.load('./LazBCDEF_r6_sele_merged.npy')

LazBCDEF_sequences_pos = []
LazBCDEF_sequences_neg = []

for i in tqdm(range(LazBCDEF_pos.shape[0])):
  peptide = ''.join([str(aa)[2] for aa in LazBCDEF_pos[i]])
  LazBCDEF_sequences_pos.append(peptide)

for i in tqdm(range(LazBCDEF_neg.shape[0])):
  peptide = ''.join([str(aa)[2] for aa in LazBCDEF_neg[i]])
  LazBCDEF_sequences_neg.append(peptide)

# Remove duplicates
LazBCDEF_sequences_pos = list(set(LazBCDEF_sequences_pos))
LazBCDEF_sequences_neg = list(set(LazBCDEF_sequences_neg))

# Remove sequences found in both the selection and antiselection data
common_seqs = set(LazBCDEF_sequences_pos) & set(LazBCDEF_sequences_neg)
LazBCDEF_sequences_pos = [seq for seq in LazBCDEF_sequences_pos if seq not in common_seqs]
LazBCDEF_sequences_neg = [seq for seq in LazBCDEF_sequences_neg if seq not in common_seqs]

random.seed(1)
random.shuffle(LazBCDEF_sequences_pos)
random.shuffle(LazBCDEF_sequences_neg)

LazBCDEF_test_pos = LazBCDEF_sequences_pos[:625_000]
LazBCDEF_test_neg = LazBCDEF_sequences_neg[:625_000]

for i in range(len(LazBCDEF_test_pos)):
  LazBCDEF_test_pos[i] = LazBCDEF_test_pos[i].replace("'","")
  LazBCDEF_test_neg[i] = LazBCDEF_test_neg[i].replace("'","")

LazBCDEF_val_pos = LazBCDEF_sequences_pos[-25000:]
LazBCDEF_val_neg = LazBCDEF_sequences_neg[-25000:]
for i in range(len(LazBCDEF_val_pos)):
  LazBCDEF_val_pos[i] = LazBCDEF_val_pos[i].replace("'","")
  LazBCDEF_val_neg[i] = LazBCDEF_val_neg[i].replace("'","")

LazBF_model = AutoModelForSequenceClassification.from_pretrained('/content/drive/MyDrive/Models/LazBF_ft_alt7/checkpoint-9766').to('cuda').eval()
LazDEF_model = AutoModelForSequenceClassification.from_pretrained('/content/drive/MyDrive/Models/LazDEF_ft/checkpoint-9766').to('cuda').eval()
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

# Trainers
training_args = TrainingArguments(
    output_dir="esm_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=True,
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,
)

metric = load_metric('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

LazBF_trainer = Trainer(
    model=LazBF_model,
    args=training_args,
    #train_dataset=train,
    #eval_dataset=test,
    compute_metrics=compute_metrics
)

LazDEF_trainer = Trainer(
    model=LazDEF_model,
    args=training_args,
    #train_dataset=train,
    #eval_dataset=test,
    compute_metrics=compute_metrics
)

seqs = LazBCDEF_test_pos + LazBCDEF_test_neg
labels = [1]*625_000 + [0]*625_000
lbcdef = Dataset.from_dict(tokenizer(seqs, padding='longest'))
lbcdef = lbcdef.add_column("labels", labels)

seqs = LazBCDEF_val_pos + LazBCDEF_val_neg
labels = [1]*25_000 + [0]*25_000
lbcdef_VAL = Dataset.from_dict(tokenizer(seqs, padding='longest'))
lbcdef_VAL = lbcdef_VAL.add_column("labels", labels)

LazBF_trainer.evaluate(lbcdef)
LazDEF_trainer.evaluate(lbcdef)

# Now train vanilla-esm to classify substrates/non-substrates of the entire pathway.
LazBCDEF_ft = AutoModelForSequenceClassification.from_pretrained('facebook/esm2_t12_35M_UR50D').to('cuda').eval()
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

training_args = TrainingArguments(
    output_dir="./LazBCDEF",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
    load_best_model_at_end=True,
    # gradient_accumulation_steps=2,
)

metric = load_metric('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=LazBCDEF_ft,
    args=training_args,
    train_dataset=lbcdef,
    eval_dataset=lbcdef_VAL,
    #compute_metrics=compute_metrics
)
trainer.train()

# Evaluate on each held-out data set
trainer.evaluate(lbcdef_VAL)