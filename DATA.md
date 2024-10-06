Descriptions of the data files that can be found [here](https://drive.google.com/drive/folders/1hDGo4JQDic0i8sRVqtTpIuD0PtE0TsXH?usp=drive_link).

`LazBF_sequences.csv`, `LazBF_sample.csv`, `LazDEF_sequences.csv`, `LazDEF_sample.csv`, `LazBCDEF_sequences.csv`, `LazBCDEF_sample.csv`. The ‘_sequences.csv’ files contain the sequences used for masked language modeling and the ‘_sample.csv’ files contain the held-out data sets. We reccomend that you place the `.csv` files in `LazBFDEF/Data`.

`LazBF_mlm_none.npy`: matrix of LazBF held-out sequence embeddings from Vanilla-ESM

`LazDEF_mlm_none.npy`: matrix of LazDEF held-out sequence embeddings from Vanilla-ESM

`LazBF_mlm_PA.npy`: matrix of LazBF held-out sequence embeddings from Peptide-ESM

`LazDEF_mlm_PA.npy`: matrix of LazDEF held-out sequence embeddings from Peptide-ESM

`LazBF_mlm_LazBF.npy`: matrix of LazBF held-out sequence embeddings from LazBF-ESM

`LazDEF_mlm_LazBF.npy`: matrix of LazDEF held-out sequence embeddings from LazBF-ESM

`LazBF_mlm_LazDEF.npy`: matrix of LazBF held-out sequence embeddings from LazDEF-ESM

`LazDEF_mlm_LazDEF.npy`: matrix of LazDEF held-out sequence embeddings from LazDEF-ESM

`LazBF_mlm_LazBCDEF.npy`: matrix of LazBF held-out sequence embeddings from LazBCDEF-ESM

`LazDEF_mlm_LazBCDEF.npy`: matrix of LazDEF held-out sequence embeddings from LazBCDEF-ESM

`LazBF_protbert.npy`: matrix of LazBF held-out sequence embeddings from ProtBERT

`LazDEF_protbert.npy`: matrix of LazDEF held-out sequence embeddings from ProtBERT

`pretraining_peptides_4_2.fasta`: Fasta file of the peptide sequences used for pretraining Peptide-ESM

We reccomend that you place the `.npy` files in `LazBFDEF/Embeddings`. We also reccomend placing `pretraining_peptides_4_2.fasta` in `LazBFDEF/Data`.
