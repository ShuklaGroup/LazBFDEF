# LazBFDEF
Code for [*Substrate Prediction for RiPP Biosynthetic Enzymes via Masked Language Modeling and Transfer Learning*](https://arxiv.org/abs/2402.15181).

<!--Trained model weights can be accessed [here](https://drive.google.com/drive/folders/104klsza_oNzCbj3UOgczbsuUQ1VAy9K0?usp=drive_link).-->

All data needed to reproduce the results can be found [here](https://drive.google.com/drive/folders/1hDGo4JQDic0i8sRVqtTpIuD0PtE0TsXH?usp=drive_link). The drive link contains the following files:

`LazBF_sequences.csv`, `LazBF_sample.csv`, `LazDEF_sequences.csv`, `LazDEF_sample.csv`, `LazBCDEF_sequences.csv`, `LazBCDEF_sample.csv`. The ‘_sequences.csv’ files contain the sequences used for masked language modeling and the ‘_sample.csv’ files contain the held-out data sets.

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

Trained model weights can be accessed [here](https://huggingface.co/jjoecclark).
