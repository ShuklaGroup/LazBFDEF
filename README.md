# LazBFDEF
Code for [*Substrate Prediction for RiPP Biosynthetic Enzymes via Masked Language Modeling and Transfer Learning*](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00170b).

<!--Trained model weights can be accessed [here](https://drive.google.com/drive/folders/104klsza_oNzCbj3UOgczbsuUQ1VAy9K0?usp=drive_link).-->

All data needed to reproduce the results can be found [here](https://drive.google.com/drive/folders/1hDGo4JQDic0i8sRVqtTpIuD0PtE0TsXH?usp=drive_link). Descriptions of the data files can be found in `DATA.md`

Trained model weights can be accessed [here](https://huggingface.co/ShuklaGroupIllinois).

## Reproducing the work

All results can be reproduced by running the `.ipynb` notebooks contained in the `scripts` folder. The code can either be run on Google Colab or on a local machine. If using Google Colab, one can simply upload the notebooks one at a time and run the `pip install` cells to install the required packages as needed. If running the code locally, one can create a conda enviornment using the `lazbfenv.yaml` which lists the versions of all software libraries used in this work.

The Jupyter notebooks are numbered based on the order in which they should be run (i.e., start with `1_VanillaESMEmbeddings.ipynb`). Each notebook contains comments which guides the user through the code. Below is a brief description of each notebook and its purpose.

- `1_VanillaESMEmbeddings.ipynb`: Code used to extract LazBF/DEF sequence representations from Vanilla-ESM.
- `2_LazBFESMEmbeddings.ipynb`: Code used to train LazBF-ESM and extract LazBF/DEF sequence representations from LazBF-ESM.
- `3_LazDEFESMEmbeddings.ipynb`: Code used to train LazDEF-ESM and extract LazBF/DEF sequence representations from LazDEF-ESM.
- `4_PeptideESMEmbeddings.ipynb`: Code used to extract LazBF/DEF sequence representations from Peptide-ESM.
- `5_LazBCDEF.ipynb`: Code used to train LazBCDEF-ESM and extract LazBF/DEF sequence representations from LazBCDEF-ESM.
- `6_DownstreamModelTraining.ipynb`: Code used to train LazBF/DEF substrate classification models on embeddings from each of the 5 language models for the high, medium, and low-N conditions.
- `7_tsne.ipynb`: Code for t-SNE visualization of language model embeddings.
- `8_FineTuning.ipynb`: Code for fine-tuning 35M and 650M parameter versions of ESM-2 for LazBF/DEF/BCDEF substrate prediction.
- `9_Interpretation_650M.ipynb`: Code for Zero-shot prediction with 650M parameter models and code to reproduce figure 7.
- `10_interpretation_35M.ipynb`: Code for Zero-shot prediction with 35M parameter models and code to reproduce figures 8, S3, S4.
- `Figures.ipynb`: Code to reproduce figures 4, 6. Data were collected from `6_DownstreamModelTraining.ipynb`.
- `PeptideESMTraining.ipynb`: Pretraining code for Peptide-ESM model described in the paper.
- `OptionalDataPreprocessing.ipynb`: Code for data preprocessing. Optional since preprocessed sequences are provided. See `DATA.md`.

The code was originally run on a single A100 GPU via Google Colab.

Citation:
```
@article{Clark2024,
  title = {Substrate prediction for RiPP biosynthetic enzymes via masked language modeling and transfer learning},
  ISSN = {2635-098X},
  url = {http://dx.doi.org/10.1039/D4DD00170B},
  DOI = {10.1039/d4dd00170b},
  journal = {Digital Discovery},
  publisher = {Royal Society of Chemistry (RSC)},
  author = {Clark,  Joseph D. and Mi,  Xuenan and Mitchell,  Douglas A. and Shukla,  Diwakar},
  year = {2024}
}
```
