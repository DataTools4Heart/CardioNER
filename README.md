# Named Entity Recognition (NER) Pipeline

This repository contains a pipeline for building, training, and evaluating Named Entity Recognition (NER) models for multiple languages and entity categories.

## 📁 Project Structure

* `notebooks/build_ds_ner.ipynb`: Preprocesses raw text and annotation `.tsv` files into sentence-level IOB format files for NER training.
* `notebooks/train_ner.ipynb`: Trains a NER model using the Hugging Face `Trainer` API and pushes the model to the Hugging Face Hub.
* `scripts/evaluate_ner.py`: Evaluates a trained NER model and outputs predictions in `.tsv` format. (you can use the notebook as well)

---

## 🛠️ 1. Dataset Preparation: `build_ds_ner.ipynb`

This script converts raw annotated clinical text in `.tsv` format into IOB-labeled sentence files for each language.

### 📂 Expected Directory Structure

```
dataset/
├── English/
│   ├── txt/
│   │   ├── file1.txt
│   │   ├── file1.tsv
│   │   └── ...
│   ├── test_cardioccc_en_med.tsv
│   └── train_cardioccc_en_med.tsv
│
├── Spanish/
│   ├── txt/
│   │   ├── file1.txt
│   │   ├── file1.tsv
│   │   └── ...
│   ├── test_cardioccc_es_med.tsv
│   └── train_cardioccc_es_med.tsv
```

### 📄 TSV File Format

Each `.tsv` file must follow this structure:

```
filename	label	start_span	end_span	text	note
casos_clinicos_cardiologia508	DISEASE	83	95	toxic habits	
casos_clinicos_cardiologia508	DISEASE	130	143	dyslipidaemia	
```



## 🚀 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚪 License

MIT License. See `LICENSE` file for details.
