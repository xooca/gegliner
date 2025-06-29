# GEGLiNER: Graph-Enhanced Generalist Entity Linking and NER

GEGLiNER is a research-oriented framework for Named Entity Recognition (NER) and Entity Linking that combines transformer-based models with knowledge graph (KG) structure using Graph Neural Networks (GNNs). It is designed for experiments on datasets like AIDA-CoNLL and supports graph-based enhancements for entity disambiguation.

---

## Features

- **Transformer backbone** (e.g., BERT, RoBERTa) for contextual token embeddings.
- **Graph Neural Network (GNN)** layers (e.g., GATv2) for leveraging KG structure.
- **Flexible data pipeline** using PyTorch Geometric and HuggingFace Transformers.
- **Span-based NER** with support for arbitrary entity types and span lengths.
- **Configurable training, evaluation, and prediction** with YAML-based configuration.

---

## Repository Structure

```
gegliner/
├── src/
│   ├── data_loader.py      # Data loading and batching utilities
│   ├── model.py            # GEGLiNER model definition
├── scripts/
│   ├── create_dataset.py   # Script to preprocess and create PyG datasets
│   ├── train.py            # Training and validation script
│   ├── evaluate.py         # Evaluation script for test/validation sets
│   ├── predict.py          # Prediction/inference script
├── config.yaml             # Example configuration file
├── README.md               # This file
```

---

## Getting Started

### 1. Install Requirements

```bash
pip install torch torch-geometric transformers spacy tqdm pyyaml
python -m spacy download en_core_web_sm
# For entity linker (optional, if using KG edges)
pip install spacy-entity-linker
```

### 2. Prepare Data

- Place your raw AIDA-CoNLL data file as specified in `config.yaml` under `data.raw_path`.

### 3. Create Processed Dataset

```bash
python scripts/create_dataset.py
```

This will generate `train.pt`, `val.pt`, and `test.pt` files in the processed data directory.

### 4. Train the Model

Edit `config.yaml` as needed, then run:

```bash
python scripts/train.py
```

### 5. Evaluate

To evaluate your model on the test set:

```bash
python scripts/evaluate.py
```

### 6. Predict

To run inference and generate predictions:

```bash
python scripts/predict.py
```

Predictions will be saved as a JSON file in your model's save directory.

---

## Configuration

All major settings (model, data paths, training hyperparameters, entity types, etc.) are controlled via `config.yaml`.

Example:
```yaml
model:
  name: bert-base-uncased
  gnn_hidden_dim: 256
  num_gnn_layers: 2
  gnn_heads: 4
  max_span_length: 10
  save_dir: ./checkpoints

data:
  raw_path: ./data/aida_conll.txt
  processed_path: ./data/processed
  entity_types: ["PER", "ORG", "LOC", "MISC"]

training:
  batch_size: 8
  learning_rate: 3e-5
  epochs: 10
  eval_threshold: 0.5
```

---

## Notes

- The entity linker in spaCy is optional; if not available, the script will fall back to a blank model and only use co-occurrence edges.
- The PyG `Data` objects include all fields required by the model and loss functions (`words`, `edge_index`, `node_to_token_idx`, `y_spans`, `y_labels`, `y_spans_ptr`).
- For custom datasets, adapt the `parse_aida_conll` and `build_graph_for_document` functions as needed.
- The `evaluate.py` and `predict.py` scripts require a trained model checkpoint (by default, `best_model.pt` in your save directory).

---

## Citation

If you use this codebase in your research, please cite the original GEGLiNER paper (if available) or this repository.

---

**Maintainer:** [Your Name]