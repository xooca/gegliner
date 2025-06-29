# scripts/create_dataset.py

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import spacy
import re
import os
import yaml

# Load spaCy model with entity linker
try:
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("entityLinker", last=True)
except Exception as e:
    print("spaCy entity linker not available. Please install and download the knowledge base.")
    nlp = spacy.blank("en")

def parse_aida_conll(file_path):
    """Parses the AIDA-CoNLL tsv file and groups tokens by document."""
    documents = []
    current_doc_tokens = []
    doc_id = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("-DOCSTART-"):
                if current_doc_tokens:
                    documents.append({'doc_id': doc_id, 'tokens': current_doc_tokens})
                doc_id_match = re.search(r'\((.*?)\)', line)
                doc_id = doc_id_match.group(1) if doc_id_match else "unknown"
                current_doc_tokens = []
            elif line.strip():
                parts = line.strip().split('\t')
                # Format: Token, IOB, Mention, YAGO2_URL, Wiki_ID, Wiki_URL, Freebase_ID
                if len(parts) >= 6:
                    token_data = {
                        'text': parts[0],
                        'iob': parts[1],
                        'mention': parts[2] if parts[2] != '--NME--' else None,
                        'yago_id': parts[3] if parts[3] != '--NME--' else None,
                        'wiki_id': parts[4] if parts[4] != '--NME--' else None
                    }
                    current_doc_tokens.append(token_data)
        # Add last document
        if current_doc_tokens:
            documents.append({'doc_id': doc_id, 'tokens': current_doc_tokens})
    return documents

def build_graph_for_document(doc_tokens):
    """Builds a knowledge graph for a single document."""
    text = " ".join([tok['text'] for tok in doc_tokens])
    try:
        spacy_doc = nlp(text)
    except Exception:
        spacy_doc = nlp.make_doc(text)

    # 1. Node Identification: Map unique KG entities to node indices
    kg_entities = {}  # {yago_id: {mentions, indices}}
    node_map = {}     # {yago_id: node_idx}
    node_idx_counter = 0

    for i, token in enumerate(doc_tokens):
        if token['yago_id']:
            yago_id = token['yago_id']
            if yago_id not in kg_entities:
                kg_entities[yago_id] = {'mentions': set(), 'indices': []}
                node_map[yago_id] = node_idx_counter
                node_idx_counter += 1
            kg_entities[yago_id]['mentions'].add(token['mention'])
            kg_entities[yago_id]['indices'].append(i)

    if not kg_entities:
        return None, None  # No graph to build

    # 2. Edge Formulation
    edges = set()
    yago_ids = list(kg_entities.keys())

    # 2a. Co-occurrence edges (within sentences)
    # (This can be improved, but for now, skip if no KG edges found)
    linked_spacy_ents = []
    if hasattr(spacy_doc._, "linkedEntities"):
        linked_spacy_ents = [ent for ent in spacy_doc._.linkedEntities]

    # 2b. KG-based edges using spacy-entity-linker's knowledge base
    for i in range(len(linked_spacy_ents)):
        for j in range(i + 1, len(linked_spacy_ents)):
            ent1 = linked_spacy_ents[i]
            ent2 = linked_spacy_ents[j]
            try:
                super_classes1 = {e.get_id() for e in ent1.get_super_entities()}
                super_classes2 = {e.get_id() for e in ent2.get_super_entities()}
                if super_classes1 and super_classes1.intersection(super_classes2):
                    yago1 = next((y_id for y_id, data in kg_entities.items() if ent1.get_label() in data['mentions']), None)
                    yago2 = next((y_id for y_id, data in kg_entities.items() if ent2.get_label() in data['mentions']), None)
                    if yago1 and yago2 and yago1 != yago2:
                        u, v = node_map[yago1], node_map[yago2]
                        edges.add(tuple(sorted((u, v))))
            except Exception:
                pass

    # If no KG edges found, fall back to simple co-occurrence (optional)
    if not edges and len(yago_ids) > 1:
        for i in range(len(yago_ids)):
            for j in range(i + 1, len(yago_ids)):
                u, v = node_map[yago_ids[i]], node_map[yago_ids[j]]
                edges.add(tuple(sorted((u, v))))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    # Map nodes to the first token of their first mention
    node_to_token_idx = torch.zeros(len(node_map), dtype=torch.long)
    for yago_id, node_idx in node_map.items():
        first_token_idx = min(kg_entities[yago_id]['indices'])
        node_to_token_idx[node_idx] = first_token_idx

    return edge_index, node_to_token_idx

def create_pyg_dataset(documents):
    """Creates a list of PyG Data objects from parsed documents."""
    dataset = []
    for doc in tqdm(documents, desc="Processing documents"):
        tokens = doc['tokens']
        words = [tok['text'] for tok in tokens]

        edge_index, node_to_token_idx = build_graph_for_document(tokens)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            node_to_token_idx = torch.empty((0,), dtype=torch.long)

        # Extract ground truth spans and labels
        spans = []
        labels = []
        in_entity = False
        start_idx = 0
        current_label = None

        for i, token in enumerate(tokens):
            if token['iob'] == 'B':
                if in_entity:
                    spans.append([start_idx, i - 1])
                    labels.append(current_label)
                start_idx = i
                current_label = token['yago_id'] if token['yago_id'] else "O"
                in_entity = True
            elif token['iob'] == 'I' and not in_entity:
                start_idx = i
                current_label = token['yago_id'] if token['yago_id'] else "O"
                in_entity = True
            elif token['iob'] == 'O' and in_entity:
                spans.append([start_idx, i - 1])
                labels.append(current_label)
                in_entity = False
        if in_entity:
            spans.append([start_idx, len(tokens) - 1])
            labels.append(current_label)

        # Map YAGO IDs to integer labels for training
        simple_labels = []
        label_map = {}
        label_idx = 0
        for lbl in labels:
            if lbl not in label_map:
                label_map[lbl] = label_idx
                label_idx += 1
            simple_labels.append(label_map[lbl])

        # y_spans_ptr for batching: cumulative sum of number of spans per document
        y_spans = torch.tensor(spans, dtype=torch.long) if spans else torch.empty((0, 2), dtype=torch.long)
        y_labels = torch.tensor(simple_labels, dtype=torch.long) if simple_labels else torch.empty((0,), dtype=torch.long)
        y_spans_ptr = torch.tensor([0, len(y_spans)], dtype=torch.long)  # For single doc, just [0, N]

        data = Data(
            words=words,
            edge_index=edge_index,
            node_to_token_idx=node_to_token_idx,
            y_spans=y_spans,
            y_labels=y_labels,
            y_spans_ptr=y_spans_ptr
        )
        dataset.append(data)
    return dataset

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    raw_data_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']
    os.makedirs(processed_path, exist_ok=True)

    print("Parsing AIDA-CoNLL dataset...")
    documents = parse_aida_conll(raw_data_path)

    # Split documents into train/testa/testb based on CoNLL splits
    train_docs = [d for d in documents if d['doc_id'].isdigit() and int(d['doc_id']) <= 946]
    val_docs = [d for d in documents if d['doc_id'].isdigit() and 947 <= int(d['doc_id']) <= 1162]
    test_docs = [d for d in documents if d['doc_id'].isdigit() and int(d['doc_id']) >= 1163]

    print(f"Found {len(train_docs)} train, {len(val_docs)} validation, {len(test_docs)} test documents.")

    print("Creating training dataset...")
    train_dataset = create_pyg_dataset(train_docs)
    torch.save(train_dataset, os.path.join(processed_path, 'train.pt'))

    print("Creating validation dataset...")
    val_dataset = create_pyg_dataset(val_docs)
    torch.save(val_dataset, os.path.join(processed_path, 'val.pt'))

    print("Creating test dataset...")
    test_dataset = create_pyg_dataset(test_docs)
    torch.save(test_dataset, os.path.join(processed_path, 'test.pt'))

    print("Dataset creation complete.")

if __name__ == "__main__":
    main()