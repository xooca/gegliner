# src/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer

class GEGliNERDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """
    Custom collate function to handle PyG Data objects and text for tokenization.
    Returns a PyG Batch object with all necessary fields.
    """
    # PyG Batch will concatenate graph fields and keep track of batch indices
    pyg_batch = Batch.from_data_list(batch)
    return pyg_batch

def create_dataloader(data_path, batch_size, shuffle=True):
    dataset = GEGliNERDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

class DataCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch_list):
        """
        batch_list: list of PyG Data objects (already batched by collate_fn)
        Returns a PyG Batch object with tokenized text and all graph fields on the correct device.
        """
        # Extract texts for tokenization
        texts = [" ".join(data.words) for data in batch_list]
        tokenized = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )

        # Create a PyG Batch object
        pyg_batch = Batch.from_data_list(batch_list)

        # Add tokenized outputs to the batch
        pyg_batch.input_ids = tokenized['input_ids'].to(self.device)
        pyg_batch.attention_mask = tokenized['attention_mask'].to(self.device)

        # Move graph fields to device if present
        for attr in ['edge_index', 'node_to_token_idx', 'y_spans', 'y_labels', 'y_spans_ptr', 'batch']:
            if hasattr(pyg_batch, attr):
                setattr(pyg_batch, attr, getattr(pyg_batch, attr).to(self.device))

        return pyg_batch