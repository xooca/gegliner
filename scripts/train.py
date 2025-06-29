# scripts/train.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import yaml
import os

from src.data_loader import create_dataloader, DataCollator
from src.model import GEGliNER

def compute_loss(outputs, pyg_batch, max_span_length):
    """Computes the loss for a batch based on GLiNER's span-matching approach."""
    scores = outputs['scores']  # [batch_size, max_spans, num_types]
    span_masks = outputs['span_masks']  # [batch_size, max_spans]
    attention_mask = pyg_batch.attention_mask
    device = scores.device

    batch_size, max_spans, num_types = scores.shape
    targets = torch.zeros_like(scores)

    # Create a mapping from (batch_idx, start, end) to the flattened span_idx
    span_map = {}
    for b_idx in range(batch_size):
        valid_len = attention_mask[b_idx].sum().item()
        s_idx = 0
        for start in range(valid_len):
            for end in range(start, min(start + max_span_length, valid_len)):
                if s_idx < max_spans:
                    span_map[(b_idx, start, end)] = s_idx
                s_idx += 1

    # Use the `ptr` attribute from PyG batch to get spans for each item
    # This assumes `y_spans` and `y_labels` are correctly batched by PyG
    y_spans_ptr = pyg_batch.y_spans_ptr
    for b_idx in range(batch_size):
        gt_spans = pyg_batch.y_spans[y_spans_ptr[b_idx]:y_spans_ptr[b_idx+1]]
        gt_labels = pyg_batch.y_labels[y_spans_ptr[b_idx]:y_spans_ptr[b_idx+1]]

        for gt_span, gt_label in zip(gt_spans, gt_labels):
            span_key = (b_idx, gt_span[0].item(), gt_span[1].item())
            if span_key in span_map:
                span_idx = span_map[span_key]
                targets[b_idx, span_idx, gt_label.item()] = 1.0

    # Compute Binary Cross Entropy loss on valid spans
    active_loss_mask = span_masks.view(-1)
    active_logits = scores.view(-1, num_types)[active_loss_mask]
    active_labels = targets.view(-1, num_types)[active_loss_mask]

    if active_logits.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
    return loss

def train_epoch(model, dataloader, collator, optimizer, scheduler, device, tokenized_entity_types, max_span_length):
    model.train()
    total_loss = 0
    for batch_list in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        pyg_batch = collator(batch_list)
        
        # Prepare entity type inputs for the batch
        batch_size = pyg_batch.input_ids.size(0)
        entity_type_ids = tokenized_entity_types['input_ids'].expand(batch_size, -1, -1).to(device)
        entity_type_mask = tokenized_entity_types['attention_mask'].expand(batch_size, -1, -1).to(device)

        outputs = model(
            input_ids=pyg_batch.input_ids,
            attention_mask=pyg_batch.attention_mask,
            entity_type_ids=entity_type_ids,
            entity_type_mask=entity_type_mask,
            edge_index=pyg_batch.edge_index,
            node_to_token_idx=pyg_batch.node_to_token_idx,
            batch_map=pyg_batch.batch
        )
        
        # Calculate loss using the span-matching approach
        loss = compute_loss(outputs, pyg_batch, max_span_length)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, collator, device, tokenized_entity_types, max_span_length, threshold=0.5):
    """Evaluates the model on the validation set, computing loss, precision, recall, and F1-score."""
    model.eval()
    total_loss = 0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for batch_list in tqdm(dataloader, desc="Validating"):
            pyg_batch = collator(batch_list)
            
            # Prepare entity type inputs for the batch
            batch_size = pyg_batch.input_ids.size(0)
            entity_type_ids = tokenized_entity_types['input_ids'].expand(batch_size, -1, -1).to(device)
            entity_type_mask = tokenized_entity_types['attention_mask'].expand(batch_size, -1, -1).to(device)

            # Get model outputs for loss calculation
            outputs = model(
                input_ids=pyg_batch.input_ids,
                attention_mask=pyg_batch.attention_mask,
                entity_type_ids=entity_type_ids,
                entity_type_mask=entity_type_mask,
                edge_index=pyg_batch.edge_index,
                node_to_token_idx=pyg_batch.node_to_token_idx,
                batch_map=pyg_batch.batch
            )
            
            # Calculate loss
            loss = compute_loss(outputs, pyg_batch, max_span_length)
            total_loss += loss.item()

            # Get predictions for metrics calculation
            predictions = model.predict(
                input_ids=pyg_batch.input_ids,
                attention_mask=pyg_batch.attention_mask,
                entity_type_ids=entity_type_ids,
                entity_type_mask=entity_type_mask,
                threshold=threshold,
                edge_index=pyg_batch.edge_index,
                node_to_token_idx=pyg_batch.node_to_token_idx,
                batch_map=pyg_batch.batch
            )

            # Compare predictions with ground truth for each item in the batch
            y_spans_ptr = pyg_batch.y_spans_ptr
            for b_idx in range(batch_size):
                # Predicted spans (convert to inclusive end for comparison)
                pred_spans = {(p['start'], p['end'] - 1, p['entity_type']) for p in predictions[b_idx]}

                # Ground truth spans
                gt_spans_tensor = pyg_batch.y_spans[y_spans_ptr[b_idx]:y_spans_ptr[b_idx+1]]
                gt_labels_tensor = pyg_batch.y_labels[y_spans_ptr[b_idx]:y_spans_ptr[b_idx+1]]
                gt_spans = {(span[0].item(), span[1].item(), label.item()) for span, label in zip(gt_spans_tensor, gt_labels_tensor)}

                total_tp += len(pred_spans.intersection(gt_spans))
                total_fp += len(pred_spans - gt_spans)
                total_fn += len(gt_spans - pred_spans)

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0.0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0
            
    return total_loss / len(dataloader), precision, recall, f1

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = GEGliNER(
        model_name=config['model']['name'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        num_gnn_layers=config['model']['num_gnn_layers'],
        n_heads=config['model']['gnn_heads'],
        max_span_length=config['model']['max_span_length']
    ).to(device)
    
    train_loader = create_dataloader(os.path.join(config['data']['processed_path'], 'train.pt'), config['training']['batch_size'])
    val_loader = create_dataloader(os.path.join(config['data']['processed_path'], 'val.pt'), config['training']['batch_size'], shuffle=False)
    
    collator = DataCollator(tokenizer, device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    num_training_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Prepare entity types for the model from config
    entity_types = config['data']['entity_types']
    tokenized_entity_types = tokenizer(entity_types, padding=True, return_tensors="pt")
    max_span_length = config['model']['max_span_length']
    eval_threshold = config['training'].get('eval_threshold', 0.5)

    best_val_f1 = 0.0
    save_dir = config['model']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        train_loss = train_epoch(
            model, train_loader, collator, optimizer, scheduler, device,
            tokenized_entity_types, max_span_length
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation loop
        val_loss, precision, recall, f1 = validate_epoch(
            model, val_loader, collator, device,
            tokenized_entity_types, max_span_length,
            threshold=eval_threshold
        )
        print(f"Validation Loss: {val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
        
        # Save model checkpoint if validation F1-score improves
        if f1 > best_val_f1:
            best_val_f1 = f1
            print(f"New best validation F1: {best_val_f1:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

if __name__ == "__main__":
    main()