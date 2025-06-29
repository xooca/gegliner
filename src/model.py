import torch
import torch.nn as nn
from transformers import AutoModel
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

class GEGliNER(nn.Module):
    """
    Graph Enhanced GLiNER: Combines transformer-based NER with graph neural networks
    for improved entity recognition using knowledge graph information.
    """
    def __init__(self, model_name: str, gnn_hidden_dim: int, num_gnn_layers: int, 
                 n_heads: int, max_span_length: int = 10, dropout: float = 0.1):
        super().__init__()
        
        # Core transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        self.hidden_dim = self.config.hidden_size
        self.num_gnn_layers = num_gnn_layers
        self.max_span_length = max_span_length
        self.dropout_rate = dropout

        # GNN layers for graph enhancement
        self.gnn_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        input_dim = self.hidden_dim
        for i in range(num_gnn_layers):
            # GAT layer for graph convolution
            self.gnn_layers.append(
                GATv2Conv(input_dim, gnn_hidden_dim // n_heads, heads=n_heads, dropout=dropout)
            )
            # Fusion gate to combine transformer and GNN features
            self.fusion_layers.append(
                nn.Linear(self.hidden_dim + gnn_hidden_dim, self.hidden_dim)
            )
            # Layer normalization for stability
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim))
            input_dim = gnn_hidden_dim

        # Span representation components (GLiNER style)
        self.span_ffnn = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),  # start + end + width embedding
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Width embeddings for span length encoding
        self.width_embeddings = nn.Embedding(max_span_length, self.hidden_dim)
        
        # Entity type representation
        self.entity_type_ffnn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                entity_type_ids: torch.Tensor, entity_type_mask: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None, 
                node_to_token_idx: Optional[torch.Tensor] = None, 
                batch_map: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining transformer encoding, graph enhancement, and span scoring.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            entity_type_ids: Entity type token IDs [batch_size, num_types, type_seq_len]
            entity_type_mask: Entity type attention mask [batch_size, num_types, type_seq_len]
            edge_index: Graph edges [2, num_edges] (optional for graph enhancement)
            node_to_token_idx: Mapping from graph nodes to tokens (optional)
            batch_map: Batch mapping for graph nodes (optional)
        
        Returns:
            Dictionary with span representations and scores
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Get initial token embeddings from transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # 2. Graph enhancement (if graph data is provided)
        if edge_index is not None and node_to_token_idx is not None and batch_map is not None:
            token_embeddings = self._apply_graph_enhancement(
                token_embeddings, attention_mask, edge_index, node_to_token_idx, batch_map
            )
        
        # 3. Get entity type representations
        entity_type_embeddings = self._get_entity_type_embeddings(
            entity_type_ids, entity_type_mask
        )  # [batch_size, num_types, hidden_dim]
        
        # 4. Generate span representations
        span_representations, span_masks = self._generate_span_representations(
            token_embeddings, attention_mask
        )
        
        # 5. Compute span-entity type scores
        scores = self._compute_scores(span_representations, entity_type_embeddings, span_masks)
        
        return {
            'span_representations': span_representations,
            'entity_type_embeddings': entity_type_embeddings,
            'scores': scores,
            'span_masks': span_masks,
            'token_embeddings': token_embeddings
        }
    
    def _apply_graph_enhancement(self, token_embeddings: torch.Tensor, 
                               attention_mask: torch.Tensor, edge_index: torch.Tensor,
                               node_to_token_idx: torch.Tensor, 
                               batch_map: torch.Tensor) -> torch.Tensor:
        """Apply graph neural network enhancement to token embeddings."""
        batch_size = token_embeddings.size(0)
        
        for layer_idx in range(self.num_gnn_layers):
            # Extract node features from token embeddings
            node_features = self._extract_node_features(
                token_embeddings, attention_mask, node_to_token_idx, batch_map
            )
            
            if node_features.size(0) == 0:  # No nodes to process
                continue
                
            # Apply GNN layer
            enhanced_node_features = self.gnn_layers[layer_idx](node_features, edge_index)
            enhanced_node_features = F.relu(enhanced_node_features)
            enhanced_node_features = self.dropout(enhanced_node_features)
            
            # Fuse enhanced features back into token embeddings
            token_embeddings = self._fuse_node_features(
                token_embeddings, enhanced_node_features, attention_mask,
                node_to_token_idx, batch_map, layer_idx
            )
            
            # Apply layer normalization
            token_embeddings = self.layer_norms[layer_idx](token_embeddings)
        
        return token_embeddings
    
    def _extract_node_features(self, token_embeddings: torch.Tensor, 
                             attention_mask: torch.Tensor, node_to_token_idx: torch.Tensor,
                             batch_map: torch.Tensor) -> torch.Tensor:
        """Extract node features from token embeddings using node-to-token mapping."""
        node_features_list = []
        
        for batch_idx in range(token_embeddings.size(0)):
            # Get nodes belonging to this batch
            batch_nodes = (batch_map == batch_idx)
            if not batch_nodes.any():
                continue
                
            batch_node_indices = node_to_token_idx[batch_nodes]
            # Clamp indices to valid range
            valid_indices = torch.clamp(batch_node_indices, 0, attention_mask[batch_idx].sum() - 1)
            
            # Extract corresponding token embeddings
            batch_node_features = token_embeddings[batch_idx, valid_indices]
            node_features_list.append(batch_node_features)
        
        if node_features_list:
            return torch.cat(node_features_list, dim=0)
        else:
            return torch.empty(0, token_embeddings.size(-1), device=token_embeddings.device)
    
    def _fuse_node_features(self, token_embeddings: torch.Tensor, 
                          enhanced_node_features: torch.Tensor, attention_mask: torch.Tensor,
                          node_to_token_idx: torch.Tensor, batch_map: torch.Tensor,
                          layer_idx: int) -> torch.Tensor:
        """Fuse enhanced node features back into token embeddings."""
        if enhanced_node_features.size(0) == 0:
            return token_embeddings
            
        fused_embeddings = token_embeddings.clone()
        node_offset = 0
        
        for batch_idx in range(token_embeddings.size(0)):
            batch_nodes = (batch_map == batch_idx)
            num_batch_nodes = batch_nodes.sum().item()
            
            if num_batch_nodes == 0:
                continue
                
            batch_node_indices = node_to_token_idx[batch_nodes]
            valid_indices = torch.clamp(batch_node_indices, 0, attention_mask[batch_idx].sum() - 1)
            
            # Get enhanced features for this batch
            batch_enhanced_features = enhanced_node_features[node_offset:node_offset + num_batch_nodes]
            
            # Get original token features
            original_features = token_embeddings[batch_idx, valid_indices]
            
            # Apply fusion gate
            combined_features = torch.cat([original_features, batch_enhanced_features], dim=-1)
            gate = torch.sigmoid(self.fusion_layers[layer_idx](combined_features))
            
            fused_features = gate * batch_enhanced_features + (1 - gate) * original_features
            
            # Update token embeddings
            fused_embeddings[batch_idx, valid_indices] = fused_features
            node_offset += num_batch_nodes
            
        return fused_embeddings
    
    def _get_entity_type_embeddings(self, entity_type_ids: torch.Tensor, 
                                  entity_type_mask: torch.Tensor) -> torch.Tensor:
        """Get embeddings for entity types."""
        batch_size, num_types, type_seq_len = entity_type_ids.shape
        
        # Flatten for processing
        flat_type_ids = entity_type_ids.view(-1, type_seq_len)
        flat_type_mask = entity_type_mask.view(-1, type_seq_len)
        
        # Get embeddings from transformer
        type_outputs = self.transformer(input_ids=flat_type_ids, attention_mask=flat_type_mask)
        type_embeddings = type_outputs.last_hidden_state  # [batch_size * num_types, type_seq_len, hidden_dim]
        
        # Pool embeddings (mean pooling over sequence length)
        pooled_embeddings = []
        for i in range(type_embeddings.size(0)):
            mask = flat_type_mask[i].bool()
            if mask.any():
                pooled = type_embeddings[i][mask].mean(dim=0)
            else:
                pooled = torch.zeros(self.hidden_dim, device=type_embeddings.device)
            pooled_embeddings.append(pooled)
        
        pooled_embeddings = torch.stack(pooled_embeddings)
        pooled_embeddings = pooled_embeddings.view(batch_size, num_types, self.hidden_dim)
        
        # Apply entity type transformation
        return self.entity_type_ffnn(pooled_embeddings)
    
    def _generate_span_representations(self, token_embeddings: torch.Tensor, 
                                     attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate representations for all possible spans."""
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        
        span_representations = []
        span_masks = []
        
        for batch_idx in range(batch_size):
            batch_spans = []
            batch_span_masks = []
            valid_len = attention_mask[batch_idx].sum().item()
            
            # Generate all spans up to max_span_length
            for start in range(valid_len):
                for end in range(start, min(start + self.max_span_length, valid_len)):
                    span_length = end - start + 1
                    
                    # Get start and end token embeddings
                    start_emb = token_embeddings[batch_idx, start]
                    end_emb = token_embeddings[batch_idx, end]
                    
                    # Get width embedding
                    width_emb = self.width_embeddings(torch.tensor(span_length - 1, device=token_embeddings.device))
                    
                    # Combine span features
                    span_repr = torch.cat([start_emb, end_emb, width_emb], dim=0)
                    batch_spans.append(span_repr)
                    batch_span_masks.append(True)
            
            # Pad spans to consistent length across batch
            if batch_spans:
                batch_spans = torch.stack(batch_spans)
                span_representations.append(batch_spans)
                span_masks.append(torch.tensor(batch_span_masks, device=token_embeddings.device))
            else:
                # Handle empty case
                empty_span = torch.zeros(hidden_dim * 3, device=token_embeddings.device)
                span_representations.append(empty_span.unsqueeze(0))
                span_masks.append(torch.tensor([False], device=token_embeddings.device))
        
        # Pad all batches to same number of spans
        max_spans = max(spans.size(0) for spans in span_representations)
        padded_spans = []
        padded_masks = []
        
        for spans, mask in zip(span_representations, span_masks):
            pad_size = max_spans - spans.size(0)
            if pad_size > 0:
                padding = torch.zeros(pad_size, spans.size(1), device=spans.device)
                spans = torch.cat([spans, padding], dim=0)
                mask_padding = torch.zeros(pad_size, dtype=torch.bool, device=mask.device)
                mask = torch.cat([mask, mask_padding], dim=0)
            
            padded_spans.append(spans)
            padded_masks.append(mask)
        
        span_representations = torch.stack(padded_spans)  # [batch_size, max_spans, hidden_dim * 3]
        span_masks = torch.stack(padded_masks)  # [batch_size, max_spans]
        
        # Apply span transformation
        span_representations = self.span_ffnn(span_representations)
        
        return span_representations, span_masks
    
    def _compute_scores(self, span_representations: torch.Tensor, 
                       entity_type_embeddings: torch.Tensor,
                       span_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute compatibility scores between spans and entity types using dot-product similarity.
        This is a core concept from GLiNER.
        """
        # span_representations: [batch_size, max_spans, hidden_dim]
        # entity_type_embeddings: [batch_size, num_types, hidden_dim]
        
        # Use batch matrix multiplication for efficient dot-product scoring
        # (b, s, h) @ (b, h, t) -> (b, s, t)
        scores = torch.bmm(span_representations, entity_type_embeddings.transpose(1, 2))
        
        num_types = entity_type_embeddings.size(1)
        
        # Mask invalid spans
        span_mask_expanded = span_masks.unsqueeze(-1).expand(-1, -1, num_types)
        scores = scores.masked_fill(~span_mask_expanded, float('-inf'))
        
        return scores

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                entity_type_ids: torch.Tensor, entity_type_mask: torch.Tensor,
                threshold: float = 0.5, **kwargs) -> List[List[Dict]]:
        """
        Predict entities for input sequences.
        
        Returns:
            List of predictions for each sequence in the batch
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_type_ids=entity_type_ids,
                entity_type_mask=entity_type_mask,
                **kwargs
            )
            
            scores = torch.sigmoid(outputs['scores'])  # Apply sigmoid for probabilities
            span_masks = outputs['span_masks']
            
            batch_predictions = []
            for batch_idx in range(scores.size(0)):
                batch_preds = []
                valid_len = attention_mask[batch_idx].sum().item()
                span_idx = 0
                
                # Reconstruct spans and their predictions
                for start in range(valid_len):
                    for end in range(start, min(start + self.max_span_length, valid_len)):
                        if span_idx < span_masks.size(1) and span_masks[batch_idx, span_idx]:
                            span_scores = scores[batch_idx, span_idx]
                            
                            # Find entity types above threshold
                            for type_idx, score in enumerate(span_scores):
                                if score > threshold:
                                    batch_preds.append({
                                        'start': start,
                                        'end': end + 1,  # End is exclusive
                                        'entity_type': type_idx,
                                        'score': score.item()
                                    })
                        span_idx += 1
                
                batch_predictions.append(batch_preds)
            
            return batch_predictions