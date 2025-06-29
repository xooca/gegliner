# scripts/predict.py

import torch
import yaml
import argparse
from transformers import AutoTokenizer

from src.model import GEGliNER

def predict(model, tokenizer, text, entity_types, device, threshold=0.5):
    """
    Runs inference on a single piece of text.

    Args:
        model (GEGliNER): The trained model.
        tokenizer: The tokenizer.
        text (str): The input text.
        entity_types (list): A list of entity type names.
        device: The torch device.
        threshold (float): The probability threshold for predictions.

    Returns:
        A list of dictionaries, where each dictionary represents a found entity.
    """
    model.eval()

    # 1. Tokenize input text and entity types
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    tokenized_entity_types = tokenizer(entity_types, padding=True, return_tensors="pt")

    # 2. Prepare entity type inputs for the model (batch size of 1)
    entity_type_ids = tokenized_entity_types['input_ids'].unsqueeze(0).to(device)
    entity_type_mask = tokenized_entity_types['attention_mask'].unsqueeze(0).to(device)

    # 3. Run prediction using the model's predict method
    predictions = model.predict(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        entity_type_ids=entity_type_ids,
        entity_type_mask=entity_type_mask,
        threshold=threshold
    )

    # 4. Format the output
    results = []
    # The result is a list containing one list of predictions for our single input
    if predictions and predictions[0]:
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0], skip_special_tokens=True)
        for pred in predictions[0]:
            start, end = pred['start'], pred['end']
            entity_label = entity_types[pred['entity_type']]
            # Use tokenizer to correctly convert tokens back to a string
            span_text = tokenizer.convert_tokens_to_string(tokens[start:end])
            
            results.append({
                "span": span_text.strip(),
                "label": entity_label,
                "score": pred['score']
            })
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference with GE-GliNER model.")
    parser.add_argument("text", type=str, help="The text to run NER on.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model checkpoint. Defaults to the path in config.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction score threshold.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    entity_types = config['data']['entity_types']

    model = GEGliNER(**config['model']).to(device)

    model_path = args.model_path or f"{config['model']['save_dir']}/best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    predictions = predict(model, tokenizer, args.text, entity_types, device, args.threshold)

    print("\n--- Predictions ---")
    print(f"Input text: \"{args.text}\"\n")
    if predictions:
        for p in sorted(predictions, key=lambda x: x['score'], reverse=True):
            print(f"  - Span: '{p['span']}'\n    Label: {p['label']}\n    Score: {p['score']:.4f}")
    else:
        print("No entities found.")

if __name__ == "__main__":
    main()