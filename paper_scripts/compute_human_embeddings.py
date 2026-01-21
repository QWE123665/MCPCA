#!/usr/bin/env python3
"""
Compute contextualized word embeddings for "human" in each sentence using the output of extract_human_hf_dataset.py
Uses transformer models (BERT, RoBERTa, etc.) to get context-dependent embeddings
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import re
from pathlib import Path
import time

def find_human_token_positions(sentence, tokenizer):
    """
    Find the token positions corresponding to the word "human" in the sentence
    Returns list of (start_token_idx, end_token_idx) tuples for each occurrence
    """
    # Tokenize the sentence
    tokens = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    
    # Get the character offsets for each token
    offset_mapping = tokens['offset_mapping']
    
    # Find all occurrences of "human" (case-insensitive) in the original sentence
    human_positions = []
    for match in re.finditer(r'\bhuman\b', sentence, re.IGNORECASE):
        char_start = match.start()
        char_end = match.end()
        
        # Find which tokens correspond to this occurrence
        token_start = None
        token_end = None
        
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            # Skip special tokens (they have offset (0, 0))
            if tok_start == 0 and tok_end == 0 and idx != 0:
                continue
                
            # Check if this token overlaps with "human"
            if tok_start < char_end and tok_end > char_start:
                if token_start is None:
                    token_start = idx
                token_end = idx
        
        if token_start is not None:
            human_positions.append((token_start, token_end))
    
    return human_positions

def get_human_embedding(sentence, model, tokenizer, pooling='mean'):
    """
    Get the contextualized embedding for "human" in a sentence
    
    Args:
        sentence: The sentence containing "human"
        model: The transformer model
        tokenizer: The tokenizer
        pooling: How to combine subword tokens ('mean', 'first', 'last')
    
    Returns:
        numpy array of the embedding, or None if "human" not found
    """
    # Find token positions for "human"
    human_positions = find_human_token_positions(sentence, tokenizer)
    
    if not human_positions:
        return None
    
    # Tokenize and get embeddings
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Get last hidden state (shape: batch_size, seq_len, hidden_dim)
        hidden_states = outputs.last_hidden_state[0]  # Remove batch dimension
    
    # Get embedding for the first occurrence of "human"
    # (If multiple occurrences, you might want to handle differently)
    token_start, token_end = human_positions[0]
    
    # Extract tokens corresponding to "human"
    human_tokens = hidden_states[token_start:token_end+1]
    
    # Pool the subword tokens
    if pooling == 'mean':
        embedding = human_tokens.mean(dim=0)
    elif pooling == 'first':
        embedding = human_tokens[0]
    elif pooling == 'last':
        embedding = human_tokens[-1]
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")
    
    return embedding.cpu().numpy()

def compute_embeddings_for_dataframe(df, model_name='bert-base-uncased', pooling='mean', batch_size=32):
    """
    Compute embeddings for all sentences in the dataframe
    
    Args:
        df: DataFrame with 'sentence' column
        model_name: HuggingFace model name
        pooling: How to pool subword tokens
        batch_size: Process sentences in batches (not used in simple version)
    
    Returns:
        numpy array of shape (n_sentences, embedding_dim)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    embeddings = []
    failed_indices = []
    start_time = time.time()
    
    print(f"\nComputing embeddings for {len(df)} sentences...")
    
    for idx, row in df.iterrows():
        # Progress update
        if (idx + 1) % 100 == 0 or idx == 0:
            if idx > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = len(df) - idx - 1
                eta_seconds = avg_time * remaining
                eta_minutes = int(eta_seconds / 60)
                eta_seconds = int(eta_seconds % 60)
                progress_pct = ((idx + 1) / len(df)) * 100
                print(f"[{idx+1}/{len(df)}] {progress_pct:.1f}% | ETA: {eta_minutes}m {eta_seconds:02d}s", end='\r')
        
        sentence = row['sentence']
        
        try:
            # Move inputs to device
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Find token positions
            human_positions = find_human_token_positions(sentence, tokenizer)
            
            if not human_positions:
                failed_indices.append(idx)
                embeddings.append(None)
                continue
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state[0]
            
            # Extract "human" tokens
            token_start, token_end = human_positions[0]
            human_tokens = hidden_states[token_start:token_end+1]
            
            # Pool
            if pooling == 'mean':
                embedding = human_tokens.mean(dim=0)
            elif pooling == 'first':
                embedding = human_tokens[0]
            elif pooling == 'last':
                embedding = human_tokens[-1]
            
            embeddings.append(embedding.cpu().numpy())
            
        except Exception as e:
            print(f"\nError processing sentence {idx}: {e}")
            failed_indices.append(idx)
            embeddings.append(None)
    
    print(f"\n\nEmbedding computation complete!")
    print(f"Successfully processed: {len(df) - len(failed_indices)}")
    print(f"Failed: {len(failed_indices)}")
    
    return embeddings, failed_indices

def main():
    INPUT_FILE = "human_sentences.csv"
    OUTPUT_EMBEDDINGS = "human_embeddings.npy"
    OUTPUT_DF = "human_sentences_with_embeddings.parquet"
    
    # Model options:
    # - 'bert-base-uncased' (768 dim, good baseline)
    # - 'roberta-base' (768 dim, often better)
    # - 'bert-large-uncased' (1024 dim, slower but more accurate)
    # - 'distilbert-base-uncased' (768 dim, faster)
    MODEL_NAME = 'bert-base-uncased'
    POOLING = 'mean'  # 'mean', 'first', or 'last'
    
    # Load the dataframe
    print(f"Loading {INPUT_FILE}...")
    if not Path(INPUT_FILE).exists():
        print(f"Error: {INPUT_FILE} not found!")
        print("Please run extract_human_sentences.py first.")
        return
    
    df = pd.read_csv(INPUT_FILE)
    
    ds_metadata = pd.read_pickle('ds_metadata.pkl')
    df = df[(df['dataset_index'].isin(ds_metadata['ds_index'])) & (df['sentence'].str.len() < 600)]
    print(f"Loaded {len(df)} sentences")
    
    # Optional: sample for testing
    # df = df.head(100)  # Uncomment to test on first 100 sentences
    
    # Compute embeddings
    embeddings, failed_indices = compute_embeddings_for_dataframe(
        df, 
        model_name=MODEL_NAME,
        pooling=POOLING
    )
    
    # Remove failed sentences
    if failed_indices:
        print(f"\nRemoving {len(failed_indices)} failed sentences...")
        df = df.drop(failed_indices).reset_index(drop=True)
        embeddings = [emb for i, emb in enumerate(embeddings) if i not in failed_indices]
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    print(f"\nEmbeddings shape: {embeddings_array.shape}")
    
    # Save embeddings
    print(f"Saving embeddings to {OUTPUT_EMBEDDINGS}...")
    np.save(OUTPUT_EMBEDDINGS, embeddings_array)
    
    # Save dataframe with embedding indices
    print(f"Saving dataframe to {OUTPUT_DF}...")
    df['embedding_index'] = range(len(df))
    df.to_parquet(OUTPUT_DF, index=False)
    
    print("\nDone!")
    print(f"Embeddings saved to: {OUTPUT_EMBEDDINGS}")
    print(f"DataFrame saved to: {OUTPUT_DF}")
    print(f"\nTo load later:")
    print(f"  embeddings = np.load('{OUTPUT_EMBEDDINGS}')")
    print(f"  df = pd.read_parquet('{OUTPUT_DF}')")
    print(f"  # Access embedding for row i: embeddings[df.loc[i, 'embedding_index']]")

if __name__ == "__main__":
    main()