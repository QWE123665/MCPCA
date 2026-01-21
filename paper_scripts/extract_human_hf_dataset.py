#!/usr/bin/env python3
"""
Extract all sentences containing the word "human" from HuggingFace dataset
Creates a dataframe with: sentence, dataset_index, position, metadata
"""

import pandas as pd
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
import time

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

def extract_sentences_with_human(text, dataset_index, metadata):
    """
    Extract all sentences containing 'human' (case-insensitive)
    Returns list of dicts with sentence info
    """
    if not text:
        return []
    
    # Replace newlines with spaces to make text uniform
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        # Fallback to simple split if sent_tokenize fails
        sentences = re.split(r'[.!?]+', text)
    
    results = []
    char_position = 0
    
    for sent_idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        # Ensure no internal newlines in sentence (extra safety)
        sentence = ' '.join(sentence.split())
        
        # Check if sentence contains "human" (case-insensitive)
        if re.search(r'\bhuman\b', sentence, re.IGNORECASE):
            # Calculate character position in original text
            sentence_start = text.find(sentence, char_position)
            if sentence_start == -1:
                sentence_start = char_position
            
            results.append({
                'sentence': sentence,
                'dataset_index': dataset_index,
                'sentence_index': sent_idx,
                'char_position': sentence_start,
                'text_length': len(text),
                'source': metadata.get('source', ''),
                'metadata': json.dumps(metadata) if metadata else ''
            })
        
        char_position += len(sentence)
    
    return results

def main():
    # # You need to load your dataset first
    # # Example:
    
    
    # print("Note: Please ensure your dataset 'ds' is loaded before running this script.")
    # print("Example:")
    # print("  from datasets import load_dataset")
    # print("  ds = load_dataset('your_dataset_name')")
    # print()
    
    # # For now, assuming ds is already loaded in your environment
    # # If running as standalone script, you'll need to load it here
    # try:
    #     # Try to import from the calling environment
    #     import __main__
    #     if hasattr(__main__, 'ds'):
    #         ds = __main__.ds
    #     else:
    #         print("Error: Dataset 'ds' not found in environment.")
    #         print("Please load your dataset first, then run this function.")
    #         return None
    # except:
    #     print("Error: Could not access dataset.")
    #     print("Run this script interactively or modify to load your dataset.")
    #     return None
    from datasets import load_dataset
    ds = load_dataset("sedthh/gutenberg_english")['train']
    print(f"Processing dataset with {len(ds)} rows...")
    
    # Process all rows
    all_sentences = []
    start_time = time.time()
    
    for i in range(len(ds)):
        # Get text and metadata
        text = ds[i]['TEXT']
        source = ds[i].get('SOURCE', '')
        metadata = ds[i].get('METADATA', {})
        
        # Parse metadata if it's a string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {'raw': metadata}
        
        # Add source to metadata for convenience
        if not isinstance(metadata, dict):
            metadata = {}
        metadata['source'] = source
        
        # Calculate ETA
        if i > 0 and i % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = len(ds) - i
            eta_seconds = avg_time * remaining
            eta_minutes = int(eta_seconds / 60)
            eta_seconds = int(eta_seconds % 60)
            eta_str = f"ETA: {eta_minutes}m {eta_seconds:02d}s"
            progress_pct = (i / len(ds)) * 100
            print(f"[{i}/{len(ds)}] {progress_pct:.1f}% | {eta_str} | Found {len(all_sentences)} sentences".ljust(100), end='\r')
        
        # Extract sentences
        sentences = extract_sentences_with_human(text, i, metadata)
        all_sentences.extend(sentences)
    
    print(f"\n\nProcessing complete!")
    print(f"Total sentences with 'human': {len(all_sentences)}")
    
    # Create dataframe
    print("\nCreating dataframe...")
    df = pd.DataFrame(all_sentences)
    
    # Sort by dataset index and sentence position
    df = df.sort_values(['dataset_index', 'sentence_index'])
    df = df.reset_index(drop=True)
    
    # Show some statistics
    print("\nDataframe Statistics:")
    print(f"Total rows: {len(df)}")
    print(f"\nUnique sources:")
    print(df['source'].value_counts().head(10))
    
    # Show sample
    print("\nSample rows:")
    print(df[['dataset_index', 'source', 'sentence_index', 'sentence']].head(10))
    
    # Save to CSV
    output_file = "human_sentences.csv"
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Done! CSV saved with {len(df)} rows")
    
    # Also save as parquet for better compression
    parquet_file = output_file.replace('.csv', '.parquet')
    print(f"Saving to {parquet_file} (more efficient format)...")
    df.to_parquet(parquet_file, index=False)
    print(f"Done! Parquet saved")
    
    return df

# Standalone function that can be imported
def extract_from_dataset(ds, output_file="human_sentences.csv"):
    """
    Extract sentences containing 'human' from a HuggingFace dataset
    
    Args:
        ds: HuggingFace Dataset with 'TEXT' column
        output_file: Path to output CSV file
    
    Returns:
        pandas DataFrame with extracted sentences
    """
    print(f"Processing dataset with {len(ds)} rows...")
    
    all_sentences = []
    start_time = time.time()
    
    for i in range(len(ds)):
        text = ds[i]['TEXT']
        source = ds[i].get('SOURCE', '')
        metadata = ds[i].get('METADATA', {})
        
        # Parse metadata if it's a string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {'raw': metadata}
        
        if not isinstance(metadata, dict):
            metadata = {}
        metadata['source'] = source
        
        # Progress
        if i > 0 and i % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = len(ds) - i
            eta_seconds = avg_time * remaining
            eta_minutes = int(eta_seconds / 60)
            eta_seconds = int(eta_seconds % 60)
            progress_pct = (i / len(ds)) * 100
            print(f"[{i}/{len(ds)}] {progress_pct:.1f}% | ETA: {eta_minutes}m {eta_seconds:02d}s | Found {len(all_sentences)} sentences".ljust(100), end='\r')
        
        sentences = extract_sentences_with_human(text, i, metadata)
        all_sentences.extend(sentences)
    
    print(f"\n\nComplete! Found {len(all_sentences)} sentences with 'human'")
    
    # Create and save dataframe
    df = pd.DataFrame(all_sentences)
    df = df.sort_values(['dataset_index', 'sentence_index']).reset_index(drop=True)
    
    df.to_csv(output_file, index=False)
    df.to_parquet(output_file.replace('.csv', '.parquet'), index=False)
    
    print(f"Saved to {output_file}")
    return df

if __name__ == "__main__":
    main()