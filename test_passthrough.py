#!/usr/bin/env python3

"""
Minimal test script to debug the PassthroughTokenizer pipeline
"""

import os
from levanter.data.text import UrlSingleDatasetLMConfig
from levanter.data.passthrough_tokenizer import PassthroughTokenizer

def test_basic_tokenizer():
    print("=== Testing PassthroughTokenizer directly ===")
    tokenizer = PassthroughTokenizer(55028)
    
    # Test with a sample line from your data
    sample_line = "55026 0 10042 27430 46 10018 19769 46"
    tokens = tokenizer._tokenize(sample_line)
    print(f"Input: {sample_line}")
    print(f"Tokens: {tokens}")
    print(f"Type: {type(tokens)}")
    print(f"Length: {len(tokens)}")
    print()

def test_data_source():
    print("=== Testing TextUrlDataSource ===")
    from levanter.data.sharded_datasource import TextUrlDataSource
    
    # Create a small test file
    test_file = "/tmp/test_tokens.txt"
    with open(test_file, "w") as f:
        f.write("55026 0 10042 27430 46\n")
        f.write("55026 0 10007 14260 7\n")
        f.write("55026 0 10007 27424 0\n")
    
    try:
        ds = TextUrlDataSource([test_file])
        print(f"Shard names: {ds.shard_names}")
        
        # Test reading from the data source
        for i, line in enumerate(ds.open_shard_at_row(ds.shard_names[0], 0)):
            print(f"Line {i}: {repr(line)}")
            if i >= 2:  # Only show first 3 lines
                break
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
    print()

def test_config_creation():
    print("=== Testing Config Creation ===")
    try:
        config = UrlSingleDatasetLMConfig(
            train_urls=["/user/s/siqichen/projects/amt/anticipation/data/train.txt"],
            validation_urls=["/user/s/siqichen/projects/amt/anticipation/data/valid.txt"],
            cache_dir="/tmp/test_cache",
            tokenizer="passthrough",
            vocab_size=55028,
            enforce_eos=False
        )
        print("Config created successfully")
        print(f"Tokenizer: {config.the_tokenizer}")
        print(f"Vocab size: {config.the_tokenizer.vocab_size}")
        
        # Test getting shard source
        source = config.get_shard_source("train")
        print(f"Train source: {source}")
        if source:
            print(f"Shard names: {source.shard_names}")
    except Exception as e:
        print(f"Error creating config: {e}")
        import traceback
        traceback.print_exc()
    print()

if __name__ == "__main__":
    test_basic_tokenizer()
    test_data_source()
    test_config_creation() 