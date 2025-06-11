#!/usr/bin/env python3

import asyncio
import numpy as np
from levanter.data.text import load_lm_dataset_cache, TextLmDatasetFormat
from levanter.data.passthrough_tokenizer import PassthroughTokenizer

async def check_cache():
    print("Checking validation cache...")
    
    # Create tokenizer and format
    tokenizer = PassthroughTokenizer(55028)
    format = TextLmDatasetFormat()
    
    try:
        # Load the cache
        cache = load_lm_dataset_cache(
            "/user/s/siqichen/projects/amt/anticipation/data/cache/validation",
            format,
            tokenizer,
            enforce_eos=False
        )
        
        print("✅ Cache loaded successfully!")
        print(f"Number of rows: {cache.store.tree['input_ids'].num_rows}")
        print(f"Data size: {cache.store.tree['input_ids'].data_size}")
        
        # Get a small sample - need to await the read operation
        sample_future = cache.store.tree['input_ids'].data[0:20].read()
        sample = await sample_future
        print(f"Sample data shape: {sample.shape}")
        print(f"Sample data: {sample}")
        print(f"Sample data type: {sample.dtype}")
        
        # Check if we have any non-zero values
        non_zero_count = np.count_nonzero(sample)
        print(f"Non-zero values in sample: {non_zero_count}/{len(sample)}")
        
        if non_zero_count > 0:
            print("✅ Data looks good - contains actual token IDs!")
        else:
            print("❌ Data appears to be all zeros - there might be an issue")
            
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_cache()) 