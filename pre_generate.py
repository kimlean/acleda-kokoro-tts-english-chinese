#!/usr/bin/env python3
"""
Pre-generate audio cache for all parameter combinations to improve response time.
This script generates audio for all combinations of amount, currency, language, and thx_mode.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_server import TTSEngine
import time

def pre_generate_cache():
    """Pre-generate audio cache for all parameter combinations"""
    print("Starting audio cache pre-generation...")
    
    # Initialize TTS engine
    tts_engine = TTSEngine()
    
    # Parameters
    languages = ["EN", "CH"]  # English and Chinese
    currencies = ["USD", "KHR"]  # US Dollar and Khmer Riel
    thx_modes = [True, False]  # Thank you mode on/off
    
    # Amount ranges based on currency
    usd_amounts = []
    khr_amounts = []
    
    # Generate USD amounts: 0.10 to 100000 with step 0.01
    print("Generating USD amount list...")
    amount = 0.10
    while amount <= 100000:
        usd_amounts.append(round(amount, 2))
        amount += 0.01
    
    # Generate KHR amounts: 100 to 1000000 with step 100
    print("Generating KHR amount list...")
    khr_amounts = list(range(100, 1000001, 100))
    
    print(f"USD amounts: {len(usd_amounts)} values")
    print(f"KHR amounts: {len(khr_amounts)} values")
    
    total_combinations = 0
    generated_count = 0
    failed_count = 0
    
    # Calculate total combinations
    for language in languages:
        for thx_mode in thx_modes:
            total_combinations += len(usd_amounts) + len(khr_amounts)
    
    print(f"Total combinations to generate: {total_combinations}")
    
    start_time = time.time()
    
    # Generate for all combinations
    for language in languages:
        print(f"\nGenerating for language: {language}")
        
        for thx_mode in thx_modes:
            thx_status = "with thanks" if thx_mode else "without thanks"
            print(f"  Mode: {thx_status}")
            
            # Generate USD amounts
            print(f"    Generating USD amounts...")
            for i, amount in enumerate(usd_amounts):
                try:
                    audio_bytes = tts_engine.generate_speech(amount, "USD", language, 0.8, True, thx_mode=thx_mode)
                    generation_time = time.time() - start_time
                    print(f"Audio generated in {generation_time:.2f}s, size: {len(audio_bytes)} bytes")
                    
                    generated_count += 1
                    
                    if (i + 1) % 1000 == 0:
                        elapsed = time.time() - start_time
                        progress = (generated_count / total_combinations) * 100
                        print(f"      Progress: {generated_count}/{total_combinations} ({progress:.1f}%) - {elapsed:.1f}s elapsed")
                        
                except Exception as e:
                    print(f"      Failed to generate USD {amount} for {language} (thx:{thx_mode}): {e}")
                    failed_count += 1
            
            # Generate KHR amounts
            print(f"    Generating KHR amounts...")
            for i, amount in enumerate(khr_amounts):
                try:
                    audio_bytes = tts_engine.generate_speech(amount, "KHR", language, 0.8, True, thx_mode=thx_mode)
                    generation_time = time.time() - start_time
                    print(f"Audio generated in {generation_time:.2f}s, size: {len(audio_bytes)} bytes")
                    generated_count += 1
                    
                    if (i + 1) % 100 == 0:
                        elapsed = time.time() - start_time
                        progress = (generated_count / total_combinations) * 100
                        print(f"      Progress: {generated_count}/{total_combinations} ({progress:.1f}%) - {elapsed:.1f}s elapsed")
                        
                except Exception as e:
                    print(f"      Failed to generate KHR {amount} for {language} (thx:{thx_mode}): {e}")
                    failed_count += 1
    
    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== Cache Pre-generation Complete ===")
    print(f"Total combinations: {total_combinations}")
    print(f"Successfully generated: {generated_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per generation: {total_time/generated_count:.3f} seconds")

if __name__ == "__main__":
    pre_generate_cache()