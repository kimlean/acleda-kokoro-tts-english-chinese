#!/usr/bin/env python3
"""
Pre-generate audio cache for all parameter combinations to improve response time.
This script generates audio for all combinations of amount, currency, language, and thx_mode.
"""

import sys
import os
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from queue import Queue

class APIService:
    """REST API client for TTS service"""
    
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_speech(self, amount, currency, language, speed=0.8, thx_mode=False):
        """Generate speech via REST API"""
        url = f"{self.base_url}/voicegenerate"
        
        payload = {
            "amount": amount,
            "currency": currency,
            "language": language,
            "speed": speed,
            "thx_mode": thx_mode
        }
        
        try:
            response = self.session.post(url, params=payload, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")

def generate_single_audio(api_service, amount, currency, language, thx_mode):
    """Generate a single audio file"""
    try:
        audio_bytes = api_service.generate_speech(amount, currency, language, 0.8, thx_mode=thx_mode)
        return {
            'success': True,
            'amount': amount,
            'currency': currency,
            'language': language,
            'thx_mode': thx_mode,
            'size': len(audio_bytes),
            'audio_bytes': audio_bytes
        }
    except Exception as e:
        return {
            'success': False,
            'amount': amount,
            'currency': currency,
            'language': language,
            'thx_mode': thx_mode,
            'error': str(e)
        }

def pre_generate_cache():
    """Pre-generate audio cache for all parameter combinations using threading"""
    print("Starting audio cache pre-generation with threading...")
    
    # Initialize API service
    api_service = APIService()
    
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
    
    # Create task list for all combinations
    tasks = []
    for language in languages:
        for thx_mode in thx_modes:
            # Add USD tasks
            for amount in usd_amounts:
                tasks.append((amount, "USD", language, thx_mode))
            # Add KHR tasks
            for amount in khr_amounts:
                tasks.append((amount, "KHR", language, thx_mode))
    
    total_combinations = len(tasks)
    print(f"Total combinations to generate: {total_combinations}")
    
    generated_count = 0
    failed_count = 0
    start_time = time.time()
    
    # Use ThreadPoolExecutor for concurrent processing
    max_workers = 5  # Adjust based on your system capacity
    print(f"Using {max_workers} worker threads")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(generate_single_audio, api_service, amount, currency, language, thx_mode): 
            (amount, currency, language, thx_mode)
            for amount, currency, language, thx_mode in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            result = future.result()
            
            if result['success']:
                generated_count += 1
                elapsed = time.time() - start_time
                print(f"✓ {result['currency']} {result['amount']} ({result['language']}, thx:{result['thx_mode']}) - Size: {result['size']} bytes")
                
                # Progress update every 100 successful generations
                if generated_count % 100 == 0:
                    progress = (generated_count + failed_count) / total_combinations * 100
                    print(f"  Progress: {generated_count + failed_count}/{total_combinations} ({progress:.1f}%) - {elapsed:.1f}s elapsed")
            else:
                failed_count += 1
                print(f"✗ Failed {result['currency']} {result['amount']} ({result['language']}, thx:{result['thx_mode']}): {result['error']}")
    
    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== Cache Pre-generation Complete ===")
    print(f"Total combinations: {total_combinations}")
    print(f"Successfully generated: {generated_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time:.2f} seconds")
    if generated_count > 0:
        print(f"Average time per generation: {total_time/generated_count:.3f} seconds")
    print(f"Threads used: {max_workers}")

if __name__ == "__main__":
    pre_generate_cache()