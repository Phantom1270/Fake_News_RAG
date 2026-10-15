import json
import logging
import os
import time
import random
from pathlib import Path
from config import VERDICT_MAP, CLASS_VERDICT_MAP, RATE_LIMIT

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('debug', exist_ok=True)
    
    logger = logging.getLogger('altnews_scraper')
    logger.setLevel(logging.INFO)
    
    # File handlers
    fh = logging.FileHandler('logs/scraper.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    err_fh = logging.FileHandler('logs/scraper_errors.log', encoding='utf-8')
    err_fh.setLevel(logging.ERROR)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    err_fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(err_fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

def normalize_verdict(raw_text: str, css_class: str = None) -> str:
    """Normalize the raw verdict text or CSS class to a standard label."""
    if css_class:
        # Check class map first as it's more reliable
        for class_key, normalized in CLASS_VERDICT_MAP.items():
            if class_key in css_class.lower():
                return normalized
                
    if raw_text:
        raw_lower = raw_text.lower().strip()
        for text_key, normalized in VERDICT_MAP.items():
            if text_key in raw_lower:
                return normalized
                
    return "UNKNOWN"

class CheckpointManager:
    def __init__(self, filename="checkpoint.json"):
        self.filename = filename
        
    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {"listing_last_page": 0, "articles_scraped": 0}
        
    def save(self, data):
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=2)

def wait_with_jitter(base_delay: float):
    """Wait for base_delay + up to jitter_seconds random jitter."""
    jitter = random.uniform(0, RATE_LIMIT['jitter_seconds'])
    total_wait = base_delay + jitter
    time.sleep(total_wait)

def save_debug_html(page_num: int, content: str, reason: str = "failed"):
    """Save raw HTML to debug folder when something goes wrong."""
    timestamp = int(time.time())
    filename = f"debug/altnews_page_{page_num}_{reason}_{timestamp}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Saved debug HTML to {filename}")
