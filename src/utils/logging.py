import logging
import os
from datetime import datetime

def setup_logging(log_dir: str = 'logs'):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set log level for specific modules
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.INFO)
    
    return logging.getLogger(__name__) 