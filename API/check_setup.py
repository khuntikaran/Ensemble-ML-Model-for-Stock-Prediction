import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check the Python environment and dependencies"""
    try:
        # Add project root to Python path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(project_root)
        
        logger.info("Checking environment setup...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Python path: {sys.path}")
        
        # Check required modules
        import flask
        logger.info(f"Flask version: {flask.__version__}")
        
        import pandas as pd
        logger.info(f"Pandas version: {pd.__version__}")
        
        # Try importing the prediction module
        from src.models.predict import predict_next_price
        logger.info("Successfully imported prediction module")
        
        # Check directory structure
        dirs_to_check = [
            os.path.join(project_root, 'src', 'models'),
            os.path.join(project_root, 'src', 'API'),
            os.path.join(project_root, 'src', 'templates'),
            os.path.join(project_root, 'src', 'static'),
            os.path.join(project_root, 'src', 'frontend', 'build'),
        ]
        
        for directory in dirs_to_check:
            status = "✓ exists" if os.path.exists(directory) else "✗ missing"
            logger.info(f"Directory {directory}: {status}")
        
        logger.info("Environment check completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Environment check failed: {str(e)}")
        return False

if __name__ == "__main__":
    check_environment() 