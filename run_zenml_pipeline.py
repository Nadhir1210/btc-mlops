"""
ZenML Pipeline Runner for BTC MLOps.

This script initializes and runs the complete ZenML pipeline
for BTC price prediction model training and deployment.

Usage:
    python run_zenml_pipeline.py
    
    # With verbose logging
    python run_zenml_pipeline.py --verbose
    
    # Dry run (show pipeline DAG without execution)
    python run_zenml_pipeline.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_zenml_environment():
    """Initialize ZenML project and stack."""
    try:
        from zenml.client import Client
        
        logger.info("🔧 Checking ZenML environment...")
        
        client = Client()
        
        # Ensure default stack exists
        try:
            stack = client.active_stack
            logger.info(f"✅ Using stack: {stack.name}")
        except Exception as e:
            logger.warning(f"⚠️ Stack not found: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize ZenML: {e}")
        return False


def run_pipeline(dry_run=False, verbose=False):
    """
    Run the ZenML pipeline for BTC MLOps.
    
    Args:
        dry_run: If True, only visualize the pipeline DAG
        verbose: If True, enable verbose logging
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize ZenML
        logger.info("=" * 70)
        logger.info("🚀 Starting BTC MLOps ZenML Pipeline")
        logger.info("=" * 70)
        
        if not setup_zenml_environment():
            return False
        
        # Import pipeline
        from pipelines.zenml_pipeline import btc_training_pipeline
        
        logger.info("\n📋 Pipeline Structure:")
        logger.info("  1. prepare_data_step  → Features engineering & splitting")
        logger.info("  2. train_model_step   → CatBoost model training")
        logger.info("  3. evaluate_model_step → Metrics calculation")
        logger.info("  4. export_model_step   → MLflow registration")
        
        if dry_run:
            logger.info("\n🔍 Dry-run mode: Pipeline structure only (no execution)")
            logger.info(f"✅ Pipeline DAG created successfully")
            logger.info(f"   Pipeline name: btc_training_pipeline")
            logger.info(f"   Steps: prepare_data → train_model → evaluate → export")
            return True
        
        # Run pipeline
        logger.info("\n▶️  Running pipeline...")
        result = btc_training_pipeline()
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Pipeline Execution Completed Successfully!")
        logger.info("=" * 70)
        logger.info(f"📊 Output files:")
        logger.info(f"   - data/processed/X_train.pkl")
        logger.info(f"   - data/processed/X_test.pkl")
        logger.info(f"   - data/processed/y_train.pkl")
        logger.info(f"   - data/processed/y_test.pkl")
        logger.info(f"   - data/processed/scaler.pkl")
        logger.info(f"   - data/processed/metrics.json")
        logger.info(f"   - src/training/catboost_model.pkl")
        logger.info(f"   - MLflow registry: BTC_Price_Predictor")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("   Ensure all dependencies are installed:")
        logger.error("   pip install zenml mlflow scikit-learn catboost pandas")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run ZenML pipeline for BTC MLOps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_zenml_pipeline.py              # Run pipeline normally
  python run_zenml_pipeline.py --dry-run    # Show pipeline structure only
  python run_zenml_pipeline.py --verbose    # Run with debug logging
  python run_zenml_pipeline.py --help       # Show this help
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pipeline DAG without executing steps"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    success = run_pipeline(
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
