"""ZenML Stack Configuration for BTC MLOps."""

import logging

logger = logging.getLogger(__name__)


def initialize_zenml_stack():
    """
    Initialize ZenML default stack with necessary components.
    
    Stack Components:
    - Orchestrator: Local (default)
    - Artifact Store: Local filesystem
    - Experiment Tracker: MLflow
    - Model Registry: MLflow
    """
    try:
        from zenml.client import Client
        
        logger.info("🔧 Initializing ZenML stack...")
        
        client = Client()
        
        # Check if stack exists
        try:
            stack = client.active_stack
            logger.info(f"✅ Using existing stack: {stack.name}")
            logger.info(f"   Orchestrator: {stack.orchestrator}")
            if hasattr(stack, 'experiment_tracker') and stack.experiment_tracker:
                logger.info(f"   Experiment Tracker: {stack.experiment_tracker}")
            if hasattr(stack, 'model_registry') and stack.model_registry:
                logger.info(f"   Model Registry: {stack.model_registry}")
            return True
        except Exception as e:
            logger.info(f"⚠️  Creating default stack: {e}")
            
            # Default stack will be created automatically
            stack = client.active_stack
            logger.info(f"✅ Default stack created: {stack.name}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize ZenML stack: {e}")
        return False


def get_zenml_config():
    """Get ZenML configuration details."""
    try:
        from zenml.client import Client
        
        client = Client()
        stack = client.active_stack
        
        config = {
            "stack_name": stack.name,
            "orchestrator": str(stack.orchestrator),
            "artifact_store": str(stack.artifact_store),
        }
        
        if hasattr(stack, 'experiment_tracker') and stack.experiment_tracker:
            config["experiment_tracker"] = str(stack.experiment_tracker)
        
        if hasattr(stack, 'model_registry') and stack.model_registry:
            config["model_registry"] = str(stack.model_registry)
        
        return config
    except Exception as e:
        logger.error(f"Failed to get ZenML config: {e}")
        return {}


if __name__ == "__main__":
    initialize_zenml_stack()
    config = get_zenml_config()
    print("ZenML Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
