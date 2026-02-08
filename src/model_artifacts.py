"""Model artifact management for saving and loading models with full metadata."""

import joblib
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class ModelArtifact:
    """
    Manages saving and loading of models with all associated metadata.
    
    Each model is saved with:
    - The model itself (pickle)
    - Feature list (for inference)
    - Configuration used for training
    - Metadata (timestamp, metrics, etc.)
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize model artifact manager.
        
        Args:
            models_dir: Directory where models are stored
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def save(
        self, 
        model: Any,
        config: Dict,
        feature_list: List[str],
        metrics: Optional[Dict] = None,
        run_id: Optional[str] = None,
        additional_metadata: Optional[Dict] = None
    ) -> str:
        """
        Save model with all metadata.
        
        Creates directory structure:
        models/{run_id}/
            ├── model.pkl
            ├── feature_list.json
            ├── config.yaml
            └── metadata.json
        
        Args:
            model: Trained model to save
            config: Configuration dict used for training
            feature_list: Ordered list of feature names
            metrics: Optional performance metrics dict
            run_id: Optional run identifier (auto-generated if None)
            additional_metadata: Optional additional metadata to save
            
        Returns:
            run_id: Identifier for this model run
        """
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        run_dir = self.models_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save model
        joblib.dump(model, run_dir / 'model.pkl')
        
        # Save feature list (critical for inference)
        with open(run_dir / 'feature_list.json', 'w') as f:
            json.dump({
                'features': feature_list,
                'feature_count': len(feature_list)
            }, f, indent=2)
        
        # Save config
        with open(run_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Save metadata
        metadata = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(feature_list),
            'metrics': metrics or {}
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        with open(run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Model saved (message printed in calling script)
        print(f"  Features: {len(feature_list)}")
        print(f"  Run ID: {run_id}")
        
        return run_id
    
    def load(self, run_id: str) -> Dict:
        """
        Load model with all metadata.
        
        Args:
            run_id: Identifier of the model to load
            
        Returns:
            Dictionary containing:
                - model: The loaded model
                - feature_list: List of feature names
                - config: Configuration dict
                - metadata: Metadata dict
                
        Raises:
            ValueError: If model not found
        """
        run_dir = self.models_dir / run_id
        
        if not run_dir.exists():
            raise ValueError(f"Model {run_id} not found in {self.models_dir}")
        
        # Load model
        model = joblib.load(run_dir / 'model.pkl')
        
        # Load feature list
        with open(run_dir / 'feature_list.json', 'r') as f:
            feature_data = json.load(f)
        
        # Load config
        with open(run_dir / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load metadata
        with open(run_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"Model loaded: {run_id}")
        print(f"  Features: {feature_data['feature_count']}")
        print(f"  Timestamp: {metadata['timestamp']}")
        
        return {
            'model': model,
            'feature_list': feature_data['features'],
            'config': config,
            'metadata': metadata
        }
    
    def list_models(self) -> List[Dict]:
        """
        List all saved models with metadata.
        
        Returns:
            List of metadata dicts, sorted by timestamp (newest first)
        """
        models = []
        
        for run_dir in self.models_dir.iterdir():
            if run_dir.is_dir() and (run_dir / 'metadata.json').exists():
                with open(run_dir / 'metadata.json', 'r') as f:
                    metadata = json.load(f)
                models.append(metadata)
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)
    
    def delete(self, run_id: str):
        """
        Delete a saved model.
        
        Args:
            run_id: Identifier of the model to delete
        """
        run_dir = self.models_dir / run_id
        
        if not run_dir.exists():
            raise ValueError(f"Model {run_id} not found")
        
        import shutil
        shutil.rmtree(run_dir)
        print(f"Deleted model: {run_id}")
    
    def get_latest(self) -> Optional[str]:
        """
        Get run_id of the most recently saved model.
        
        Returns:
            run_id of latest model, or None if no models exist
        """
        models = self.list_models()
        return models[0]['run_id'] if models else None
