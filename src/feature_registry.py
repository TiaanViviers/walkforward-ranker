"""Feature registry for managing feature ordering and metadata."""

import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime


class FeatureRegistry:
    """
    Manages feature ordering and metadata for reproducibility.
    
    Ensures that features are always in the same order during training
    and inference, which is critical for model consistency.
    """
    
    def __init__(self, registry_path: str):
        """
        Initialize feature registry.
        
        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry: Optional[Dict] = None
        
        if self.registry_path.exists():
            self.load()
    
    def create(self, feature_list: List[str], df: pd.DataFrame, version: str = "1.0.0"):
        """
        Create a new feature registry.
        
        Args:
            feature_list: Ordered list of feature names
            df: DataFrame containing the features (for dtype validation)
            version: Registry version string
        """
        self.registry = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'feature_count': len(feature_list),
            'feature_order': feature_list,
            'features': [
                {
                    'index': i,
                    'name': feat,
                    'dtype': str(df[feat].dtype)
                }
                for i, feat in enumerate(feature_list)
            ]
        }
        self.save()
        print(f"Created feature registry with {len(feature_list)} features")
    
    def load(self):
        """Load registry from disk."""
        with open(self.registry_path, 'r') as f:
            self.registry = json.load(f)
        print(f"Loaded feature registry: {len(self.registry['feature_order'])} features")
    
    def save(self):
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_feature_order(self) -> List[str]:
        """
        Get ordered list of features.
        
        Returns:
            List of feature names in correct order
        """
        if self.registry is None:
            raise ValueError("Registry not loaded. Call load() or create() first.")
        return self.registry['feature_order']
    
    def get_feature_count(self) -> int:
        """Get number of features in registry."""
        if self.registry is None:
            raise ValueError("Registry not loaded. Call load() or create() first.")
        return self.registry['feature_count']
    
    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that dataframe has correct features and return in correct order.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            DataFrame with features in correct order
            
        Raises:
            ValueError: If required features are missing
        """
        expected = self.get_feature_order()
        actual = df.columns.tolist()
        
        missing = set(expected) - set(actual)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        extra = set(actual) - set(expected)
        if extra:
            print(f"Warning: DataFrame contains extra features (will be ignored): {extra}")
        
        return df[expected]
    
    def get_info(self) -> Dict:
        """Get registry information."""
        if self.registry is None:
            raise ValueError("Registry not loaded.")
        
        return {
            'version': self.registry['version'],
            'created_at': self.registry['created_at'],
            'feature_count': self.registry['feature_count']
        }
