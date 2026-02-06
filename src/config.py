"""Configuration management and validation."""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data-related configuration."""
    date_col: str = "date"
    group_col: str = "group_id"
    label_col: str = "relevance_grade"  # Label for training (relevance grades)
    pnl_col: str = "pnl"  # Actual PnL for evaluation metrics
    config_features: List[str] = field(default_factory=list)
    market_features: List[str] = field(default_factory=list)
    calendar_features: List[str] = field(default_factory=list)
    path: str = ""  # Optional, for backwards compatibility


@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration."""
    initial_train_days: int = 365
    train_window_days: int = 365
    test_window_days: int = 30
    retrain_frequency_days: int = 7
    expanding_window: bool = False
    max_training_window_days: int = 504  # Cap for expanding window (2 trading years default)


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    objective: str = "lambdarank"
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 100
    random_state: int = 42
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionConfig:
    """Selection policy configuration."""
    top_k: int = 3


@dataclass
class FeatureSelectionConfig:
    """Feature selection configuration."""
    correlation_threshold: float = 0.95
    min_importance_percentile: int = 5
    variance_threshold: float = 1e-6
    removal_strategy: str = "conservative"  # aggressive, moderate, conservative


@dataclass
class PathsConfig:
    """Path configuration."""
    feature_registry: str = "config/feature_registry.json"
    models_dir: str = "models"
    results_dir: str = "results"


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: str):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.data = self._parse_data_config(config_dict.get('data', {}))
        self.walkforward = self._parse_walkforward_config(config_dict.get('walkforward', {}))
        self.model = self._parse_model_config(config_dict.get('model', {}))
        self.selection = self._parse_selection_config(config_dict.get('selection', {}))
        self.feature_selection = self._parse_feature_selection_config(
            config_dict.get('feature_selection', {})
        )
        self.paths = self._parse_paths_config(config_dict.get('paths', {}))
        
        self.raw_config = config_dict
    
    @staticmethod
    def _parse_data_config(data_dict: Dict) -> DataConfig:
        return DataConfig(**data_dict)
    
    @staticmethod
    def _parse_walkforward_config(wf_dict: Dict) -> WalkForwardConfig:
        return WalkForwardConfig(**wf_dict)
    
    @staticmethod
    def _parse_model_config(model_dict: Dict) -> ModelConfig:
        # Separate known params from extra params
        known_params = {
            'objective', 'num_leaves', 'learning_rate', 
            'n_estimators', 'random_state'
        }
        known = {k: v for k, v in model_dict.items() if k in known_params}
        extra = {k: v for k, v in model_dict.items() if k not in known_params}
        
        known['extra_params'] = extra
        return ModelConfig(**known)
    
    @staticmethod
    def _parse_selection_config(sel_dict: Dict) -> SelectionConfig:
        return SelectionConfig(**sel_dict)
    
    @staticmethod
    def _parse_feature_selection_config(fs_dict: Dict) -> FeatureSelectionConfig:
        return FeatureSelectionConfig(**fs_dict)
    
    @staticmethod
    def _parse_paths_config(paths_dict: Dict) -> PathsConfig:
        return PathsConfig(**paths_dict)
    
    def get_all_features(self) -> List[str]:
        """Get combined list of all features."""
        return (
            self.data.config_features +
            self.data.market_features +
            self.data.calendar_features
        )
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters as dict for LightGBM."""
        params = {
            'objective': self.model.objective,
            'num_leaves': self.model.num_leaves,
            'learning_rate': self.model.learning_rate,
            'n_estimators': self.model.n_estimators,
            'random_state': self.model.random_state,
        }
        params.update(self.model.extra_params)
        return params
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.raw_config
    
    def save(self, output_path: str):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.raw_config, f, default_flow_style=False)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config object
    """
    return Config(config_path)
