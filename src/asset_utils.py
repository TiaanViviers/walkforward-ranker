"""Utility functions for asset-specific data paths."""

from pathlib import Path
from typing import Dict


class AssetDataPaths:
    """
    Manages asset-specific data file paths.
    
    Expected directory structure:
        data/
            calibration/
                {asset}_calibration.parquet
            replay_window/
                {asset}_replay_window.parquet
            holdout/
                {asset}_holdout.parquet
    """
    
    def __init__(self, asset: str, data_dir: str = "data"):
        """
        Initialize asset data paths.
        
        Args:
            asset: Asset identifier (e.g., 'us30', 'us500')
            data_dir: Base data directory
        """
        self.asset = asset
        self.data_dir = Path(data_dir)
        
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate that required directories exist."""
        required_dirs = ['calibration', 'replay_window', 'holdout']
        
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                raise ValueError(f"Required directory not found: {dir_path}")
    
    @property
    def calibration(self) -> Path:
        """Path to calibration data (for feature selection)."""
        path = self.data_dir / 'calibration' / f'{self.asset}_calibration.parquet'
        if not path.exists():
            raise FileNotFoundError(f"Calibration data not found: {path}")
        return path
    
    @property
    def replay_window(self) -> Path:
        """Path to replay window data (for walk-forward training)."""
        path = self.data_dir / 'replay_window' / f'{self.asset}_replay_window.parquet'
        if not path.exists():
            raise FileNotFoundError(f"Replay window data not found: {path}")
        return path
    
    @property
    def holdout(self) -> Path:
        """Path to holdout data (for final testing)."""
        path = self.data_dir / 'holdout' / f'{self.asset}_holdout.parquet'
        if not path.exists():
            raise FileNotFoundError(f"Holdout data not found: {path}")
        return path
    
    def get_all_paths(self) -> Dict[str, Path]:
        """Get all data paths as dictionary."""
        return {
            'calibration': self.calibration,
            'replay_window': self.replay_window,
            'holdout': self.holdout
        }
    
    def __repr__(self):
        return f"AssetDataPaths(asset='{self.asset}', data_dir='{self.data_dir}')"


def get_asset_paths(asset: str, data_dir: str = "data") -> AssetDataPaths:
    """
    Get asset-specific data paths.
    
    Args:
        asset: Asset identifier (e.g., 'us30')
        data_dir: Base data directory
        
    Returns:
        AssetDataPaths object
    """
    return AssetDataPaths(asset, data_dir)
