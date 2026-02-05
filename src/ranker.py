"""LightGBM ranking model wrapper."""

import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import contextlib
from typing import Optional, Dict, Any


class LGBMRanker:
    """
    Wrapper for LightGBM LambdaRank model.
    
    Handles grouping and provides consistent interface for ranking.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize ranker.
        
        Args:
            params: LightGBM parameters dict
        """
        self.params = params or self._default_params()
        self.model = None
        self.feature_names = None
    
    @staticmethod
    def _default_params() -> Dict[str, Any]:
        """Get default parameters."""
        return {
            'objective': 'lambdarank',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'random_state': 42,
            'verbosity': -1,
            'force_row_wise': True
        }
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        eval_set: Optional[tuple] = None
    ):
        """
        Train ranking model.
        
        Args:
            X: Feature matrix
            y: Target values (realized PnL)
            groups: Group IDs (date groups)
            eval_set: Optional (X_val, y_val, groups_val) for validation
        """
        self.feature_names = X.columns.tolist()
        
        # Convert groups to group sizes (required by LightGBM)
        group_sizes = groups.value_counts().sort_index().values
        
        # Create model
        self.model = lgb.LGBMRanker(**self.params)
        
        # Prepare evaluation set if provided
        eval_group = None
        if eval_set is not None:
            X_val, y_val, groups_val = eval_set
            eval_group = groups_val.value_counts().sort_index().values
            eval_set = [(X_val, y_val)]
        
        # Train
        callbacks = [lgb.early_stopping(50, verbose=False)] if eval_set else []
        callbacks.append(lgb.log_evaluation(period=0))
        
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self.model.fit(
                    X, y,
                    group=group_sizes,
                    eval_set=eval_set,
                    eval_group=eval_group if eval_set else None,
                    callbacks=callbacks
                )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict scores for ranking.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of scores (higher is better)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance ('gain', 'split', etc.)
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance = self.model.booster_.feature_importance(
            importance_type=importance_type
        )
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        self.model.booster_.save_model(path)
    
    def load_model(self, path: str):
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)


def train_ranker(
    train_df: pd.DataFrame,
    feature_cols: list,
    label_col: str,
    group_col: str,
    params: Dict[str, Any],
    val_df: Optional[pd.DataFrame] = None
) -> LGBMRanker:
    """
    Train a ranking model.
    
    Args:
        train_df: Training data
        feature_cols: List of feature column names
        label_col: Label column name
        group_col: Group column name
        params: Model parameters
        val_df: Optional validation data
        
    Returns:
        Trained LGBMRanker
    """
    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    groups_train = train_df[group_col]
    
    eval_set = None
    if val_df is not None:
        X_val = val_df[feature_cols]
        y_val = val_df[label_col]
        groups_val = val_df[group_col]
        eval_set = (X_val, y_val, groups_val)
    
    ranker = LGBMRanker(params)
    ranker.fit(X_train, y_train, groups_train, eval_set=eval_set)
    
    return ranker
