"""
Hyperparameter Tuning with Optuna

Provides Bayesian optimization for LightGBM ranking model hyperparameters.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime

from src.ranker import train_ranker
from src.evaluator import evaluate_predictions
from src.selector import add_rankings


class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuning for LightGBM ranking.
    
    Uses simple train/validation split for fast optimization.
    """
    
    def __init__(
        self,
        feature_cols: list,
        label_col: str,
        group_col: str,
        pnl_col: str,
        n_trials: int = 50,
        timeout_minutes: Optional[int] = 30,
        study_name: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            feature_cols: List of feature column names
            label_col: Label column name (relevance_grade)
            group_col: Grouping column name (group_id)
            pnl_col: PnL column for evaluation
            n_trials: Number of Optuna trials to run
            timeout_minutes: Maximum time for tuning (minutes)
            study_name: Optional study name for tracking
            verbose: Print trial progress
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.group_col = group_col
        self.pnl_col = pnl_col
        self.n_trials = n_trials
        self.timeout_seconds = timeout_minutes * 60 if timeout_minutes else None
        self.study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.verbose = verbose
        
        # Will be set during tuning
        self.train_df = None
        self.val_df = None
        self.fixed_params = {}
    
    def tune(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        validation_days: int = 50,
        search_space: Optional[Dict[str, Dict]] = None,
        fixed_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], optuna.Study]:
        """
        Run hyperparameter tuning on given data.
        
        Args:
            df: Training data (typically last 252 days before quarter start)
            date_col: Date column name
            validation_days: Number of days to use for validation
            search_space: Custom search space (uses defaults if None)
            fixed_params: Parameters to keep fixed (not tuned)
            
        Returns:
            Tuple of (best_params, study)
        """
        # Split into train/validation
        unique_dates = df[date_col].unique()
        unique_dates = np.sort(unique_dates)
        
        # Adaptive validation size: use specified days or 30% of data, whichever is smaller
        min_total_days = 70  # Minimum total days needed
        if len(unique_dates) < min_total_days:
            raise ValueError(
                f"Not enough data for tuning. Need at least {min_total_days} days, "
                f"got {len(unique_dates)}"
            )
        
        # Use smaller of: requested validation_days or 30% of data
        adaptive_val_days = min(validation_days, int(len(unique_dates) * 0.3))
        val_start_date = unique_dates[-adaptive_val_days]
        
        self.train_df = df[df[date_col] < val_start_date].copy()
        self.val_df = df[df[date_col] >= val_start_date].copy()
        self.fixed_params = fixed_params or {}
        
        if self.verbose:
            print(f"Tuning: {self.train_df[date_col].nunique()}d train, {self.val_df[date_col].nunique()}d val, {self.n_trials} trials...", end='', flush=True)
        
        # Use custom or default search space
        self.search_space = search_space or self._default_search_space()
        
        # Create Optuna study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        # Run optimization (disable progress bar to keep output clean)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            show_progress_bar=False
        )
        
        # Print concise results
        if self.verbose:
            print(f" NDCG={study.best_value:.4f}")
        
        # Combine best params with fixed params
        best_params = {**self.fixed_params, **study.best_params}
        
        return best_params, study
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            NDCG@3 score on validation set
        """
        # Sample hyperparameters from search space
        params = {}
        
        for param_name, param_config in self.search_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        # Add fixed params
        params.update(self.fixed_params)
        
        # Train model with these params
        try:
            model = train_ranker(
                self.train_df,
                self.feature_cols,
                self.label_col,
                self.group_col,
                params
            )
            
            # Predict on validation set
            val_df_copy = self.val_df.copy()
            val_df_copy['predicted_score'] = model.predict(val_df_copy[self.feature_cols])
            
            # Add rankings
            val_df_copy = add_rankings(
                val_df_copy,
                self.group_col,
                'predicted_score',
                self.pnl_col
            )
            
            # Evaluate
            metrics = evaluate_predictions(
                val_df_copy,
                k=3,  # Use top-3 for tuning
                group_col=self.group_col,
                label_col=self.label_col,
                pnl_col=self.pnl_col
            )
            
            # Return NDCG@3 (Optuna maximizes this)
            return metrics['ndcg_3']
        
        except Exception as e:
            if self.verbose:
                print(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return worst score on failure
    
    def _default_search_space(self) -> Dict[str, Dict]:
        """
        Default search space for LightGBM ranking.
        
        These ranges are based on:
        - LightGBM documentation
        - Kaggle competition best practices
        - Validated for ranking tasks
        """
        return {
            'num_leaves': {
                'type': 'int',
                'low': 31,
                'high': 255
            },
            'learning_rate': {
                'type': 'float',
                'low': 0.01,
                'high': 0.15,
                'log': True  # Sample on log scale (more trials at lower values)
            },
            'n_estimators': {
                'type': 'int',
                'low': 100,
                'high': 500
            },
            'min_child_samples': {
                'type': 'int',
                'low': 10,
                'high': 50
            },
            'subsample': {
                'type': 'float',
                'low': 0.7,
                'high': 1.0
            },
            'colsample_bytree': {
                'type': 'float',
                'low': 0.7,
                'high': 1.0
            },
            'reg_lambda': {
                'type': 'float',
                'low': 0.0,
                'high': 10.0
            }
        }
    
    @staticmethod
    def save_tuning_results(
        best_params: Dict[str, Any],
        study: optuna.Study,
        output_path: str
    ):
        """
        Save tuning results to JSON file.
        
        Args:
            best_params: Best parameters found
            study: Optuna study object
            output_path: Path to save results
        """
        results = {
            'best_params': best_params,
            'best_score': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'study_name': study.study_name,
            'timestamp': datetime.now().isoformat(),
            'trial_history': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    @staticmethod
    def load_best_params(path: str) -> Dict[str, Any]:
        """
        Load best parameters from saved results.
        
        Args:
            path: Path to tuning results JSON
            
        Returns:
            Best parameters dictionary
        """
        with open(path, 'r') as f:
            results = json.load(f)
        return results['best_params']
