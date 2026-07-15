import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from functools import partial
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, r2_score, mean_squared_error
)
import warnings
import xgboost
import lightgbm
from utilities.usage_profiler import monitor

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# Remove problematic pandas options
if hasattr(pd, 'options') and hasattr(pd.options, 'mode'):
    pd.options.mode.copy_on_write = True  # Enable only if available

@dataclass
class ModelConfig:
    REMOVED_CLASSIFIERS: List[str] = field(default_factory=lambda: [
        "ClassifierChain", "ComplementNB", "GradientBoostingClassifier",
        "GaussianProcessClassifier", "HistGradientBoostingClassifier", "MLPClassifier",
        "LogisticRegressionCV", "MultiOutputClassifier", "MultinomialNB",
        "OneVsOneClassifier", "OneVsRestClassifier", "OutputCodeClassifier",
        "RadiusNeighborsClassifier", "VotingClassifier","LabelPropagation",
        "LabelSpreading" 
    ])
    
    REMOVED_REGRESSORS: List[str] = field(default_factory=lambda: [
        "TheilSenRegressor", "ARDRegression", "CCA", "IsotonicRegression",
        "StackingRegressor", "MultiOutputRegressor", "MultiTaskElasticNet",
        "MultiTaskElasticNetCV", "MultiTaskLasso", "MultiTaskLassoCV",
        "PLSCanonical", "PLSRegression", "RadiusNeighborsRegressor",
        "RegressorChain", "VotingRegressor", "SVR", "NuSVR","QuantileRegressor"
    ])

class ModelRegistry:
    @staticmethod
    def get_models(model_type: str, data_size: int = 0) -> List[Tuple[str, Any]]:
        config = ModelConfig()
        
        # Models that are problematic for large datasets
        large_data_incompatible = [
            "GaussianProcessRegressor", "KernelRidge", "SVR", "NuSVR",
            "GaussianProcessClassifier", "KNeighborsClassifier", "KNeighborsRegressor",
            "RadiusNeighborsClassifier", "RadiusNeighborsRegressor"
        ]
        
        # Add more models to remove for large datasets
        if data_size > 100000:  # If more than 100K rows
            config.REMOVED_CLASSIFIERS.extend([m for m in large_data_incompatible if m not in config.REMOVED_CLASSIFIERS])
            config.REMOVED_REGRESSORS.extend([m for m in large_data_incompatible if m not in config.REMOVED_REGRESSORS])
            
            # Add more models that are inefficient with large datasets
            if data_size > 300000:  # If more than 300K rows
                config.REMOVED_CLASSIFIERS.extend(["SVC", "LinearSVC"])
                config.REMOVED_REGRESSORS.extend(["LinearSVR", "KernelRidge"])
        
        base_models = [
            est for est in all_estimators()
            if (
                issubclass(est[1], ClassifierMixin if model_type == 'classifier' else RegressorMixin) and
                (est[0] not in (config.REMOVED_CLASSIFIERS if model_type == 'classifier' else config.REMOVED_REGRESSORS))
            )
        ]
        
        additional_models = {
            'classifier': [
                ("XGBClassifier", xgboost.XGBClassifier),
                ("LGBMClassifier", lightgbm.LGBMClassifier),
            ],
            'regressor': [
                ("XGBRegressor", xgboost.XGBRegressor),
                ("LGBMRegressor", lightgbm.LGBMRegressor),
            ]
        }
        
        return base_models + additional_models[model_type]

class PreprocessingPipeline:
    @staticmethod
    def create_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
        # Use pandas 2.0 optimized dtypes
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns
        
        # Convert categorical columns to string dtype for better performance
        X_train[categorical_features] = X_train[categorical_features].astype("string")
        
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        
        categorical_transformer_low = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        categorical_transformer_high = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoding", OrdinalEncoder())
        ])
        
        categorical_low, categorical_high = PreprocessingPipeline.get_card_split(X_train, categorical_features)
        
        return ColumnTransformer([
            ("numeric", numeric_transformer, numeric_features),
            ("categorical_low", categorical_transformer_low, categorical_low),
            ("categorical_high", categorical_transformer_high, categorical_high)
        ])
    
    @staticmethod
    def get_card_split(df: pd.DataFrame, cols: pd.Index, n: int = 11) -> Tuple[pd.Index, pd.Index]:
        # Use pandas 2.0 optimized nunique()
        nunique = df[cols].agg("nunique")
        cond = nunique > n
        return cols[~cond], cols[cond]


class ModelEvaluator:
    @staticmethod
    def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, model) -> Dict[str, float]:
        """
        Evaluate classifier performance with proper ROC-AUC handling
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Handle ROC-AUC calculation
        n_classes = len(np.unique(y_true))
        try:
            if n_classes == 2:
                # For binary classification
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            else:
                # For multiclass
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                else:
                    metrics['roc_auc'] = None
        except Exception:
            metrics['roc_auc'] = None
            
        return metrics
    
    @staticmethod
    def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray, n_samples: int, n_features: int) -> Dict[str, float]:
        r2 = r2_score(y_true, y_pred)
        return {
            'r2': r2,
            'adj_r2': 1 - (1 - r2) * ((n_samples - 1) / (n_samples - n_features - 1)),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }


def _evaluate_model_parallel(args: Tuple) -> Optional[Dict[str, Any]]:
    """Helper function for parallel model evaluation with memory protection"""
    name, model, preprocessor, X_train, X_test, y_train, y_test, random_state, ignore_warnings = args
    
    # Skip models that are known to be memory-intensive for large datasets
    memory_intensive_models = [
        "GaussianProcessRegressor", "KernelRidge", "SVR", "NuSVR", 
        "GaussianProcessClassifier", "KNeighborsClassifier", "KNeighborsRegressor"
    ]
    
    # Skip these models for large datasets
    if name in memory_intensive_models and (X_train.shape[0] > 50000 or X_train.shape[1] > 100):
        return None
        
    try:
        start = time.time()
        model_params = {}
        
        # Configure models for efficiency with large datasets
        if hasattr(model, 'get_params'):
            params = model().get_params()
            
            if 'random_state' in params:
                model_params['random_state'] = random_state
                
            # Optimize hyperparameters for large datasets
            if X_train.shape[0] > 100000:
                # For tree-based models, limit depth and estimators
                if 'max_depth' in params:
                    model_params['max_depth'] = 8  # Limit tree depth
                if 'n_estimators' in params:
                    model_params['n_estimators'] = min(100, params.get('n_estimators', 100))
                    
                # For LightGBM/XGBoost add efficient parameters
                if name in ['LGBMRegressor', 'LGBMClassifier']:
                    model_params['objective'] = 'regression' if 'Regressor' in name else 'binary'
                    model_params['verbose'] = -1
                    model_params['feature_fraction'] = 0.8
                    model_params['subsample'] = 0.8
                    
                # Avoid complex kernels for SVM models
                if 'kernel' in params and X_train.shape[0] > 50000:
                    model_params['kernel'] = 'linear'
            
        # Build and time the pipeline
        try:
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('estimator', model(**model_params))
            ])
            
            # Set up a timeout for fitting to avoid indefinite hangs
            fit_start_time = time.time()
            pipe.fit(X_train, y_train)
            
            # If fitting takes too long, consider it a failure
            if time.time() - fit_start_time > 600:  # 10 minutes timeout
                print(f"Model {name} took too long to fit. Skipping.")
                return None
                
            y_pred = pipe.predict(X_test)
        except MemoryError:
            print(f"Memory error in {name}. Skipping.")
            return None
        except Exception as e:
            if not ignore_warnings:
                print(f"Error in {name}: {str(e)}")
            return None
        
        result = {'Model': name, 'Time Taken': time.time() - start}
        
        # Determine if it's a classifier based on the model type
        if isinstance(model(), ClassifierMixin):
            # Handle ROC-AUC calculation
            n_classes = len(np.unique(y_test))
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            try:
                if n_classes == 2:
                    if hasattr(pipe, 'predict_proba'):
                        y_prob = pipe.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred)
                else:
                    if hasattr(pipe, 'predict_proba'):
                        y_prob = pipe.predict_proba(X_test)
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                    else:
                        metrics['roc_auc'] = None
            except Exception:
                metrics['roc_auc'] = None
                
            result.update(metrics)
        else:
            r2 = r2_score(y_test, y_pred)
            result.update({
                'r2': r2,
                'adj_r2': 1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            })
            
        return result, pipe
    
    except MemoryError:
        if not ignore_warnings:
            print(f"Memory error in {name}. Skipping.")
        return None
    except Exception as e:
        if not ignore_warnings:
            print(f"Error in {name}: {str(e)}")
        return None
    
    except Exception as e:
        if not ignore_warnings:
            print(f"Error in {name}: {str(e)}")
        return None


class root2:
    def __init__(
        self,
        estimator_type: str,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        estimators: str = "all",
        n_jobs: int = -1,
        max_memory_size: float = 0.8  # Default to using 80% of available memory
    ):
        self.estimator_type = estimator_type
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.random_state = random_state
        self.estimators = estimators
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, cpu_count() // 2)  # Use half of CPUs by default for large datasets
        self.max_memory_size = max_memory_size
        self.models = {}
    
    def fit(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        
        # Determine dataset size for model selection
        data_size = X_train.shape[0]
        
        # Dynamically adjust workers based on dataset size
        if data_size > 100000:
            self.n_jobs = max(1, min(self.n_jobs, 4))  # Limit workers for large datasets
        
        # Set up chunking for large datasets
        chunk_size = min(10000, data_size // 10) if data_size > 50000 else data_size
        
        preprocessor = PreprocessingPipeline.create_preprocessor(X_train)
        results = []
        predictions = {}
        
        models = (ModelRegistry.get_models('classifier', data_size) if self.estimator_type == 'classifier' 
                 else ModelRegistry.get_models('regressor', data_size)) if self.estimators == "all" else self._get_custom_estimators()
        
        # Add a progress message if verbose
        if self.verbose:
            print(f"Starting model evaluation with {len(models)} models on dataset of size {data_size}")
        
        # Updated to include ignore_warnings in eval_args
        eval_args = [
            (name, model, preprocessor, X_train, X_test, y_train, y_test, self.random_state, self.ignore_warnings)
            for name, model in models
        ]
        
        # Use ProcessPoolExecutor with reduced workers for large datasets
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Add tqdm only if verbose
            if self.verbose > 0:
                futures = list(tqdm(executor.map(_evaluate_model_parallel, eval_args), total=len(eval_args)))
            else:
                futures = list(executor.map(_evaluate_model_parallel, eval_args))
            
            for result in futures:
                if result is not None:
                    model_result, model_pipe = result
                    results.append(model_result)
                    self.models[model_result['Model']] = model_pipe
                    
                    if self.predictions:
                        predictions[model_result['Model']] = model_pipe.predict(X_test)
        
        scores = pd.DataFrame(results)
        if not scores.empty:
            scores = scores.sort_values(
                by='balanced_accuracy' if self.estimator_type == 'classifier' else 'adj_r2',
                ascending=False
            ).set_index('Model')
        
        if self.predictions:
            predictions_df = pd.DataFrame(predictions)
            return scores, predictions_df
        return scores, None
        
    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes using pandas 2.0 features"""
        # Convert numeric columns to appropriate types
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 2:
                df[col] = df[col].astype('boolean')
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')  # Use float32 instead of float64
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')  # Use int32 instead of int64
        
        # Convert categorical columns to string type
        for col in df.select_dtypes(exclude=[np.number]).columns:
            df[col] = df[col].astype('string')
        
        return df
    
    def _get_custom_estimators(self) -> List[Tuple[str, Any]]:
        try:
            return [(est.__name__, est) for est in self.estimators]
        except Exception as e:
            raise ValueError(f"Invalid estimators provided: {str(e)}")
    
    def provide_models(self, X_train, X_test, y_train, y_test):
        if not self.models:
            self.fit(X_train, X_test, y_train, y_test)
        return self.models



# Create alias classes for backward compatibility
JarvisPredict = partial(root2, estimator_type='regressor')
JarvisClassify = partial(root2, estimator_type='classifier')