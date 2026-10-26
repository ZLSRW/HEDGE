from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def _parse_models(superv_models: str) -> List[str]:
    if superv_models is None:
        return ['et', 'rf', 'svc', 'gb']
    toks = [t.strip().lower() for t in superv_models.split(',') if t.strip()]
    return toks or ['et', 'rf', 'svc', 'gb']

def _build_estimator(key: str, *, nthreads: int=4, random_state: int=42, class_weight: Optional[str]='balanced') -> Tuple[str, BaseEstimator]:
    key = key.lower()
    if key in ('et', 'extratrees'):
        est = ExtraTreesClassifier(n_estimators=500, random_state=random_state, n_jobs=nthreads, class_weight=class_weight if class_weight and class_weight != 'none' else None)
        return ('ExtraTreesClassifier', est)
    if key in ('rf', 'randomforest'):
        est = RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=nthreads, class_weight=class_weight if class_weight and class_weight != 'none' else None)
        return ('RandomForestClassifier', est)
    if key in ('gb', 'gboost', 'gradientboosting'):
        est = GradientBoostingClassifier(random_state=random_state)
        return ('GradientBoostingClassifier', est)
    if key in ('svc', 'svm'):
        est = SVC(probability=True, random_state=random_state, class_weight=class_weight if class_weight and class_weight != 'none' else None)
        return ('SVC', est)
    if key in ('lr', 'logreg', 'logistic'):
        est = LogisticRegression(max_iter=2000, solver='liblinear', class_weight=class_weight if class_weight and class_weight != 'none' else None, random_state=random_state)
        return ('LogisticRegression', est)
    if key in ('xgb', 'xgboost'):
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError('xgboost 未安装，但 superv_models 中包含 xgb。请安装 xgboost 或从 superv_models 移除 xgb。') from e
        est = XGBClassifier(n_estimators=800, max_depth=5, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=nthreads, random_state=random_state, eval_metric='logloss')
        return ('XGBoost', est)
    raise ValueError(f'未知模型短码: {key} (支持: et, rf, svc, gb, lr, xgb, stack)')

class MantisML:

    def __init__(self, *, superv_models: str='et,rf,svc,gb', mode: str='soft_vote', nthreads: int=4, random_state: int=42, class_weight: str='balanced'):
        self.superv_models = superv_models
        self.mode = str(mode).lower()
        self.nthreads = int(nthreads)
        self.random_state = int(random_state)
        self.class_weight = str(class_weight).lower() if class_weight is not None else 'balanced'
        self.model_keys: List[str] = _parse_models(superv_models)
        self.base_models: List[Tuple[str, BaseEstimator]] = []
        self.stacker: Optional[BaseEstimator] = None
        self.fitted_: bool = False
        self._build()

    def _build(self):
        keys = [k for k in self.model_keys if k != 'stack']
        self.base_models = [_build_estimator(k, nthreads=self.nthreads, random_state=self.random_state, class_weight=None if self.class_weight == 'none' else self.class_weight) for k in keys]
        if self.mode == 'stack' or 'stack' in self.model_keys:
            final_est = LogisticRegression(max_iter=2000, solver='liblinear', class_weight=None if self.class_weight == 'none' else self.class_weight, random_state=self.random_state)
            self.stacker = StackingClassifier(estimators=self.base_models, final_estimator=final_est, stack_method='predict_proba', passthrough=False, n_jobs=self.nthreads)
        else:
            self.stacker = None

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f'X must be 2D array, got shape={X.shape}')
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MantisML':
        X = self._ensure_2d(X)
        y = np.asarray(y).astype(int)
        if self.stacker is not None:
            self.stacker.fit(X, y)
        else:
            for _, m in self.base_models:
                m.fit(X, y)
        self.fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError('Model not fitted. Call fit() first.')
        X = self._ensure_2d(X)
        if self.stacker is not None:
            return self.stacker.predict_proba(X)
        probs = []
        for _, m in self.base_models:
            probs.append(m.predict_proba(X))
        return np.mean(probs, axis=0)

    def compute_repr(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        if not self.fitted_:
            raise RuntimeError('Model not fitted. Call fit() first.')
        X = self._ensure_2d(X)
        feats = []
        names: List[str] = []
        if self.stacker is not None:
            fitted_estimators = getattr(self.stacker, 'estimators_', None)
            if fitted_estimators is not None:
                for (name, _), est in zip(self.base_models, fitted_estimators):
                    try:
                        p = est.predict_proba(X)[:, 1:2]
                        feats.append(p)
                        names.append(f'{name}_p1')
                    except Exception:
                        pass
            p_stack = self.stacker.predict_proba(X)[:, 1:2]
            feats.append(p_stack)
            names.append('Stacking_p1')
        else:
            for name, m in self.base_models:
                p = m.predict_proba(X)[:, 1:2]
                feats.append(p)
                names.append(f'{name}_p1')
        Z = np.concatenate(feats, axis=1) if feats else np.zeros((X.shape[0], 0), dtype=float)
        return (Z, names)
MantisML = MantisML
