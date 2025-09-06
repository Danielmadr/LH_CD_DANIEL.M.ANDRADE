import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# ========================
# 🎯 Objetivo para Optuna
# ========================
def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 800, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_idx, valid_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx].astype(np.float32), X.iloc[valid_idx].astype(np.float32)
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        pred = model.predict(X_va)
        rmses.append(np.sqrt(mean_squared_error(y_va, pred)))

        trial.report(np.mean(rmses), step=len(rmses))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(rmses))


# ========================
# 🔎 Otimização em 2 fases
# ========================
def run_optuna_two_phase(X, y, n_trials_phase1=80, n_trials_phase2=10):
    print("\n🔎 Fase 1: Busca ampla")
    study1 = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study1.optimize(lambda t: objective(t, X, y), n_trials=n_trials_phase1)

    best_params = study1.best_params
    print("\n🏆 Melhores parâmetros Fase 1:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")
    print(f"✅ Melhor RMSE (CV 5-fold): {study1.best_value:.4f}")

    # -----------------
    # 🔎 Fase 2 (refinamento)
    # -----------------
    def objective_refine(trial, X, y):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", max(500, best_params["n_estimators"] - 200), best_params["n_estimators"] + 200
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", max(0.005, best_params["learning_rate"] * 0.7),
                best_params["learning_rate"] * 1.3, log=True
            ),
            "max_depth": trial.suggest_int(
                "max_depth", max(2, best_params["max_depth"] - 2), min(12, best_params["max_depth"] + 2)
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", max(1, best_params["min_child_weight"] - 2), best_params["min_child_weight"] + 2
            ),
            "subsample": trial.suggest_float(
                "subsample", max(0.5, best_params["subsample"] - 0.1), min(1.0, best_params["subsample"] + 0.1)
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", max(0.5, best_params["colsample_bytree"] - 0.1), min(1.0, best_params["colsample_bytree"] + 0.1)
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", max(1e-3, best_params["reg_lambda"] * 0.5), best_params["reg_lambda"] * 1.5, log=True
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", max(1e-3, best_params["reg_alpha"] * 0.5), best_params["reg_alpha"] * 1.5, log=True
            ),
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses = []

        for train_idx, valid_idx in kf.split(X):
            X_tr, X_va = X.iloc[train_idx].astype(np.float32), X.iloc[valid_idx].astype(np.float32)
            y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            pred = model.predict(X_va)
            rmses.append(np.sqrt(mean_squared_error(y_va, pred)))

            trial.report(np.mean(rmses), step=len(rmses))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(rmses))

    print("\n🔎 Fase 2: Refinamento")
    study2 = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study2.optimize(lambda t: objective_refine(t, X, y), n_trials=n_trials_phase2)

    best_params_final = study2.best_params
    print("\n🏆 Melhores parâmetros finais (Fase 2):")
    for k, v in best_params_final.items():
        print(f"   {k}: {v}")
    print(f"✅ Melhor RMSE (CV 5-fold): {study2.best_value:.4f}")

    # Treinar modelo final
    best_model = XGBRegressor(**best_params_final)
    best_model.fit(X.astype(np.float32), y)

    return best_model
