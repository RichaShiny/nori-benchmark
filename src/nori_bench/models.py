"""Model registry: sklearn baselines plus Nori V1.

Nori uses in-context learning: fit() only stores the labeled context,
predict() runs a single forward pass. No training happens.
"""
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def get_models(device: str = "cpu", skip_nori: bool = False,
               nori_model: str = "nori-30m") -> dict:
    """Return {name: unfitted estimator}. All follow the sklearn API."""
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }
    if not skip_nori:
        from synthefy_nori import NoriRegressor

        # Always name a size: "nori-30m" (~29.2M, stronger) or "nori-6m"
        # (~6M base). Omitting model= silently falls back to the 6M base.
        # "nori-30m-thinking" is hosted-API only, no local checkpoint.
        # Weights download from Hugging Face on first use and are cached.
        models["NoriV1"] = NoriRegressor(model=nori_model, device=device)
    return models
