"""Two explicitly registered, small CPU networks; no synthetic oracle here."""
from collections import OrderedDict
import io
import math

from ..persistence import function_db

FEATURES = ("historical_mean_delay_days", "historical_delay_std_days",
            "credit_utilization_ratio", "active_dispute_flag")
SCALES = (60., 30., 1., 1.)
REGRESSOR = "ar_delay_regressor_v1"
CLASSIFIER = "ar_severe_overdue_classifier_v1"
LINEAR = "ar_linear_delay_v1"
FUNCTIONS = (REGRESSOR, CLASSIFIER, LINEAR)
_CACHE = OrderedDict()


def build_model(function_id):
    import torch.nn as nn
    if function_id == REGRESSOR:
        return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    if function_id == CLASSIFIER:
        return nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, 2))
    if function_id == LINEAR:
        return nn.Sequential(nn.Linear(4, 1))
    raise ValueError(f"unregistered function: {function_id}")


def encode_inputs(inputs):
    if set(inputs) != set(FEATURES):
        raise ValueError("model inputs must contain exactly the four declared features")
    values = []
    for name, scale in zip(FEATURES, SCALES):
        item = inputs[name]
        unit = item.get("unit", "")
        if unit not in (("day", "days") if scale != 1 else ("", "1", "ratio")):
            raise ValueError(f"invalid unit for {name}: {unit}")
        value = item["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid numeric feature {name}")
        if name == FEATURES[-1] and value not in (0, 1):
            raise ValueError("dispute flag must be 0 or 1")
        values.append(value / scale)
    return values


def serialize(model):
    import torch
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def infer(db_path, inputs, request):
    import torch
    function_id = request.get("function_id")
    if function_id not in FUNCTIONS or str(request.get("function_version")) != "1":
        raise ValueError("unregistered function implementation/version")
    values = encode_inputs(inputs)
    weights, metadata, digest = function_db.load(db_path, function_id, request.get("parameter_version"))
    if metadata.get("feature_version") != "ar_features_v1" or metadata.get("function_version") != "1":
        raise ValueError("incompatible model contract")
    key = (function_id, digest)
    model = _CACHE.get(key)
    if model is None:
        # Loading a model must not advance the application's training RNG.
        with torch.random.fork_rng():
            model = build_model(function_id)
        model.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True))
        model.eval()
        _CACHE[key] = model
        if len(_CACHE) > 8:
            _CACHE.popitem(last=False)
    _CACHE.move_to_end(key)
    with torch.inference_mode():
        raw = model(torch.tensor([values], dtype=torch.float32))[0]
        if function_id != CLASSIFIER:
            outputs = {"predicted_delay_days": {"value": float(raw[0]) * 60, "unit": "day"}}
        else:
            probability = torch.softmax(raw, dim=0)
            outputs = {"class_id": {"value": int(probability.argmax()), "unit": ""},
                       "severe_overdue_probability": {"value": float(probability[1]), "unit": ""}}
    return {"outputs": outputs, "artifact_sha256": digest, "feature_version": "ar_features_v1",
            "applicability": {"within_training_range": all(0 <= v <= 1 for v in values)}}
