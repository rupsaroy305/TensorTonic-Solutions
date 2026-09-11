import numpy as np

def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    n = len(production_log)

    production_accuracy = np.mean([
        p["prediction"] == p["actual"]
        for p in production_log
    ])

    shadow_accuracy = np.mean([
        s["prediction"] == s["actual"]
        for s in shadow_log
    ])

    accuracy_gain = shadow_accuracy - production_accuracy

    latencies = np.sort([
        s["latency_ms"]
        for s in shadow_log
    ])

    index = int(np.ceil(0.95 * n)) - 1
    shadow_latency_p95 = latencies[index]

    agreement_rate = np.mean([
        p["prediction"] == s["prediction"]
        for p, s in zip(production_log, shadow_log)
    ])

    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": bool(promote),
        "metrics": {
            "shadow_accuracy": float(shadow_accuracy),
            "production_accuracy": float(production_accuracy),
            "accuracy_gain": float(accuracy_gain),
            "shadow_latency_p95": float(shadow_latency_p95),
            "agreement_rate": float(agreement_rate)
        }
    }