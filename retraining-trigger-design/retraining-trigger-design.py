def retraining_policy(daily_stats: list, config: dict) -> list:
    budget = config["budget"]
    last_retrain_day = -config["cooldown"]
    days_since_retrain = 0
    result = []

    for x in daily_stats:
        day = x["day"]
        days_since_retrain += 1

        trigger = (
            x["drift_score"] > config["drift_threshold"]
            or x["performance"] < config["performance_threshold"]
            or days_since_retrain >= config["max_staleness"]
        )

        if (trigger
            and day - last_retrain_day >= config["cooldown"]
            and budget >= config["retrain_cost"]):

            result.append(day)
            budget -= config["retrain_cost"]
            last_retrain_day = day
            days_since_retrain = 0

    return result