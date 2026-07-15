def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    if total_steps == 0:
        return float(final_lr)

    if warmup_steps > 0 and step < warmup_steps:
        return float(step * initial_lr / warmup_steps)

    if step <= total_steps:
        return float(final_lr + (initial_lr - final_lr) * (total_steps - step) / (total_steps - warmup_steps))

    return float(final_lr)