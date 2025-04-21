def simulate_cost_reduction(legacy_cost=10000, efficiency_gain=0.35):
    new_cost = legacy_cost * (1 - efficiency_gain)
    print(f"Legacy System Cost: ${legacy_cost}")
    print(f"Optimized Cloud-Native Cost: ${new_cost:.2f} (-{efficiency_gain*100:.0f}%)")
    return legacy_cost, new_cost
