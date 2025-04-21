from hospital_model import train_hospital_model
from credit_model import train_credit_model
from utils.cost_optimizer import simulate_cost_reduction

if __name__ == "__main__":
    print("🏥 Hospital Readmission Model")
    _, improved_hospital = train_hospital_model()

    print("\n💳 Credit Default Risk Model")
    _, improved_credit = train_credit_model()

    print("\n💰 Operational Cost Comparison")
    simulate_cost_reduction(legacy_cost=10000, efficiency_gain=0.35)
