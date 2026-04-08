"""
seed.py - Populates the database with a realistic A/B/C test scenario.

Run AFTER starting the server:
    python seed.py
"""

import httpx
import sys

BASE = "http://127.0.0.1:8000"


def main():
    print("🌱  Seeding Multi-Armed Bandit API demo data...\n")

    # 1. Create experiment
    r = httpx.post(f"{BASE}/experiments", json={
        "name": "Homepage Hero CTR – May 2024",
        "description": "Comparing three hero-section designs for CTR optimisation.",
        "metric": "ctr",
        "variants": ["control", "variant_a", "variant_b"],
    })
    r.raise_for_status()
    exp = r.json()
    exp_id = exp["id"]
    print(f"✅  Created experiment #{exp_id}: {exp['name']}")

    # 2. Push 14 days of synthetic observations
    #    control ~4%, variant_a ~6.5%, variant_b ~5.2%
    obs = []
    import datetime, random
    random.seed(42)
    start = datetime.date(2024, 5, 1)
    for day_offset in range(14):
        d = (start + datetime.timedelta(days=day_offset)).isoformat()
        for variant, base_ctr in [("control", 0.04), ("variant_a", 0.065), ("variant_b", 0.052)]:
            imp = random.randint(900, 1200)
            clk = int(imp * (base_ctr + random.uniform(-0.005, 0.005)))
            obs.append({"variant_name": variant, "obs_date": d, "impressions": imp, "clicks": clk})

    r = httpx.post(f"{BASE}/experiments/{exp_id}/observations", json={"observations": obs})
    r.raise_for_status()
    print(f"✅  Pushed {r.json()['inserted_or_updated']} observation rows")

    # 3. Get Thompson Sampling recommendation
    r = httpx.get(f"{BASE}/experiments/{exp_id}/allocations?algorithm=thompson_sampling")
    r.raise_for_status()
    data = r.json()
    print(f"\n📊  Thompson Sampling allocation for {data['target_date']}:")
    for a in data["allocations"]:
        bar = "█" * int(a["allocation_pct"] / 2)
        print(f"   {a['variant_name']:12s}  {a['allocation_pct']:5.1f}%  {bar}  "
              f"(CTR={a['smoothed_ctr']*100:.2f}%)")

    # 4. Get UCB1 recommendation
    r = httpx.get(f"{BASE}/experiments/{exp_id}/allocations?algorithm=ucb1")
    r.raise_for_status()
    data = r.json()
    print(f"\n📊  UCB1 allocation for {data['target_date']}:")
    for a in data["allocations"]:
        bar = "█" * int(a["allocation_pct"] / 2)
        print(f"   {a['variant_name']:12s}  {a['allocation_pct']:5.1f}%  {bar}")

    print(f"\n🎉  Done!  Visit http://127.0.0.1:8000/docs for the interactive API docs.")


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print("❌  Could not connect to the server. Start it first with:")
        print("    uvicorn app.main:app --reload")
        sys.exit(1)
