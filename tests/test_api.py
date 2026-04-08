"""
tests/test_api.py – Suite completa de testes para o Multi-Armed Bandit API.

Cobertura:
  - Testes unitários do algoritmo (Thompson Sampling e UCB1)
  - Testes de integração via FastAPI TestClient
  - Validação de schemas Pydantic
  - Casos de borda (sem dados, variante inválida, clicks > impressions)

Execute com:
    pytest tests/ -v
"""

import pytest
from app.bandit import thompson_sampling, ucb1, compute_allocations


# =============================================================================
# Helpers
# =============================================================================

def make_arms(ctrs_and_impressions):
    return [
        {
            "variant_id": i,
            "variant_name": f"arm_{i}",
            "is_control": i == 0,
            "total_impressions": imp,
            "total_clicks": int(imp * ctr),
            "smoothed_ctr": ctr,
        }
        for i, (ctr, imp) in enumerate(ctrs_and_impressions)
    ]


# =============================================================================
# 1. Testes Unitários – Thompson Sampling
# =============================================================================

class TestThompsonSampling:

    def test_allocations_sum_to_100(self):
        arms = make_arms([(0.05, 1000), (0.10, 1000)])
        result = thompson_sampling(arms, seed=42)
        assert abs(sum(result) - 100.0) < 0.5

    def test_higher_ctr_gets_more_traffic(self):
        arms = make_arms([(0.02, 2000), (0.15, 2000)])
        result = thompson_sampling(arms, seed=7)
        assert result[1] > result[0]

    def test_three_arms_multivariate(self):
        """Suporte a 3+ braços (requisito bônus)."""
        arms = make_arms([(0.04, 1000), (0.07, 1000), (0.12, 1000)])
        result = thompson_sampling(arms, seed=99)
        assert len(result) == 3
        assert abs(sum(result) - 100.0) < 0.5
        assert result[2] > result[0]

    def test_equal_ctrs_roughly_equal_allocation(self):
        arms = make_arms([(0.10, 5000), (0.10, 5000)])
        result = thompson_sampling(arms, n_samples=100_000, seed=1)
        assert abs(result[0] - result[1]) < 8.0

    def test_zero_impressions_still_runs(self):
        arms = make_arms([(0.0, 0), (0.0, 0)])
        result = thompson_sampling(arms, seed=5)
        assert abs(sum(result) - 100.0) < 0.5

    def test_seed_reproducibility(self):
        arms = make_arms([(0.05, 1000), (0.10, 1000)])
        r1 = thompson_sampling(arms, seed=123)
        r2 = thompson_sampling(arms, seed=123)
        assert r1 == r2

    def test_single_dominant_arm(self):
        arms = make_arms([(0.01, 10000), (0.99, 10000)])
        result = thompson_sampling(arms, seed=42)
        assert result[1] > 90.0


# =============================================================================
# 2. Testes Unitários – UCB1
# =============================================================================

class TestUCB1:

    def test_allocations_sum_to_100(self):
        arms = make_arms([(0.05, 1000), (0.10, 1000)])
        result = ucb1(arms)
        assert abs(sum(result) - 100.0) < 0.5

    def test_three_arms(self):
        arms = make_arms([(0.04, 500), (0.07, 500), (0.12, 500)])
        result = ucb1(arms)
        assert len(result) == 3
        assert abs(sum(result) - 100.0) < 0.5

    def test_zero_impressions_equal_split(self):
        arms = make_arms([(0.0, 0), (0.0, 0)])
        result = ucb1(arms)
        assert abs(result[0] - result[1]) < 0.01

    def test_unseen_arm_gets_exploration_bonus(self):
        arms = [
            {"total_impressions": 1000, "total_clicks": 50,  "variant_name": "seen"},
            {"total_impressions": 0,    "total_clicks": 0,   "variant_name": "unseen"},
        ]
        result = ucb1(arms)
        assert result[1] > result[0]


# =============================================================================
# 3. Testes Unitários – compute_allocations dispatcher
# =============================================================================

class TestComputeAllocations:

    def test_dispatches_thompson(self):
        arms = make_arms([(0.05, 1000), (0.10, 1000)])
        result = compute_allocations(arms, algorithm="thompson_sampling")
        assert abs(sum(result) - 100.0) < 0.5

    def test_dispatches_ucb1(self):
        arms = make_arms([(0.05, 1000), (0.10, 1000)])
        result = compute_allocations(arms, algorithm="ucb1")
        assert abs(sum(result) - 100.0) < 0.5

    def test_default_is_thompson(self):
        arms = make_arms([(0.05, 1000), (0.10, 1000)])
        result = compute_allocations(arms)
        assert abs(sum(result) - 100.0) < 0.5


# =============================================================================
# 4. Testes de Integração – API REST
# =============================================================================

class TestHealth:
    def test_health_check(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestExperiments:

    def test_create_experiment_returns_201(self, client):
        r = client.post("/experiments", json={
            "name": "E2E Test Experiment",
            "description": "Criado pelo test suite",
            "metric": "ctr",
            "variants": ["control", "variant_a"],
        })
        assert r.status_code == 201
        assert "id" in r.json()

    def test_create_multivariant_experiment(self, client):
        r = client.post("/experiments", json={
            "name": "Multi-Arm Experiment",
            "variants": ["control", "variant_a", "variant_b", "variant_c"],
        })
        assert r.status_code == 201

    def test_create_duplicate_name_fails(self, client):
        payload = {"name": "Duplicate Name Test", "variants": ["control", "v_a"]}
        client.post("/experiments", json=payload)
        r2 = client.post("/experiments", json=payload)
        assert r2.status_code == 400

    def test_create_single_variant_fails(self, client):
        r = client.post("/experiments", json={
            "name": "Only One Variant",
            "variants": ["control"],
        })
        assert r.status_code == 422

    def test_list_experiments(self, client):
        r = client.get("/experiments")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_experiment_by_id(self, client):
        created = client.post("/experiments", json={
            "name": "Get By ID Test",
            "variants": ["control", "v_b"],
        }).json()
        r = client.get(f"/experiments/{created['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == created["id"]

    def test_get_nonexistent_returns_404(self, client):
        r = client.get("/experiments/99999")
        assert r.status_code == 404


class TestObservations:

    @pytest.fixture(scope="class")
    def exp_id(self, client):
        r = client.post("/experiments", json={
            "name": "Observations Test Exp",
            "variants": ["control", "variant_a", "variant_b"],
        })
        return r.json()["id"]

    def test_push_observations_returns_201(self, client, exp_id):
        payload = {"observations": [
            {"variant_name": "control",   "obs_date": "2024-06-01", "impressions": 1000, "clicks": 42},
            {"variant_name": "variant_a", "obs_date": "2024-06-01", "impressions": 1000, "clicks": 68},
            {"variant_name": "variant_b", "obs_date": "2024-06-01", "impressions": 1000, "clicks": 55},
        ]}
        r = client.post(f"/experiments/{exp_id}/observations", json=payload)
        assert r.status_code == 201
        assert r.json()["inserted_or_updated"] == 3

    def test_push_multiple_days(self, client, exp_id):
        payload = {"observations": [
            {"variant_name": "control",   "obs_date": "2024-06-02", "impressions": 1100, "clicks": 45},
            {"variant_name": "variant_a", "obs_date": "2024-06-02", "impressions": 1100, "clicks": 75},
        ]}
        r = client.post(f"/experiments/{exp_id}/observations", json=payload)
        assert r.status_code == 201

    def test_upsert_same_date_does_not_fail(self, client, exp_id):
        payload = {"observations": [
            {"variant_name": "control", "obs_date": "2024-06-01", "impressions": 1050, "clicks": 44},
        ]}
        r = client.post(f"/experiments/{exp_id}/observations", json=payload)
        assert r.status_code == 201

    def test_clicks_exceeding_impressions_rejected(self, client, exp_id):
        payload = {"observations": [
            {"variant_name": "control", "obs_date": "2024-06-10", "impressions": 100, "clicks": 200},
        ]}
        r = client.post(f"/experiments/{exp_id}/observations", json=payload)
        assert r.status_code == 422

    def test_invalid_variant_returns_400(self, client, exp_id):
        payload = {"observations": [
            {"variant_name": "ghost_arm", "obs_date": "2024-06-03", "impressions": 100, "clicks": 5},
        ]}
        r = client.post(f"/experiments/{exp_id}/observations", json=payload)
        assert r.status_code == 400

    def test_nonexistent_experiment_returns_404(self, client):
        payload = {"observations": [
            {"variant_name": "control", "obs_date": "2024-06-01", "impressions": 100, "clicks": 5},
        ]}
        r = client.post("/experiments/99999/observations", json=payload)
        assert r.status_code == 404


class TestAllocations:

    @pytest.fixture(scope="class")
    def exp_id_with_data(self, client):
        r = client.post("/experiments", json={
            "name": "Allocation Test Exp",
            "variants": ["control", "variant_a", "variant_b"],
        })
        exp_id = r.json()["id"]

        import random
        from datetime import date, timedelta
        random.seed(42)
        obs = []
        start = date(2024, 5, 1)
        for day in range(14):
            d = (start + timedelta(days=day)).isoformat()
            for name, base_ctr in [("control", 0.04), ("variant_a", 0.065), ("variant_b", 0.052)]:
                imp = random.randint(900, 1200)
                clk = int(imp * (base_ctr + random.uniform(-0.005, 0.005)))
                obs.append({"variant_name": name, "obs_date": d, "impressions": imp, "clicks": clk})

        client.post(f"/experiments/{exp_id}/observations", json={"observations": obs})
        return exp_id

    def test_thompson_allocation_returns_200(self, client, exp_id_with_data):
        r = client.get(f"/experiments/{exp_id_with_data}/allocations?algorithm=thompson_sampling")
        assert r.status_code == 200
        data = r.json()
        assert data["algorithm"] == "thompson_sampling"
        assert len(data["allocations"]) == 3
        total = sum(a["allocation_pct"] for a in data["allocations"])
        assert abs(total - 100.0) < 1.0

    def test_best_variant_gets_most_traffic(self, client, exp_id_with_data):
        r = client.get(f"/experiments/{exp_id_with_data}/allocations?algorithm=thompson_sampling")
        by_name = {a["variant_name"]: a for a in r.json()["allocations"]}
        assert by_name["variant_a"]["allocation_pct"] > by_name["control"]["allocation_pct"]

    def test_ucb1_allocation_returns_200(self, client, exp_id_with_data):
        r = client.get(f"/experiments/{exp_id_with_data}/allocations?algorithm=ucb1")
        assert r.status_code == 200
        total = sum(a["allocation_pct"] for a in r.json()["allocations"])
        assert abs(total - 100.0) < 1.0

    def test_custom_target_date(self, client, exp_id_with_data):
        r = client.get(f"/experiments/{exp_id_with_data}/allocations?target_date=2025-01-01")
        assert r.status_code == 200
        assert r.json()["target_date"] == "2025-01-01"

    def test_invalid_algorithm_returns_400(self, client, exp_id_with_data):
        r = client.get(f"/experiments/{exp_id_with_data}/allocations?algorithm=random_bandit")
        assert r.status_code == 400

    def test_nonexistent_experiment_returns_404(self, client):
        r = client.get("/experiments/99999/allocations")
        assert r.status_code == 404

    def test_no_data_returns_equal_split_with_note(self, client):
        r = client.post("/experiments", json={
            "name": "Empty Exp For Allocation",
            "variants": ["control", "variant_a"],
        })
        exp_id = r.json()["id"]
        r2 = client.get(f"/experiments/{exp_id}/allocations")
        assert r2.status_code == 200
        data = r2.json()
        assert data["note"] is not None
        total = sum(a["allocation_pct"] for a in data["allocations"])
        assert abs(total - 100.0) < 1.0


class TestAnalytics:

    @pytest.fixture(scope="class")
    def exp_id_with_series(self, client):
        r = client.post("/experiments", json={
            "name": "Analytics Test Exp",
            "variants": ["control", "variant_a"],
        })
        exp_id = r.json()["id"]
        client.post(f"/experiments/{exp_id}/observations", json={"observations": [
            {"variant_name": "control",   "obs_date": "2024-07-01", "impressions": 500, "clicks": 20},
            {"variant_name": "variant_a", "obs_date": "2024-07-01", "impressions": 500, "clicks": 35},
            {"variant_name": "control",   "obs_date": "2024-07-02", "impressions": 600, "clicks": 22},
            {"variant_name": "variant_a", "obs_date": "2024-07-02", "impressions": 600, "clicks": 40},
        ]})
        return exp_id

    def test_analytics_returns_series(self, client, exp_id_with_series):
        r = client.get(f"/experiments/{exp_id_with_series}/analytics")
        assert r.status_code == 200
        data = r.json()
        assert "series" in data
        assert len(data["series"]) == 4

    def test_ctr_values_are_correct(self, client, exp_id_with_series):
        series = client.get(f"/experiments/{exp_id_with_series}/analytics").json()["series"]
        for point in series:
            if point["impressions"] > 0:
                expected = point["clicks"] / point["impressions"]
                assert abs(point["daily_ctr"] - expected) < 0.001

    def test_observations_endpoint_works(self, client, exp_id_with_series):
        r = client.get(f"/experiments/{exp_id_with_series}/observations")
        assert r.status_code == 200
        assert "series" in r.json()

    def test_analytics_nonexistent_experiment(self, client):
        r = client.get("/experiments/99999/analytics")
        assert r.status_code == 404
