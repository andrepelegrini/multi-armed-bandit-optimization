# Multi-Armed Bandit Optimization API

API REST em **Python + FastAPI** que recebe dados temporais de experimentos A/B (ou multi-variante), armazena em **SQL (SQLite/SQLAlchemy)** e retorna a alocação de tráfego recomendada para o próximo dia usando **Thompson Sampling** ou **UCB1** (Upper Confidence Bound 1).

---

## Stack

| Camada       | Tecnologia                         |
|--------------|------------------------------------|
| Framework    | FastAPI + Uvicorn                  |
| Banco        | SQLite via SQLAlchemy ORM          |
| Algoritmos   | Thompson Sampling · UCB1           |
| Validação    | Pydantic v2                        |
| Testes       | pytest + FastAPI TestClient        |
| Docs auto    | Swagger UI em `/docs`              |

---

## Arquitetura do Projeto

```
multi-armed-bandit-optimization/
├── app/
│   ├── __init__.py        ← package marker
│   ├── main.py            ← FastAPI app + todas as rotas
│   ├── database.py        ← ORM models, engine, SQL queries nomeadas
│   ├── schemas.py         ← Pydantic request/response models
│   ├── crud.py            ← Data-access layer (queries + lógica de negócio)
│   └── bandit.py          ← Algoritmos MAB (Thompson Sampling + UCB1)
├── tests/
│   ├── __init__.py
│   ├── conftest.py        ← Fixtures pytest (DB isolado em memória)
│   └── test_api.py        ← Suite completa (unit + integração)
├── sql/
│   └── schema_and_queries.sql  ← Schema DDL + queries analíticas comentadas
├── seed.py                ← Script para popular dados de demo
├── requirements.txt
└── README.md
```

---

## Início Rápido

### Pré-requisitos

- Python 3.11+
- Git

### 1. Clonar o repositório

```bash
git clone https://github.com/andrepelegrini/multi-armed-bandit-optimization.git
cd multi-armed-bandit-optimization
```

### 2. Criar e ativar o virtualenv

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Subir o servidor

```bash
uvicorn app.main:app --reload
```

Acesse:
- **Swagger UI (recomendado):** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **API base:** `http://127.0.0.1:8000`

### 5. Popular com dados de demonstração (opcional)

Com o servidor rodando, abra um segundo terminal (com o virtualenv ativo) e execute:

```bash
python seed.py
```

Cria um experimento A/B/C com 14 dias de observações e imprime a alocação recomendada por Thompson Sampling e UCB1 diretamente no terminal.

---

## Rodar os Testes

```bash
# Todos os testes com saída detalhada
pytest tests/ -v

# Com relatório de cobertura
pytest tests/ -v --cov=app --cov-report=term-missing

# Apenas testes unitários de algoritmo
pytest tests/ -v -k "TestThompson or TestUCB1"

# Apenas testes de integração da API
pytest tests/ -v -k "TestExperiments or TestObservations or TestAllocations"
```

Os testes usam um banco SQLite separado (`test_bandit.db`) que é criado e destruído automaticamente.

---

## Schema do Banco de Dados

```sql
experiments   (id, name, description, metric, created_at)
variants      (id, experiment_id, name, is_control, created_at)
observations  (id, variant_id, experiment_id, obs_date, impressions, clicks, created_at)
allocations   (id, experiment_id, variant_id, target_date, allocation_pct, algorithm, computed_at)
```

**Destaques:**
- `UNIQUE(variant_id, obs_date)` → upsert idempotente via `ON CONFLICT`
- `ON DELETE CASCADE` → integridade referencial automática
- Índices em `experiment_id` e `obs_date` para queries analíticas

---

## Algoritmos MAB

### Thompson Sampling (padrão)

Para cada braço i, mantemos uma distribuição posterior `Beta(α, β)`:
- `α = total_clicks + 1` (prior de Laplace)
- `β = (impressions - clicks) + 1`

A cada iteração Monte-Carlo (N=50.000):
1. Amostramos `θᵢ ~ Beta(αᵢ, βᵢ)` para cada braço
2. O braço com maior `θᵢ` "vence"

A fração de vitórias de cada braço é a **alocação recomendada (%)**.

### UCB1 (alternativo)

```
score_i = CTR_i + sqrt(2 * ln(N_total) / n_i)
```

Alocação proporcional ao score. Braços sem dados recebem score infinito (100% exploração).

---

## Endpoints

### `POST /experiments`
Cria experimento. O primeiro variant na lista é tratado como **controle**.

```json
{
  "name": "Homepage Hero Test",
  "description": "Comparando 3 layouts",
  "metric": "ctr",
  "variants": ["control", "variant_a", "variant_b"]
}
```

---

### `GET /experiments` · `GET /experiments/{id}`
Lista ou detalha experimentos.

---

### `POST /experiments/{id}/observations`
Insere dados diários (batch). Chamadas repetidas para o mesmo `(variant, date)` fazem **upsert**.

```json
{
  "observations": [
    {"variant_name": "control",   "obs_date": "2024-05-01", "impressions": 1000, "clicks": 42},
    {"variant_name": "variant_a", "obs_date": "2024-05-01", "impressions": 1000, "clicks": 68}
  ]
}
```

---

### `GET /experiments/{id}/allocations`
**Endpoint principal.** Executa o MAB e retorna alocação para `target_date`.

Query params:
- `algorithm`: `thompson_sampling` (padrão) | `ucb1`
- `target_date`: data alvo no formato `YYYY-MM-DD` (padrão: amanhã)

```json
{
  "experiment_id": 1,
  "experiment_name": "Homepage Hero Test",
  "target_date": "2024-05-15",
  "algorithm": "thompson_sampling",
  "allocations": [
    {"variant_name": "control",   "allocation_pct": 12.5, "smoothed_ctr": 0.0418},
    {"variant_name": "variant_a", "allocation_pct": 71.3, "smoothed_ctr": 0.0652},
    {"variant_name": "variant_b", "allocation_pct": 16.2, "smoothed_ctr": 0.0521}
  ]
}
```

---

### `GET /experiments/{id}/analytics` · `GET /experiments/{id}/observations`
Série temporal diária por variante (CTR, impressões, cliques por dia).

---

## Exemplos cURL

```bash
# 1. Criar experimento
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{"name":"Landing Page Test","variants":["control","v_a","v_b"]}'

# 2. Inserir observações (substitua {ID})
curl -X POST http://localhost:8000/experiments/{ID}/observations \
  -H "Content-Type: application/json" \
  -d '{"observations":[
    {"variant_name":"control","obs_date":"2024-05-01","impressions":1000,"clicks":40},
    {"variant_name":"v_a","obs_date":"2024-05-01","impressions":1000,"clicks":70}
  ]}'

# 3. Obter recomendação Thompson Sampling
curl http://localhost:8000/experiments/{ID}/allocations

# 4. Obter recomendação UCB1
curl "http://localhost:8000/experiments/{ID}/allocations?algorithm=ucb1"
```

---

## Decisões de Design

| Decisão | Justificativa |
|---------|--------------|
| **FastAPI** | Validação automática via Pydantic, Swagger UI gratuito, alta performance |
| **SQLAlchemy ORM + raw SQL** | ORM para operações simples, SQL raw para queries analíticas complexas |
| **Thompson Sampling como padrão** | Sem parâmetros de tuning, naturalmente Bayesiano, suporta N braços |
| **UCB1 como alternativa** | Frequentista, mais interpretável, útil para comparação |
| **Upsert via ON CONFLICT** | Permite ingestão idempotente, simplifica o cliente |
| **Bônus: N braços** | Schema, ORM e algoritmos suportam qualquer número de variantes |
