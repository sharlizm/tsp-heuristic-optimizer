# TSP OPTIMIZER — Streamlit

Aplikasi GUI berbasis Streamlit untuk eksperimen **Travelling Salesman Problem (TSP)** dengan:
- Construction heuristic: Nearest Neighbor, Nearest Insertion, Cheapest Insertion, Farthest Insertion, Arbitrary Insertion, Manual Route
- Distance: Euclidean / Manhattan
- Local Search: Swap / 2-opt / 3-opt, First/Best Improvement
- Metaheuristic utama: Simulated Annealing dan Tabu Search
- Parameter algoritma dapat diubah
- Tabu Search Explorer dengan histori eksekusi aktual
- Convergence dan comparison chart
- Multi-run dan sensitivity experiment
- Import CSV/XLSX
- CSV export
- Reproducible random seed

## Struktur

```text
tsp-optimizer/
├── streamlit_app.py          # Entry point utama
├── requirements.txt          # Dependency Streamlit Cloud
├── .streamlit/config.toml
├── backend/
│   └── app/
│       ├── algorithms/       # Distance, construction, neighborhood, local search, SA, TS
│       ├── models/           # Pydantic models
│       └── services/         # Multi-run experiment
├── backend/tests/             # Unit tests algoritma
└── sample_data/
    ├── sample_tsp.csv
    └── sample_tsp.xlsx
```

## Jalankan lokal

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional, untuk testing
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser pada URL lokal yang diberikan Streamlit.

## Deploy ke Streamlit Community Cloud

1. Push repository ini ke GitHub.
2. Buka `share.streamlit.io` lalu **Create app**.
3. Pilih repository, branch, dan entrypoint `streamlit_app.py`.
4. Pastikan `requirements.txt` berada di root repository.
5. Deploy.

Repository GitHub menjadi source app. Setelah perubahan Python di-commit/push, Community Cloud akan mendeteksi perubahan dan memperbarui app. Perubahan dependency di `requirements.txt` memicu redeploy dependency. 

## Core workflow

```text
INPUT DATA
Manual / Random / CSV / XLSX
        ↓
CONSTRUCTION HEURISTIC
NN / NI / CI / FI / AI / Manual
        ↓
INITIAL SOLUTION
Route + Distance
        ↓
LOCAL SEARCH
Swap / 2-opt / 3-opt
        ↓
METAHEURISTIC
Simulated Annealing / Tabu Search
        ↓
RESULT
        ↓
COMPARISON
        ↓
VISUALIZATION + ANALYSIS
```

## Catatan algoritma

### Simulated Annealing

- Jika `delta <= 0`, candidate diterima.
- Jika `delta > 0`, probabilitas penerimaan menggunakan `exp(-delta / temperature)`.
- Pendinginan memakai `T_next = alpha * T_current`.
- History menyimpan candidate, delta, temperature, acceptance probability, dan acceptance aktual.

### Tabu Search

- Setiap iterasi membentuk neighborhood.
- Candidate dievaluasi terhadap distance.
- Candidate tabu ditolak kecuali memenuhi aspiration criterion.
- Move terpilih masuk tabu list dan age diperbarui.
- History menyimpan neighborhood, selected move, tabu list, aspiration, dan best distance.

## Reproducibility

Untuk data, parameter, dan seed yang sama, proses menggunakan pseudo-random generator Python yang sama pada environment yang sama.

## Testing

```bash
pytest -q backend/tests
```

## Sample data

Gunakan `sample_data/sample_tsp.csv` atau `sample_data/sample_tsp.xlsx` untuk demo 6 kota.
