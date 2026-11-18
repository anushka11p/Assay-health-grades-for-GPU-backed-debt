# Assay: verifiable health records for GPUs

**Lenders now finance GPU clusters, but they can't see the condition of their collateral.**
GPUs fail in ways a spec sheet doesn't show: worn memory, overheating, degraded links.
Assay is the inspector. It measures a GPU, signs the result on the machine where it was
taken, logs it in a tamper-evident transparency log, and grades it A–F against its peers.
The result is a health record a lender, buyer or insurer can check *without trusting the
operator*.

```
 GPU machine (Colab / Kaggle / server)                        Anyone
┌──────────────────────────────────────────────┐   ┌────────────────────────────────┐
│ assay check                                  │   │ assay verify report.json       │
│  ├─ NVML telemetry, every 250 ms (read-only) │   │  └─ signature over every byte  │
│  ├─ CUDA stress tests (separate processes)   │──▶│ assay verify --proof ...       │
│  └─ Ed25519 signature over canonical JSON    │   │  └─ Merkle inclusion proof     │
└──────────────────────────────────────────────┘   │ assay log consistency old.json │
                      │                             │  └─ log only grew, never edited│
                      ▼                             └────────────────────────────────┘
        assay log append (RFC 6962 Merkle log, signed tree head)
                      │
                      ▼
        grading (A–F + reasons) ─▶ static census site (GitHub Pages)
```

## What a check measures

| | What | How |
|---|---|---|
| Identity | board serial, UUID, model, VBIOS | NVML |
| Memory health | ECC errors (lifetime and since boot), retired pages / row remapping | NVML |
| Errors | XID driver errors during the check | NVML event set |
| Interconnect | PCIe generation and width **under load**, replay counter | NVML, sampled every 250 ms |
| Thermals | temperature, power, clocks, throttle reasons | NVML, sampled every 250 ms |
| `gemm` | FP16 (tensor core) and FP32 matrix-multiply GFLOPS | cuBLAS |
| `membw` | memory bandwidth (STREAM copy/triad) | CUDA |
| `memtest` | 6 bit patterns over 75% of free memory, with a self-test | CUDA |
| `soak` | 3-minute sustained load: does it hold its speed? | cuBLAS |
| `allreduce` | multi-GPU bus bandwidth | NCCL |

Each stress test runs as a separate process with a timeout, only on an idle GPU, while the
agent keeps sampling. Every result therefore comes with what the GPU was doing at the time
(e.g. *"lost 60% of its speed during the soak; hardware thermal slowdown in 34 readings"*).

## Try it (no GPU needed)

```bash
cd agent
go run ./cmd/assay keygen
go run ./cmd/assay check --backend=fake --scenario=thermal-throttle --interval=100ms
go run ./cmd/assay verify --pubkey ~/.assay/agent.pub report.json
```

Fake scenarios: `healthy`, `hbm-degrading`, `thermal-throttle`, `xid-storm`, `pcie-flaky`.
On a real NVIDIA GPU, use `--backend=nvml` (the default) after building the probes with
`make -C probes`. On free GPUs, open [`notebooks/colab_check.ipynb`](notebooks/colab_check.ipynb)
in Google Colab or [`notebooks/kaggle_check.ipynb`](notebooks/kaggle_check.ipynb) on Kaggle and
choose Run all.

## The census

Reports from free Colab and Kaggle GPUs go in [`census/raw/`](census/raw/).
[`scripts/build.sh`](scripts/build.sh) verifies the signature of every report (and stops
on the first failure), checks the log, grades every GPU and builds the static site in
`site/`, which deploys to GitHub Pages.

**Results so far:** 1 physical GPU (a Colab Tesla T4, 8 Nov 2026), checked with telemetry
only: ECC on with zero lifetime errors, no retired pages, no XID errors. That run is also
where the first real lesson came from: an **idle** T4 drops its PCIe link to Gen1 to save
power, so Assay judges link speed only under load. The census grows with every notebook
run.

## How a grade is decided

Each problem becomes a finding with a severity; the grade is the most serious one, so
every grade carries its reason.

| Grade | Severity | Examples |
|---|---|---|
| **F** | critical | uncorrectable ECC errors, double-bit retired pages, memtest mismatches, XID 48/64/79/95 |
| **D** | major | PCIe below max *under load*, hardware thermal slowdown, soak loses ≥25%, failed test, ≥30% below the model median |
| **C** | minor | software thermal slowdown, soak loses 10–25%, ≥1,000 corrected ECC errors, 15–30% below median |
| **B** | note | ECC turned off |
| **A** | none | no problems in any check that ran |

Peers are compared only with **3 or more GPUs of the same model**. Values a GPU can't
report are listed as "not checked", never treated as zero.

## What the signature proves, and what it doesn't

- **Proves:** the report is byte-for-byte what the agent wrote. Verification covers the
  exact bytes in the file, so reports stay verifiable as the format grows; a regression
  test pins the real 8 Nov Colab report. With `--pubkey`, it also proves *which* install
  signed it.
- **Proves (log):** the report was logged, and the log only grew. No logged report was
  removed or edited (RFC 6962 inclusion and consistency proofs).
- **Doesn't prove:** which physical GPU was measured. On T4/P100 a modified agent on a
  dishonest host could sign false readings. Binding readings to silicon needs hardware
  attestation (NVIDIA nvtrust on H100/Blackwell in confidential-computing mode). That's
  the planned next step.

## Repository

| Path | What |
|---|---|
| [`agent/`](agent/) | Go agent: CLI, NVML and fake backends, probe runner, signing, transparency log |
| [`probes/`](probes/) | CUDA stress tests (cuBLAS, NCCL); `make -C probes` |
| [`grading/`](grading/) | Python (standard library only): census builder, A–F grading, static site |
| [`notebooks/`](notebooks/) | Colab and Kaggle notebooks, generated by `make_notebooks.py` |
| [`census/raw/`](census/raw/) | signed reports from real GPUs |
| [`docs/`](docs/) | design spec and build plan |

## Tests

```bash
cd agent && go test -race ./...          # Go: agent, signing, Merkle log, probe runner
cd grading && python3 -m unittest -v     # Python: grading rules and census
```

CI runs both on every push and compiles the CUDA probes inside NVIDIA's CUDA container.
The Merkle tree is checked against Certificate Transparency's published test vectors and
an independent implementation.
