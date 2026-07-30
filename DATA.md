# Data: sources and expected layout

Nothing under `data/` is committed (redistribution rights differ per source; everything below
is publicly downloadable). Default root: `../data` relative to this repo (override with
`GRAPHSEM_DATA`). `v6/testbeds.py` and `v6/pool.py` are the source of truth for exact paths.

## 1. Questionnaire datasets (openpsychometrics.org)

Download the raw-data zips from the public index at `https://openpsychometrics.org/_rawdata/`
and unzip so each dataset directory contains its `data.csv` and `codebook.txt`:

```
data/BIG5/                      <- BIG5.zip
data/pool/16PF/                 <- 16PF.zip
data/pool/GCBS/                 <- GCBS.zip
data/pool/HEXACO/               <- HEXACO.zip
data/pool/HSQ/                  <- HSQ.zip
data/pool/KIMS/                 <- KIMS.zip
data/pool/MACH/                 <- MACH.zip  (MACH-IV)
data/pool/RIASEC/               <- RIASEC_data12Dec2018.zip
data/pool/RSE/                  <- RSE.zip
data/pool/SD3/                  <- SD3.zip
data/pool/...                   <- CFCS.zip, NPAS-data-16December2018.zip, SCS.zip, TMA.zip,
                                   WPI.zip, HSNS+DD.zip (training pool; see pool.py for the
                                   exact directory name each loader expects)
```

## 2. Holzinger–Swineford 1939 (`data/HS.data.csv`)

The classic 24-test cognitive battery; available e.g. as `HolzingerSwineford1939` in the R
`lavaan` package (export to CSV). Column names must match what `pool.py`'s `hs` loader reads.

## 3. TLVD benchmark files (`data/TLVD/`)

- `Final_Multitasking_Data.sav` — Himi Multitasking study data, OSF project `tn6hp`.
- `multitasking_alpha0.05_rtscale1_N-1.dot`, `multitasking_description.json` — released with
  the TLVD paper's code (`https://github.com/HYJ9999/TLVD.git`).

## 4. Dictionary sources

- ConceptNet Numberbatch English 19.08: `numberbatch-en-19.08.txt.gz` (ConceptNet releases,
  `github.com/commonsense/conceptnet-numberbatch`) -> `data/numberbatch-en-19.08.txt.gz`.
- WordNet: fetched automatically via `nltk` on first use (`v6/negop.py`).
- Cognitive Atlas vocabulary: `data/cogatlas_{concepts,tasks}.json` from
  `https://www.cognitiveatlas.org/api/v-alpha/concept?format=json` (and `.../task?format=json`).

## 5. Encoder

`intfloat/e5-large-v2` from Hugging Face (downloaded automatically; set `HF_HUB_OFFLINE=1`
after the first download for reproducible offline runs).
