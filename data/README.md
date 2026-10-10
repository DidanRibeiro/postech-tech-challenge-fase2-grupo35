# data/

**Os datasets não devem ser versionados no GitHub.** O `.gitignore` bloqueia o conteúdo de `raw/` e `processed/`.

| Pasta | Conteúdo |
|---|---|
| `raw/` | arquivos originais exatamente como baixados da fonte |
| `processed/` | `dataset_tratado.csv`, gerado pelo notebook 02 |

## Como obter

1. Baixe a base disponibilizada no Tech Challenge: https://drive.google.com/file/d/1z4yEyiCE_CGCWbvAAZQZSz-5-E5T5eYd/view?usp=sharing
2. Salve os arquivos como:
   - `data/raw/application_record.csv`
   - `data/raw/credit_record.csv`
3. Checksums SHA-256 dos arquivos usados neste projeto:
   - `application_record.csv`: `4833f502d02ad94295de3ffe74f665e726a4b04342d2e94f8cec41dce951925b`
   - `credit_record.csv`: `ba0006a4734f74422d68b0a7132ad591850be0a6affb535eb1042d207fe4b27e`

Depois execute os notebooks `01` → `04`. O notebook `02_preprocessamento.ipynb` gera `data/processed/dataset_tratado.csv`.
