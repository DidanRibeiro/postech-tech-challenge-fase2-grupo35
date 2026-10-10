# Tech Challenge — Fase 2 | POSTECH Data Analytics

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | **Data Analytics BB** |
| Grupo | **Grupo 35** |
| Data de entrega | **10/10/2026** |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| Alexandre dos Anjos de Jesus | **RM377732** | dosanjosdejesus@gmail.com |
| Didan Junqueira Ribeiro | **RM377825** | didan.ribeiro@bb.com.br |
| Fernanda Arruda Pimenta da Silva | **RM377753** | arruda.fernanda@gmail.com |
| Guilherme Caetano Peron | **RM377756** | caetano.peron@bb.com.br |
| Nathalia Fernanda dos Santos Gelais | **RM377743** | nathalia.santos@bb.com.br |

---

## 2. Links da entrega

Os links abaixo são os mesmos informados em `submissao/entrega.json` e no PDF final de submissão.

| Item | Link |
|---|---|
| Repositório | https://github.com/DidanRibeiro/postech-tech-challenge-fase2-grupo35 |
| Vídeo executivo (≤ 5 min) | https://drive.google.com/file/d/1zn14_Jz4m1fEISZhoIBry205-ASCr1T-/view?usp=sharing |
| Apresentação | https://drive.google.com/file/d/12QuFush3MjMrlNZklt61BRPfMU3m667d/view?usp=sharing |

Uma cópia da apresentação também está versionada em `docs/apresentacao_executiva.pdf`.

---

## 3. Problema de negócio

O projeto desenvolve uma solução de Machine Learning para **predição de risco em pedidos de cartão de crédito**, utilizando informações cadastrais e histórico mensal de crédito.

O objetivo é identificar clientes com maior probabilidade de apresentar atraso e disponibilizar uma evidência preditiva que possa **apoiar a decisão de crédito**, sem substituir regras, políticas ou análises de negócio.

### Variável-alvo

A variável `TARGET_BAD` foi construída a partir do histórico `STATUS`:

- `TARGET_BAD = 1`: cliente apresentou **ao menos uma ocorrência de atraso de 30 dias ou mais** (`STATUS` de `1` a `5`);
- `TARGET_BAD = 0`: cliente possui histórico observável, mas nunca apresentou `STATUS` de `1` a `5`;
- clientes cujo histórico era exclusivamente `X` foram excluídos, pois não havia comportamento de pagamento observável suficiente para classificá-los.

A regra resultou em:

- **28.819 bons pagadores (87,04%)**;
- **4.291 maus pagadores (12,96%)**;
- **33.110 clientes** na base analítica final.

Como a classe de maus pagadores representa apenas 12,96% da base, o problema é **desbalanceado**. Por isso, a avaliação não foi baseada apenas em acurácia.

---

## 4. Dados

### Bases utilizadas

| Arquivo | Dimensão original |
|---|---:|
| `application_record.csv` | 438.557 linhas × 18 colunas |
| `credit_record.csv` | 1.048.575 linhas × 3 colunas |

Após integração, tratamento e construção do target:

| Base final | Dimensão |
|---|---:|
| `dataset_tratado.csv` | **33.110 linhas × 20 colunas** |

A base final contém:

- `ID`;
- 18 features;
- `TARGET_BAD`.

Os arquivos brutos e a base processada **não são versionados no GitHub**, conforme as regras do `.gitignore`.

### Principais grupos de variáveis

| Variável / grupo | Tipo | Descrição |
|---|---|---|
| `AMT_INCOME_TOTAL` | numérica | renda total informada |
| `AGE_YEARS` | numérica derivada | idade em anos |
| `EMPLOYMENT_YEARS` | numérica derivada | tempo de emprego válido em anos |
| `CNT_CHILDREN` | numérica | quantidade de filhos |
| `CNT_FAM_MEMBERS` | numérica | quantidade de membros da família |
| flags de contato e patrimônio | binárias | carro, imóvel, telefone, e-mail etc. |
| renda, escolaridade, família, moradia e ocupação | categóricas | características cadastrais |

### Tratamentos relevantes

Entre os principais tratamentos realizados:

- preenchimento de `OCCUPATION_TYPE` ausente com `Unknown`;
- criação de `AGE_YEARS` a partir de `DAYS_BIRTH`;
- tratamento do valor sentinela `365243` em `DAYS_EMPLOYED`;
- criação de `EMPLOYMENT_YEARS`;
- criação de `FLAG_EMPLOYMENT_SENTINEL`;
- criação de `FLAG_FAMILY_INCONSISTENT`;
- conversão de flags binárias para `0/1`;
- codificação das variáveis categóricas dentro da pipeline de modelagem.

---

## 5. Metodologia

A metodologia seguiu as seguintes etapas:

1. integração das bases cadastral e de histórico de crédito;
2. análise exploratória dos dados;
3. construção da variável-alvo;
4. limpeza e feature engineering;
5. separação estratificada entre treino e teste;
6. comparação de algoritmos e estratégias de balanceamento;
7. validação cruzada estratificada;
8. seleção do modelo final;
9. avaliação independente no conjunto de teste;
10. interpretação das métricas e das features.

### Separação treino/teste

Foi utilizado:

```python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Resultado:

- treino: **26.488 clientes**;
- teste: **6.622 clientes**.

O conjunto de teste permaneceu isolado durante a seleção do modelo.

### Validação cruzada

Foi utilizada validação cruzada estratificada em 5 folds:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

### Algoritmos avaliados

Foram avaliados, entre outros:

- K-Nearest Neighbors (KNN);
- Decision Tree;
- Random Forest;
- Support Vector Machine (SVM).

Também foram avaliadas estratégias para lidar com o desbalanceamento:

- `class_weight='balanced'`;
- RandomOverSampler;
- RandomUnderSampler;
- SMOTENC.

---

## 6. Resultados da validação cruzada

### Principais finalistas

| Modelo | Precision | Recall | F1 |
|---|---:|---:|---:|
| **Random Forest + Class Weight** | **39,13%** | 47,13% | **42,76%** |
| KNN + RandomOverSampler | 33,28% | 57,27% | 42,09% |
| Decision Tree + RandomOverSampler | 32,88% | **57,82%** | 41,91% |

### ROC-AUC em validação

| Modelo | ROC-AUC |
|---|---:|
| **Random Forest + Class Weight** | **77,33%** |
| Random Forest + SMOTENC | 77,04% |
| KNN otimizado | 74,33% |
| Random Forest + UnderSampling | 72,29% |
| Decision Tree | 72,23% |

A **Random Forest com Class Weight** foi selecionada por apresentar o melhor equilíbrio geral entre F1, Precision, ROC-AUC e controle de falsos positivos entre os principais candidatos.

---

## 7. Modelo final

Modelo selecionado:

```text
RandomForestClassifier
class_weight = 'balanced'
n_estimators = 100
max_features = 'sqrt'
max_depth = None
min_samples_leaf = 1
random_state = 42
```

O modelo é treinado dentro de uma pipeline reproduzível que inclui o pré-processamento necessário.

---

## 8. Avaliação no teste independente

Após a seleção formal do modelo, o conjunto de teste foi utilizado pela primeira vez para avaliação final.

### Métricas

| Métrica | Resultado |
|---|---:|
| Accuracy | **84,13%** |
| Precision | **41,09%** |
| Recall | **51,86%** |
| F1-score | **45,85%** |
| ROC-AUC | **79,80%** |
| Especificidade | **88,93%** |

### Matriz de confusão

| | Predito bom | Predito mau |
|---|---:|---:|
| **Real bom** | **TN = 5.126** | **FP = 638** |
| **Real mau** | **FN = 413** | **TP = 445** |

Interpretação:

- dos **5.764 bons pagadores** no teste, **5.126** foram corretamente classificados;
- **638 bons pagadores** foram classificados como maus;
- dos **858 maus pagadores**, **445** foram identificados corretamente;
- **413 maus pagadores** não foram identificados pelo modelo.

A especificidade de **88,93%** mostra boa capacidade de preservar a classe de bons pagadores, enquanto o Recall de **51,86%** mostra que ainda existe espaço para melhorar a identificação da classe de maior risco.

---

## 9. Importância das variáveis

Após agregação das features transformadas para suas variáveis originais, as principais importâncias foram:

| Variável | Importância |
|---|---:|
| `AGE_YEARS` | **22,00%** |
| `AMT_INCOME_TOTAL` | **15,67%** |
| `EMPLOYMENT_YEARS` | **14,84%** |
| `OCCUPATION_TYPE` | **11,09%** |

As três primeiras variáveis concentram **52,51%** da importância total, e as quatro primeiras concentram **63,60%**.

A Feature Importance representa **relevância preditiva**, e não causalidade.

---

## 10. Principais conclusões

1. A base apresenta desbalanceamento relevante entre bons e maus pagadores, o que torna inadequado avaliar o desempenho apenas pela acurácia.
2. A Random Forest com Class Weight apresentou o melhor equilíbrio geral entre as alternativas avaliadas.
3. No teste independente, o modelo alcançou **F1 de 45,85%** e **ROC-AUC de 79,80%**.
4. O modelo preservou corretamente **88,93% dos bons pagadores**.
5. O Recall de **51,86%** mostra que parte relevante dos maus pagadores ainda não é identificada no threshold padrão.
6. O modelo deve ser utilizado como **apoio à decisão**, e não como regra automática de aprovação ou recusa.
7. Em um cenário real, a evolução natural seria incorporar os custos financeiros de falsos positivos e falsos negativos para definir um threshold operacional adequado.

### Limitações e próximos passos

- target construído a partir da regra operacional de atraso de 30+ dias;
- base desbalanceada;
- ausência de custos monetários explícitos para FP e FN;
- Feature Importance não implica causalidade;
- necessidade de monitoramento de drift;
- necessidade de calibração e revalidação temporal antes de uso em produção;
- possibilidade de otimização futura do threshold com base em critérios de negócio.

---

## 11. Como reproduzir

### 11.1 Clonar o repositório

```bash
git clone https://github.com/DidanRibeiro/postech-tech-challenge-fase2-grupo35.git
cd postech-tech-challenge-fase2-grupo35
```

### 11.2 Criar o ambiente virtual

```bash
python -m venv .venv
```

### Windows / PowerShell

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

### Linux / macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

### 11.3 Disponibilizar os dados

Os arquivos brutos devem ser colocados em:

```text
data/raw/application_record.csv
data/raw/credit_record.csv
```

Os datasets não são versionados no GitHub. Consulte `data/README.md`.

### 11.4 Executar os notebooks

Execute na seguinte ordem:

| # | Notebook | Objetivo |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | análise exploratória e distribuição do target |
| 2 | `notebooks/02_preprocessamento.ipynb` | limpeza, target, feature engineering e base tratada |
| 3 | `notebooks/03_modelagem.ipynb` | split, validação cruzada, comparação e treinamento |
| 4 | `notebooks/04_avaliacao.ipynb` | avaliação independente, ROC-AUC, matriz de confusão e importância |

A aleatoriedade é controlada com:

```text
RANDOM_STATE = 42
```

---

## 12. Estrutura do repositório

```text
.
├── .github/
│   └── workflows/
│       └── validar-entrega.yml
│
├── data/
│   ├── raw/                  # não versionado
│   ├── processed/            # não versionado
│   └── README.md
│
├── docs/
│   ├── apresentacao_executiva.pdf
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelagem.ipynb
│   ├── 04_avaliacao.ipynb
│   └── README.md
│
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── models/               # modelos ignorados pelo Git
│   └── README.md
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── README.md
│
├── submissao/
│   ├── entrega.json
│   ├── gerar_submissao.py
│   ├── submissao_DataAnalyticsBB_Grupo35.pdf
│   └── README.md
│
├── CHECKLIST.md
├── ESTRUTURA.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 13. Arquivos de resultados

### Figuras

Disponíveis em `results/figures/`:

- `01_distribuicoes.png`
- `02_boxplots.png`
- `03_correlacoes.png`
- `04_distribuicao_target.png`
- `05_f1_validacao_cruzada.png`
- `06_matriz_confusao_teste.png`
- `07_curva_roc_teste.png`
- `08_feature_importance.png`

### Métricas

Disponíveis em `results/metrics/`:

- `validacao_cruzada_modelos.csv`
- `metricas_teste_final.csv`
- `feature_importance.csv`
- `resumo_modelo_final.json`

O arquivo serializado do modelo final (`.joblib`) é regenerável pelos notebooks e, conforme a estrutura do template, não é versionado.

---

## 14. Tecnologias utilizadas

- Python 3.11
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- joblib
- Jupyter Notebook
- ReportLab

---

## 15. Submissão

Os dados oficiais da entrega estão em:

```text
submissao/entrega.json
```

O PDF final de submissão foi gerado pelo script oficial:

```bash
python submissao/gerar_submissao.py
```

Arquivo gerado:

```text
submissao/submissao_DataAnalyticsBB_Grupo35.pdf
```

A validação automatizada do template é executada pelo GitHub Actions por meio de:

```text
.github/workflows/validar-entrega.yml
```

---

## Licença

Este repositório utiliza a licença definida no arquivo [`LICENSE`](LICENSE).
