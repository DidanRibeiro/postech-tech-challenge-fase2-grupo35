# Tech Challenge — Fase 2 | POSTECH Data Analytics

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | **A informar antes da submissão** |
| Grupo | **Grupo 35** |
| Data de entrega | **A informar antes da submissão** |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| FERNANDA ARRUDA PIMENTA DA SILVA | **A informar** | arruda.fernanda@gmail.com |
| Guilherme Caetano Peron | **A informar** | caetano.peron@bb.com.br |
| ALEXANDRE DOS ANJOS DE JESUS | **A informar** | dosanjosdejesus@gmail.com |
| NATHALIA FERNANDA DOS SANTOS GELAIS | **A informar** | nathalia.santos@bb.com.br |
| DIDAN JUNQUEIRA RIBEIRO | **A informar** | didan.ribeiro@bb.com.br |

> Grupo e integrantes já preenchidos. Permanecem pendentes apenas a turma, os RMs, a data de entrega e os links finais da submissão.

---

## 2. Links da entrega

Estes três links devem ser idênticos aos do PDF gerado por `submissao/gerar_submissao.py`.

| Item | Link |
|---|---|
| Repositório | **A informar após criação do repositório público** |
| Vídeo executivo (≤ 5 min) | **A informar após upload** |
| Apresentação | `docs/apresentacao_executiva.pdf` — substituir por URL pública final |

---

## 3. O problema

O projeto desenvolve uma solução de Machine Learning para **predição de risco em pedidos de cartão de crédito**, utilizando informações cadastrais e histórico mensal de crédito. O objetivo é identificar perfis com maior probabilidade de apresentar atraso e apoiar uma decisão de crédito mais informada.

### Variável alvo

A variável `TARGET_BAD` foi construída a partir do histórico `STATUS`:

- `TARGET_BAD = 1`: cliente apresentou **ao menos uma ocorrência de atraso de 30 dias ou mais** (`STATUS` de `1` a `5`);
- `TARGET_BAD = 0`: cliente possui histórico observável, mas nunca apresentou `STATUS` de `1` a `5`;
- clientes cujo histórico é exclusivamente `X` foram excluídos, pois não existe comportamento de pagamento observável para classificá-los.

A regra produz **4.291 maus pagadores (12,96%)** e **28.819 bons pagadores (87,04%)**, totalizando **33.110 clientes** na base analítica.

### Dataset

| Campo | Valor |
|---|---|
| Fonte | Base disponibilizada no Tech Challenge — [Google Drive](https://drive.google.com/file/d/1z4yEyiCE_CGCWbvAAZQZSz-5-E5T5eYd/view?usp=sharing) |
| `application_record.csv` | 438.557 linhas × 18 colunas |
| `credit_record.csv` | 1.048.575 linhas × 3 colunas |
| Base analítica final | 33.110 clientes × 20 colunas (`ID` + 18 features + target) |
| Licença de uso | Não especificada no enunciado; arquivos brutos não são versionados no GitHub |

Principais grupos de variáveis:

| Variável / grupo | Tipo | Descrição |
|---|---|---|
| `AMT_INCOME_TOTAL` | numérica | renda total informada |
| `AGE_YEARS` | numérica derivada | idade em anos |
| `EMPLOYMENT_YEARS` | numérica derivada | tempo de emprego válido em anos |
| `CNT_CHILDREN`, `CNT_FAM_MEMBERS` | numéricas | composição familiar |
| flags de contato/patrimônio | binárias | carro, imóvel, telefone, e-mail etc. |
| renda, escolaridade, família, moradia, ocupação | categóricas | características cadastrais |

---

## 4. Como reproduzir

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Baixe os dois arquivos brutos e coloque em `data/raw/` com os nomes:

```text
data/raw/application_record.csv
data/raw/credit_record.csv
```

Os dados **não são versionados** no GitHub; veja `data/README.md`.

Execute os notebooks nesta ordem:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | distribuições, correlações, outliers e balanceamento |
| 2 | `notebooks/02_preprocessamento.ipynb` | target, limpeza, feature engineering e base tratada |
| 3 | `notebooks/03_modelagem.ipynb` | split, 5-fold CV, comparação e treino da pipeline final |
| 4 | `notebooks/04_avaliacao.ipynb` | teste independente, ROC-AUC, matriz, importância e negócio |

**Semente fixa:** `RANDOM_STATE = 42` em todo ponto com aleatoriedade.

---

## 5. Resultados

### Validação cruzada — conjunto de treinamento

| Modelo | Precisão | Recall | F1 |
|---|---:|---:|---:|
| **Random Forest + Class Weight** | **39,13%** | 47,13% | **42,76%** |
| KNN + RandomOverSampler | 33,30% | 57,24% | 42,10% |
| Decision Tree + RandomOverSampler | 32,88% | **57,82%** | 41,91% |

### Teste independente — modelo selecionado

| Métrica | Resultado |
|---|---:|
| Accuracy | **84,13%** |
| Precision | **41,09%** |
| Recall | **51,86%** |
| F1-score | **45,85%** |
| ROC-AUC | **79,80%** |
| Especificidade | **88,93%** |

Matriz de confusão final: **TN=5.126, FP=638, FN=413, TP=445**.

**Modelo escolhido:** `RandomForestClassifier` com `class_weight='balanced'`, `n_estimators=100`, `max_features='sqrt'`, `max_depth=None`, `min_samples_leaf=1` e `random_state=42`.

**Métricas priorizadas:** F1, Precision, Recall, ROC-AUC e tipos de erro. Acurácia isolada não é suficiente devido ao desbalanceamento. Sem uma função financeira explícita para FP e FN, o projeto evita impor pesos de negócio arbitrários.

---

## 6. Principais conclusões

1. A Random Forest com Class Weight apresentou o **melhor equilíbrio global** entre F1, Precision, ROC-AUC e controle de falsos positivos entre os finalistas.
2. No teste independente, o modelo alcançou **F1 de 45,85% e ROC-AUC de 79,80%**, sem deterioração relevante frente à validação.
3. As variáveis originais mais importantes foram **idade (22,00%)**, **renda (15,67%)**, **tempo de emprego (14,84%)** e **ocupação (11,09%)**.
4. O Recall final de **51,86%** mostra que ainda há espaço para melhorar a identificação da classe de maior risco.
5. Em uso real, a próxima evolução deveria incorporar uma **função de custo de negócio** para definir o threshold operacional sem reutilizar o teste final para tuning.

### Limitações e próximos passos

- target construído a partir da regra operacional de 30+ dias;
- base desbalanceada;
- ausência de custos monetários de FP e FN;
- importância por impureza representa relevância preditiva, não causalidade;
- recomenda-se monitoramento de drift, calibração e revalidação temporal antes de produção.

---

## 7. Estrutura do repositório

```text
.
├── data/          dados raw/processed — não versionados
├── notebooks/     análise numerada 01 → 04
├── src/           funções e configuração reutilizáveis
├── results/       figuras e métricas versionadas; modelos ignorados
├── docs/          apresentação executiva em PDF
└── submissao/     JSON + gerador do PDF com os três links
```

Detalhes em [`ESTRUTURA.md`](ESTRUTURA.md) e checklist em [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

Python 3.11, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib, Jupyter e ReportLab.
