É morcego? Reconhecimento de padrões com SVM

Encontra os morcegos (ordem Chiroptera) entre 8426 organismos descritos por 50
padrões do genoma mitocondrial, usando um SVM com kernel RBF e enriquecimento
iterativo da base rotulada.

**Resultado:** 79 morcegos de 32 gêneros. Validação cruzada 10-fold repetida
10x: sensibilidade 99,88%, precisão 100%, acurácia 99,999%.

Como rodar

Precisa de Python 3.10 ou mais novo.

**Windows (PowerShell ou Prompt de Comando)**

```bat
cd e-morcego
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m morcegos
```

**Linux / macOS**

```bash
cd e-morcego
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m morcegos
```

O processo leva cerca de 10 segundos e grava tudo em `resultados/`.

Opções:

```
python -m morcegos --help
python -m morcegos --indice-base 0        # índices a partir de 0
python -m morcegos --dados outra_pasta --saida outra_saida
```

Testes:

```
python -m pytest
```

Saída esperada

```
[1/6] Lendo os dados
  8426 instâncias x 50 características
[2/6] Supervisão inicial
  48 positivos (45 dos gêneros da pesquisa), 597 negativos
[3/6] Enriquecimento iterativo com SVM
  ciclo 1: +25 morcegos, 0 candidatos rejeitados, 35 previstos positivos fora da base
  ciclo 2: +6 morcegos, 19 candidatos rejeitados, 10 previstos positivos fora da base
  ciclo 3: +0 morcegos, 25 candidatos rejeitados, 0 previstos positivos fora da base
[4/6] Classificação final
  1020 instâncias conferidas, 7406 classificadas pelo SVM
  morcegos encontrados: 79
[5/6] Testes de desempenho
  CV 10-fold x10: sensibilidade 99,88%, precisão 100,00%, acurácia 99,999%
  holdout 30%: sensibilidade 95,83%, precisão 100,00%, acurácia 99,96%  (perdeu: Desmodus_rotundus)
  só a semente: precisão 88,57% -> com os ciclos: 100,00%
[6/6] Exportando resultados e figuras
```

Estrutura

```
e-morcego/
├── dados/
│   ├── wpc50.dat                
│   ├── Names.txt                
│   └── generos_chiroptera.txt    lista de referência: 235 gêneros de morcegos
├── morcegos/
│   ├── config.py                 parâmetros (semente, C, tamanho dos ciclos...)
│   ├── dados.py                  leitura dos arquivos
│   ├── referencia.py             o "especialista" que confere candidatos
│   ├── modelo.py                 o SVM (padronização + RBF + pesos balanceados)
│   ├── supervisao.py             base inicial (sementes)
│   ├── enriquecimento.py         ciclos SVM -> candidatos -> conferência
│   ├── avaliacao.py              métricas, busca em grade, CV, holdout
│   ├── exportacao.py             classes, índices, CSVs, JSON
│   ├── graficos.py               figuras
│   └── __main__.py               executa tudo em ordem
├── tests/test_pipeline.py
├── resultados/                   gerado ao rodar
└── requirements.txt
```

Arquivos do `resultados/`

| Arquivo | Conteúdo |
| --- | --- |
| `classes.txt` | 8426 linhas, 1 = morcego, 0 = outro, na ordem original |
| `indices_morcegos.txt` | os 79 índices positivos (a partir de 1, por padrão) |
| `morcegos_encontrados.csv` | índice, nome e pontuação do SVM de cada morcego |
| `resultado_completo.csv` | índice, nome, classe e pontuação das 8426 instâncias |
| `resultados.json` | ciclos, busca em grade e todas as métricas |
| `fig_pca.png`, `fig_ciclos.png`, `fig_scores.png` | figuras |

Como funciona

1. **Supervisão inicial.** Positivos: os 5 gêneros de morcegos da pesquisa
   preliminar (*Myotis*, *Pteropus*, *Rhinolophus*, *Desmodus*, *Tadarida*).
   Negativos: 600 instâncias sorteadas e conferidas.
2. **Enriquecimento.** O SVM pontua as instâncias não rotuladas; os 25 candidatos
   mais fortes de cada ciclo são conferidos com a lista de referência e voltam
   para a base, junto com 100 instâncias sorteadas. Os candidatos rejeitados
   (toupeiras, musaranhos...) viram negativos difíceis.
3. **Parada.** Quando nenhum morcego novo aparece e o SVM não prevê mais
   positivos fora da base.
4. **Classificação final** das instâncias não conferidas com o SVM treinado na
   base enriquecida.
5. **Avaliação** com busca em grade (kernel e C), validação cruzada 10-fold
   repetida 10x e holdout 70/30.
