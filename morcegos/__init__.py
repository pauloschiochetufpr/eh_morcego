"""
É morcego? Reconhecimento de padrões com SVM.

DS803 – Inteligência Computacional Aplicada I – TADS/UFPR

Módulos, na ordem em que o pipeline os usa:

    config          parâmetros do experimento
    dados           leitura de wpc50.dat e Names.txt
    referencia      o "especialista": confere se um gênero é de morcego
    modelo          o classificador SVM
    supervisao      base inicial (sementes positivas e negativas)
    enriquecimento  ciclos SVM -> candidatos -> conferência -> base
    avaliacao       métricas, busca em grade, validação cruzada, holdout
    exportacao      classes, índices, CSVs e resultados.json
    graficos        figuras para o relatório e a apresentação
    __main__        junta tudo: python -m morcegos
"""

__version__ = "1.0.0"
