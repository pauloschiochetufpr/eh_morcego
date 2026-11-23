from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    pasta_dados: Path = Path("dados")
    pasta_saida: Path = Path("resultados")

    semente: int = 42

    generos_semente: tuple = ("Myotis", "Pteropus", "Rhinolophus", "Desmodus", "Tadarida")
    n_negativos_iniciais: int = 600   # amostra aleatória conferida uma a uma

    candidatos_por_ciclo: int = 25    # os mais bem pontuados pelo SVM, conferidos
    aleatorios_por_ciclo: int = 100   # amostra extra para cobrir o espaço dos negativos

    svm_kernel: str = "rbf"
    svm_C: float = 10.0

    cv_folds: int = 10
    cv_repeticoes: int = 10
    holdout_teste: float = 0.3

    indice_base: int = 1
