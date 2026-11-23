"""Leitura dos padrões (wpc50.dat) e dos nomes (Names.txt)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Conjunto:
    X: np.ndarray          # 8426 x 50, padrões mitocondriais
    nomes: list            # nome científico de cada instância, na mesma ordem
    genero: np.ndarray     # primeira parte do nome (ex.: "Myotis")

    @property
    def n(self) -> int:
        return len(self.nomes)


def _achar(pasta: Path, nome: str) -> Path:
    """Acha o arquivo sem diferenciar maiúsculas (Wpc50.dat ou wpc50.dat)."""
    for p in pasta.iterdir():
        if p.name.lower() == nome.lower():
            return p
    raise FileNotFoundError(f"'{nome}' não encontrado em {pasta.resolve()}")


def carregar(pasta: Path) -> Conjunto:
    pasta = Path(pasta)
    X = np.loadtxt(_achar(pasta, "wpc50.dat"))
    with open(_achar(pasta, "Names.txt"), encoding="latin-1") as f:
        nomes = [linha.strip() for linha in f]   # tira espaços e o \r do Windows
    if X.shape[0] != len(nomes):
        raise ValueError(f"{X.shape[0]} padrões, mas {len(nomes)} nomes")
    genero = np.array([n.split("_")[0] for n in nomes])
    return Conjunto(X=X, nomes=nomes, genero=genero)
