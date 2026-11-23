from pathlib import Path

import numpy as np


class Especialista:
    def __init__(self, caminho_lista: Path, genero: np.ndarray):
        self.generos = set(Path(caminho_lista).read_text(encoding="utf-8").split())
        self._genero = genero
        self.conferidos = set()        

    def e_morcego(self, i: int) -> int:
        self.conferidos.add(int(i))
        return int(self._genero[i] in self.generos)

    def conferir(self, indices) -> np.ndarray:
        return np.array([self.e_morcego(i) for i in indices], dtype=int)

    def gabarito(self) -> np.ndarray:
        return np.isin(self._genero, list(self.generos)).astype(int)
