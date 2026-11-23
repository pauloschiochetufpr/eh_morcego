import numpy as np

from .config import Config
from .dados import Conjunto
from .referencia import Especialista

NAO_ROTULADO = -1


def base_inicial(conj: Conjunto, esp: Especialista, cfg: Config,
                 rng: np.random.Generator):
    y = np.full(conj.n, NAO_ROTULADO)
    pos0 = np.where(np.isin(conj.genero, cfg.generos_semente))[0]
    y[pos0] = esp.conferir(pos0)

    candidatos = np.setdiff1d(np.arange(conj.n), pos0)
    neg0 = rng.choice(candidatos, cfg.n_negativos_iniciais, replace=False)
    y[neg0] = esp.conferir(neg0)
    return y, pos0, neg0
