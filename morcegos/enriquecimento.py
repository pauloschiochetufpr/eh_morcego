import numpy as np

from .config import Config
from .dados import Conjunto
from .modelo import novo_svm
from .referencia import Especialista
from .supervisao import NAO_ROTULADO


def enriquecer(conj: Conjunto, y: np.ndarray, esp: Especialista, cfg: Config,
               rng: np.random.Generator, log=print) -> list:
    """Altera y no lugar e devolve o histórico dos ciclos."""
    historico = []
    ciclo = 0
    while True:
        ciclo += 1
        rot = y != NAO_ROTULADO
        svm = novo_svm(cfg.svm_C, cfg.svm_kernel).fit(conj.X[rot], y[rot])

        nao_rot = np.where(~rot)[0]
        score = svm.decision_function(conj.X[nao_rot])
        previstos_pos = int(np.sum(score > 0))
        candidatos = nao_rot[np.argsort(-score)][:cfg.candidatos_por_ciclo]

        y[candidatos] = esp.conferir(candidatos)
        novos_pos = [int(i) for i in candidatos if y[i] == 1]
        novos_neg = [int(i) for i in candidatos if y[i] == 0]

        resto = np.where(y == NAO_ROTULADO)[0]
        extra = rng.choice(resto, cfg.aleatorios_por_ciclo, replace=False)
        y[extra] = esp.conferir(extra)
        extra_pos = int(np.sum(y[extra] == 1))

        historico.append(dict(
            ciclo=ciclo,
            positivos=int(np.sum(y == 1)),
            negativos=int(np.sum(y == 0)),
            previstos_pos_nao_rotulados=previstos_pos,
            novos_morcegos=len(novos_pos) + extra_pos,
            falsos_candidatos=len(novos_neg),
            exemplos_novos=[conj.nomes[i] for i in novos_pos][:8],
            negativos_dificeis=[conj.nomes[i] for i in novos_neg][:6],
        ))
        log(f"  ciclo {ciclo}: +{len(novos_pos)} morcegos, {len(novos_neg)} "
            f"candidatos rejeitados, {previstos_pos} previstos positivos fora da base")

        if len(novos_pos) == 0 and previstos_pos <= len(novos_neg):
            return historico


def classificar_final(conj: Conjunto, y: np.ndarray, cfg: Config):
    rot = y != NAO_ROTULADO
    svm = novo_svm(cfg.svm_C, cfg.svm_kernel).fit(conj.X[rot], y[rot])
    classe = y.copy()
    nao_rot = np.where(~rot)[0]
    classe[nao_rot] = svm.predict(conj.X[nao_rot])
    return classe, svm.decision_function(conj.X)
