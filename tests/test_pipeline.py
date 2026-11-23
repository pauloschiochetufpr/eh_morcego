"""Testes rápidos:  python -m pytest"""
from pathlib import Path

import numpy as np
import pytest

from morcegos import dados, enriquecimento, supervisao
from morcegos.avaliacao import metricas
from morcegos.config import Config
from morcegos.referencia import Especialista

DADOS = Path(__file__).resolve().parents[1] / "dados"


@pytest.fixture(scope="module")
def conj():
    return dados.carregar(DADOS)


def test_dados_alinhados(conj):
    assert conj.X.shape == (8426, 50)
    assert conj.n == 8426
    assert conj.nomes[0] == "Candida_bracarensis"      # sem espaços nem \r


def test_especialista(conj):
    esp = Especialista(DADOS / "generos_chiroptera.txt", conj.genero)
    i_morcego = conj.nomes.index("Myotis_myotis")
    i_outro = conj.nomes.index("Candida_bracarensis")
    assert esp.e_morcego(i_morcego) == 1
    assert esp.e_morcego(i_outro) == 0
    assert esp.conferidos == {i_morcego, i_outro}


def test_metricas():
    m = metricas(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 0, 0, 1]))
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 1, 2)
    assert m["sensibilidade"] == 0.5
    assert m["precisao"] == 0.5
    assert m["acuracia"] == 0.6


def test_pipeline_encontra_79_morcegos(conj):
    cfg = Config(pasta_dados=DADOS)
    rng = np.random.default_rng(cfg.semente)
    esp = Especialista(DADOS / "generos_chiroptera.txt", conj.genero)
    y, _, _ = supervisao.base_inicial(conj, esp, cfg, rng)
    historico = enriquecimento.enriquecer(conj, y, esp, cfg, rng, log=lambda *_: None)
    classe, _ = enriquecimento.classificar_final(conj, y, cfg)
    assert len(historico) == 3
    assert classe.sum() == 79
    assert np.array_equal(classe, esp.gabarito())
