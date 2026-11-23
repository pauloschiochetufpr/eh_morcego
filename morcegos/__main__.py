"""
Ponto de entrada:  python -m morcegos  [--dados PASTA] [--saida PASTA] [--indice-base 0|1]

Executa o pipeline completo, na ordem:
  1. dados -> 2. supervisão inicial -> 3. enriquecimento com SVM ->
  4. classificação final -> 5. testes de desempenho -> 6. exportação e figuras
"""
import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from . import avaliacao, dados, enriquecimento, exportacao, graficos, supervisao
from .config import Config
from .referencia import Especialista


def _pct(x, casas=2):
    return f"{x * 100:.{casas}f}%".replace(".", ",")


def argumentos() -> Config:
    p = argparse.ArgumentParser(prog="python -m morcegos",
                                description="É morcego? Classificação com SVM.")
    p.add_argument("--dados", type=Path, default=Config.pasta_dados,
                   help="pasta com wpc50.dat, Names.txt e generos_chiroptera.txt")
    p.add_argument("--saida", type=Path, default=Config.pasta_saida,
                   help="pasta onde os resultados são gravados")
    p.add_argument("--indice-base", type=int, choices=[0, 1], default=Config.indice_base,
                   help="índices exportados a partir de 1 (padrão) ou de 0")
    p.add_argument("--semente", type=int, default=Config.semente)
    a = p.parse_args()
    return Config(pasta_dados=a.dados, pasta_saida=a.saida,
                  indice_base=a.indice_base, semente=a.semente)


def main():
    cfg = argumentos()
    rng = np.random.default_rng(cfg.semente)
    t0 = time.time()

    print("[1/6] Lendo os dados")
    conj = dados.carregar(cfg.pasta_dados)
    esp = Especialista(cfg.pasta_dados / "generos_chiroptera.txt", conj.genero)
    print(f"  {conj.n} instâncias x {conj.X.shape[1]} características")

    print("[2/6] Supervisão inicial")
    y, pos0, neg0 = supervisao.base_inicial(conj, esp, cfg, rng)
    n_sem_pos, n_sem_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    print(f"  {n_sem_pos} positivos ({len(pos0)} dos gêneros da pesquisa), "
          f"{n_sem_neg} negativos")

    print("[3/6] Enriquecimento iterativo com SVM")
    historico = enriquecimento.enriquecer(conj, y, esp, cfg, rng)

    print("[4/6] Classificação final")
    classe, score = enriquecimento.classificar_final(conj, y, cfg)
    n_rot = int(np.sum(y >= 0))
    gabarito = esp.gabarito()
    conferencia = confusion_matrix(gabarito, classe).ravel().tolist()
    print(f"  {n_rot} instâncias conferidas, {conj.n - n_rot} classificadas pelo SVM")
    print(f"  morcegos encontrados: {int(classe.sum())}")

    print("[5/6] Testes de desempenho")
    grade, melhor = avaliacao.busca_em_grade(conj.X, classe, cfg)
    cv = avaliacao.validacao_cruzada(conj.X, classe, cfg)
    ho = avaliacao.holdout(conj.X, classe, cfg, conj.nomes)
    efeito = avaliacao.efeito_do_enriquecimento(
        conj.X, classe, gabarito, np.concatenate([pos0, neg0]), cfg)
    m = cv["media"]
    print(f"  CV {cfg.cv_folds}-fold x{cfg.cv_repeticoes}: sensibilidade {_pct(m['sensibilidade'])}, "
          f"precisão {_pct(m['precisao'])}, acurácia {_pct(m['acuracia'], 3)}")
    print(f"  holdout {int(cfg.holdout_teste * 100)}%: sensibilidade {_pct(ho['sensibilidade'])}, "
          f"precisão {_pct(ho['precisao'])}, acurácia {_pct(ho['acuracia'])}"
          + (f"  (perdeu: {', '.join(ho['morcegos_perdidos'])})" if ho["morcegos_perdidos"] else ""))
    print(f"  só a semente: precisão {_pct(efeito['precisao'])} -> com os ciclos: {_pct(m['precisao'])}")

    print("[6/6] Exportando resultados e figuras")
    cfg.pasta_saida.mkdir(parents=True, exist_ok=True)
    var_pca = graficos.gerar_todas(cfg.pasta_saida, conj.X, classe, score, historico, n_sem_pos)
    positivos = np.where(classe == 1)[0]
    resumo = dict(
        semente_pos=len(pos0), semente_pos_total=n_sem_pos, semente_neg=n_sem_neg,
        semente_generos=list(cfg.generos_semente), historico=historico,
        rotulados=n_rot, previstos=conj.n - n_rot, total_morcegos=int(classe.sum()),
        conf_final_vs_ref=conferencia, grade=grade, melhor=melhor,
        cv_media=cv["media"], cv_dp=cv["dp"], cv_agregado=cv["agregado"],
        holdout=ho, semente_vs_final=efeito, pca_var=var_pca,
        generos=sorted(set(conj.genero[positivos])), indice_base=cfg.indice_base,
    )
    exportacao.salvar_tudo(cfg.pasta_saida, classe, score, conj.nomes, resumo, cfg.indice_base)
    print(f"  arquivos em {cfg.pasta_saida.resolve()}")
    print(f"Concluído em {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
