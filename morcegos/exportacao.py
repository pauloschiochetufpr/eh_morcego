import json
from pathlib import Path

import numpy as np


def salvar_tudo(pasta: Path, classe, score, nomes, resumo: dict, indice_base: int = 1):
    """
    classes.txt               uma classe por linha (1 = morcego, 0 = outro)
    indices_morcegos.txt      índices dos positivos, um por linha
    morcegos_encontrados.csv  índice; nome; pontuação do SVM
    resultado_completo.csv    índice; nome; classe; pontuação, das N instâncias
    resultados.json           números do experimento (ciclos, métricas, grade)
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    positivos = np.where(classe == 1)[0]

    np.savetxt(pasta / "classes.txt", classe, fmt="%d")
    np.savetxt(pasta / "indices_morcegos.txt", positivos + indice_base, fmt="%d")

    with open(pasta / "morcegos_encontrados.csv", "w", encoding="utf-8") as f:
        f.write("indice;nome;score_svm\n")
        for i in positivos:
            f.write(f"{i + indice_base};{nomes[i]};{score[i]:.4f}\n")

    with open(pasta / "resultado_completo.csv", "w", encoding="utf-8") as f:
        f.write("indice;nome;classe;score_svm\n")
        for i in range(len(classe)):
            f.write(f"{i + indice_base};{nomes[i]};{classe[i]};{score[i]:.4f}\n")

    with open(pasta / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=1, default=_json)


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
