from pathlib import Path

import matplotlib

matplotlib.use("Agg")                    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CINZA, VERMELHO = "#b8b8b8", "#c0392b"


def pca(X, classe, arquivo: Path) -> list:
    Z = StandardScaler().fit_transform(X)
    p = PCA(2).fit(Z)
    Z = p.transform(Z)
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    ax.scatter(Z[classe == 0, 0], Z[classe == 0, 1], s=4, c=CINZA, label="Outros")
    ax.scatter(Z[classe == 1, 0], Z[classe == 1, 1], s=22, c=VERMELHO,
               edgecolor="k", lw=.3, label="Morcegos")
    ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"Projeção PCA das {len(classe)} instâncias")
    ax.legend()
    fig.tight_layout()
    fig.savefig(arquivo)
    plt.close(fig)
    return p.explained_variance_ratio_.tolist()


def ciclos(historico: list, n_semente: int, arquivo: Path):
    cs = [h["ciclo"] for h in historico]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.bar(cs, [h["positivos"] for h in historico], color=VERMELHO)
    for c, h in zip(cs, historico):
        ax.text(c, h["positivos"] + 1, str(h["positivos"]), ha="center", fontsize=9)
    ax.axhline(n_semente, ls="--", c="k", lw=.8)
    ax.text(0.55, n_semente + 1, f"semente ({n_semente})", fontsize=8)
    ax.set_xlabel("Ciclo")
    ax.set_ylabel("Morcegos na base rotulada")
    ax.set_title("Enriquecimento da base a cada ciclo do SVM")
    ax.set_xticks(cs)
    fig.tight_layout()
    fig.savefig(arquivo)
    plt.close(fig)


def separacao(score, classe, arquivo: Path):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.hist(score[classe == 0], bins=80, color=CINZA, label="Outros")
    ax.hist(score[classe == 1], bins=30, color=VERMELHO, label="Morcegos")
    ax.axvline(0, c="k", lw=.8)
    ax.set_xlabel("Valor da função de decisão do SVM")
    ax.set_ylabel("Instâncias")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Separação entre as classes")
    fig.tight_layout()
    fig.savefig(arquivo)
    plt.close(fig)


def gerar_todas(pasta: Path, X, classe, score, historico, n_semente) -> list:
    pasta = Path(pasta)
    var = pca(X, classe, pasta / "fig_pca.png")
    ciclos(historico, n_semente, pasta / "fig_ciclos.png")
    separacao(score, np.asarray(classe), pasta / "fig_scores.png")
    return var
