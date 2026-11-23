import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import Config
from .modelo import novo_svm

MEDIDAS = ["sensibilidade", "precisao", "acuracia", "especificidade", "f1"]


def metricas(y_real, y_prev) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_real, y_prev, labels=[0, 1]).ravel()
    return dict(
        TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
        sensibilidade=tp / (tp + fn) if tp + fn else 0.0,
        precisao=tp / (tp + fp) if tp + fp else 0.0,
        acuracia=(tp + tn) / len(y_real),
        especificidade=tn / (tn + fp) if tn + fp else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0,
    )


def busca_em_grade(X, y, cfg: Config) -> tuple:
    """Kernel RBF x linear, C de 0,1 a 100, métrica F1, CV 5-fold."""
    grade = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        {"svc__kernel": ["rbf", "linear"], "svc__C": [0.1, 1, 10, 100]},
        scoring="f1",
        cv=StratifiedKFold(5, shuffle=True, random_state=cfg.semente),
    ).fit(X, y)
    tabela = [dict(kernel=p["svc__kernel"], C=p["svc__C"], f1=float(s))
              for p, s in zip(grade.cv_results_["params"],
                              grade.cv_results_["mean_test_score"])]
    return tabela, grade.best_params_


def validacao_cruzada(X, y, cfg: Config) -> dict:
    """10-fold estratificada repetida 10x: média e desvio de cada medida,
    mais as predições agregadas de uma rodada 10-fold."""
    rskf = RepeatedStratifiedKFold(n_splits=cfg.cv_folds,
                                   n_repeats=cfg.cv_repeticoes,
                                   random_state=cfg.semente)
    folds = []
    for tr, te in rskf.split(X, y):
        m = novo_svm(cfg.svm_C, cfg.svm_kernel).fit(X[tr], y[tr])
        folds.append(metricas(y[te], m.predict(X[te])))
    media = {k: float(np.mean([f[k] for f in folds])) for k in MEDIDAS}
    dp = {k: float(np.std([f[k] for f in folds])) for k in MEDIDAS}

    pred = cross_val_predict(
        novo_svm(cfg.svm_C, cfg.svm_kernel), X, y,
        cv=StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.semente))
    return dict(media=media, dp=dp, agregado=metricas(y, pred))


def holdout(X, y, cfg: Config, nomes=None) -> dict:
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=cfg.holdout_teste, stratify=y,
                              random_state=cfg.semente)
    prev = novo_svm(cfg.svm_C, cfg.svm_kernel).fit(X[tr], y[tr]).predict(X[te])
    res = metricas(y[te], prev)
    if nomes is not None:
        res["morcegos_perdidos"] = [nomes[i] for i in te[(prev == 0) & (y[te] == 1)]]
        res["falsos_morcegos"] = [nomes[i] for i in te[(prev == 1) & (y[te] == 0)]]
    return res


def efeito_do_enriquecimento(X, classe, gabarito, semente_idx, cfg: Config) -> dict:
    dentro = np.zeros(len(classe), bool)
    dentro[semente_idx] = True
    m = novo_svm(cfg.svm_C, cfg.svm_kernel).fit(X[dentro], gabarito[dentro])
    return metricas(classe[~dentro], m.predict(X[~dentro]))
