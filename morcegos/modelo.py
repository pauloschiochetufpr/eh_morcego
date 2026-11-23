from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def novo_svm(C: float = 10.0, kernel: str = "rbf"):
    return make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma="scale", class_weight="balanced"),
    )
