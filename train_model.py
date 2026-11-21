
import csv
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
)

from features import (
    prepare_features,
    BASIC_FEATURES,
    ALL_STRUCTURAL_FEATURES,
)


# ==========================================================
# 1. НАСТРОЙКИ И ПУТИ
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DATA_PATH = DATA_DIR / "sms.tsv"
TEST_PATH = DATA_DIR / "test.tsv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTER_FOLDS = 5
INNER_FOLDS = 3
CALIBRATION_FOLDS = 3

N_JOBS = 1

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================================
# 2. КОНФИГУРАЦИИ ПРИЗНАКОВ И МОДЕЛЕЙ
# ==========================================================

FEATURE_CONFIGS = {
    "tfidf_only": {
        "use_tfidf": True,
        "structural_features": [],
    },

    "structural_only": {
        "use_tfidf": False,
        "structural_features": ALL_STRUCTURAL_FEATURES,
    },

    "tfidf_basic": {
        "use_tfidf": True,
        "structural_features": BASIC_FEATURES,
    },

    "tfidf_all": {
        "use_tfidf": True,
        "structural_features": ALL_STRUCTURAL_FEATURES,
    },
}

MODEL_NAMES = [
    "LogisticRegression",
    "LinearSVC",
    "MultinomialNB",
]


# ==========================================================
# 3. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ
# ==========================================================

def read_tsv(path):
    """Читает TSV-файл с метками ham и spam."""

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        on_bad_lines="error",
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    label_mapping = {
        "ham": 0,
        "spam": 1,
    }

    df["label"] = df["label"].map(label_mapping)

    df = df.dropna(subset=["label"])
    df = df[df["text"] != ""]

    df["label"] = df["label"].astype(int)

    return df.reset_index(drop=True)


def load_data(path):
    """Загружает датасет, удаляет дубликаты и конфликты меток."""

    df = read_tsv(path)

    # Удаляем одинаковые сообщения с одинаковыми метками.
    df = df.drop_duplicates(
        subset=["text", "label"]
    )

    # Ищем сообщения с разными метками.
    label_counts = df.groupby("text")["label"].nunique()

    conflicting_texts = label_counts[
        label_counts > 1
    ].index

    if len(conflicting_texts) > 0:

        print(
            "Исключено сообщений с противоречивой разметкой:",
            len(conflicting_texts),
        )

        df = df[
            ~df["text"].isin(conflicting_texts)
        ]

    return df.reset_index(drop=True)


# ==========================================================
# 4. СОХРАНЕНИЕ НЕЗАВИСИМОЙ ТЕСТОВОЙ ВЫБОРКИ
# ==========================================================

def save_test_data(test_df):
    """Сохраняет тестовую выборку в отдельный TSV-файл."""

    with open(
        TEST_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file,
            delimiter="\t",
        )

        for row in test_df.itertuples(index=False):

            label = (
                "spam"
                if row.label == 1
                else "ham"
            )

            writer.writerow([
                label,
                row.text,
            ])


def prepare_train_test_data(df):
    """
    При первом запуске создаёт тестовую выборку.

    При последующих запусках использует ту же выборку,
    исключая её сообщения из обучения.
    """

    if TEST_PATH.exists():

        print(
            "\nИспользуется ранее сохранённая тестовая выборка."
        )

        test_df = load_data(TEST_PATH)

        test_texts = set(test_df["text"])

        train_df = df[
            ~df["text"].isin(test_texts)
        ].copy()

    else:

        print(
            "\nСоздаётся первоначальная тестовая выборка."
        )

        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            stratify=df["label"],
            random_state=RANDOM_STATE,
        )

        save_test_data(test_df)

        train_df = train_df.copy()
        test_df = test_df.copy()

    if train_df.empty or test_df.empty:

        raise ValueError(
            "Обучающая или тестовая выборка оказалась пустой."
        )

    if train_df["label"].nunique() < 2:

        raise ValueError(
            "В обучающей выборке должны быть оба класса."
        )

    if test_df["label"].nunique() < 2:

        raise ValueError(
            "В тестовой выборке должны быть оба класса."
        )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ==========================================================
# 5. СОЗДАНИЕ МОДЕЛЕЙ
# ==========================================================

def create_model(model_name):

    if model_name == "LogisticRegression":

        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "LinearSVC":

        return LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "MultinomialNB":

        return MultinomialNB()

    raise ValueError(
        f"Неизвестная модель: {model_name}"
    )


# ==========================================================
# 6. СОЗДАНИЕ PIPELINE
# ==========================================================

def create_base_pipeline(feature_config, model_name):

    config = FEATURE_CONFIGS[feature_config]

    transformers = []

    # Текстовые признаки TF-IDF.
    if config["use_tfidf"]:

        transformers.append((
            "tfidf",
            TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
            ),
            "clean_text",
        ))

    # Структурные признаки.
    structural_features = config["structural_features"]

    if structural_features:

        if model_name == "MultinomialNB":

            # NB требует неотрицательные значения.
            scaler = MinMaxScaler(clip=True)

        else:

            scaler = StandardScaler()

        transformers.append((
            "struct",
            scaler,
            structural_features,
        ))

    feature_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    pipeline = Pipeline([
        ("features", feature_transformer),
        ("classifier", create_model(model_name)),
    ])

    return pipeline


def create_pipeline(feature_config, model_name):

    pipeline = create_base_pipeline(
        feature_config,
        model_name,
    )

    if model_name == "LinearSVC":

        # Калибруем весь Pipeline,
        # включая TF-IDF и масштабирование.
        return CalibratedClassifierCV(
            estimator=pipeline,
            method="sigmoid",
            cv=CALIBRATION_FOLDS,
        )

    return pipeline


# ==========================================================
# 7. ПОДБОР ПОРОГА КЛАССИФИКАЦИИ
# ==========================================================

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5

    # Precision и recall содержат на один элемент больше,
    # чем thresholds.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (
        2 * precision * recall
        / (precision + recall + 1e-10)
    )

    best_indices = np.flatnonzero(
        f1_values == np.max(f1_values)
    )

    # При одинаковом F1 выбираем более высокий порог.
    best_index = best_indices[-1]

    return float(thresholds[best_index])


def predict_with_threshold(probabilities, threshold):

    return (
        probabilities >= threshold
    ).astype(int)


# ==========================================================
# 8. РАСЧЁТ МЕТРИК
# ==========================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities,
):

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
    }


# ==========================================================
# 9. ВЛОЖЕННАЯ КРОСС-ВАЛИДАЦИЯ
# ==========================================================

def evaluate_nested_cv(
    X,
    y,
    feature_config,
    model_name,
):
    """
    Внешний цикл оценивает качество модели.

    Внутренний цикл подбирает порог по OOF-прогнозам.

    Внешняя validation-часть не участвует
    в подборе порога.
    """

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):

        X_outer_train = X.iloc[train_idx]
        X_outer_val = X.iloc[val_idx]

        y_outer_train = y.iloc[train_idx]
        y_outer_val = y.iloc[val_idx]

        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold,
        )

        inner_model = create_pipeline(
            feature_config,
            model_name,
        )

        # OOF-вероятности на внешней обучающей части.
        oof_probabilities = cross_val_predict(
            inner_model,
            X_outer_train,
            y_outer_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=N_JOBS,
        )[:, 1]

        threshold = find_best_threshold(
            y_outer_train,
            oof_probabilities,
        )

        # Обучение на всей внешней train-части.
        outer_model = create_pipeline(
            feature_config,
            model_name,
        )

        outer_model.fit(
            X_outer_train,
            y_outer_train,
        )

        # Оценка на внешней validation-части.
        val_probabilities = outer_model.predict_proba(
            X_outer_val
        )[:, 1]

        val_predictions = predict_with_threshold(
            val_probabilities,
            threshold,
        )

        metrics = calculate_metrics(
            y_outer_val,
            val_predictions,
            val_probabilities,
        )

        fold_scores.append(metrics)

        print(
            f"    Fold {fold}/{OUTER_FOLDS}: "
            f"F1 = {metrics['f1']:.4f}; "
            f"threshold = {threshold:.4f}"
        )

    scores_df = pd.DataFrame(fold_scores)

    return {
        "mean_f1": scores_df["f1"].mean(),
        "std_f1": scores_df["f1"].std(),
        "mean_precision": scores_df["precision"].mean(),
        "mean_recall": scores_df["recall"].mean(),
        "mean_accuracy": scores_df["accuracy"].mean(),
        "mean_roc_auc": scores_df["roc_auc"].mean(),
    }


# ==========================================================
# 10. ЭКСПЕРИМЕНТЫ С ПРИЗНАКАМИ И МОДЕЛЯМИ
# ==========================================================

def run_feature_experiments(X_train, y_train):

    results = []

    total = (
        len(FEATURE_CONFIGS)
        * len(MODEL_NAMES)
    )

    current = 0

    for feature_config in FEATURE_CONFIGS:

        for model_name in MODEL_NAMES:

            current += 1

            print("\n" + "=" * 60)

            print(
                f"Эксперимент {current}/{total}: "
                f"{feature_config} + {model_name}"
            )

            print("=" * 60)

            metrics = evaluate_nested_cv(
                X_train,
                y_train,
                feature_config,
                model_name,
            )

            results.append({
                "feature_config": feature_config,
                "model": model_name,
                **metrics,
            })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["mean_f1", "std_f1"],
        ascending=[False, True],
    )

    results_df.to_csv(
        RESULTS_DIR / "feature_experiments.csv",
        index=False,
    )

    return results_df


# ==========================================================
# 11. ПОДБОР ГИПЕРПАРАМЕТРОВ
# ==========================================================

def get_param_grid(model_name):

    if model_name == "LogisticRegression":

        return {
            "classifier__C": [
                0.1,
                1.0,
                10.0,
            ],
        }

    if model_name == "LinearSVC":

        return {
            "estimator__classifier__C": [
                0.1,
                1.0,
                10.0,
            ],
        }

    if model_name == "MultinomialNB":

        return {
            "classifier__alpha": [
                0.1,
                0.5,
                1.0,
            ],
        }

    raise ValueError(
        f"Неизвестная модель: {model_name}"
    )


def tune_model(
    X_train,
    y_train,
    feature_config,
    model_name,
):
    """
    Подбирает гиперпараметры по ROC-AUC.

    ROC-AUC не зависит от выбранного порога.
    Порог для F1 подбирается отдельно.
    """

    cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipeline = create_pipeline(
        feature_config,
        model_name,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=get_param_grid(model_name),
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=N_JOBS,
        error_score="raise",
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    print(
        "Лучшие параметры:",
        grid_search.best_params_,
    )

    print(
        f"CV ROC-AUC: {grid_search.best_score_:.4f}"
    )

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
    )


# ==========================================================
# 12. ФИНАЛЬНОЕ ОБУЧЕНИЕ И ОЦЕНКА
# ==========================================================

def train_final_model(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_config,
    model_name,
):
    """
    1. Подбирает гиперпараметры на train.
    2. Получает OOF-вероятности.
    3. Подбирает порог на train.
    4. Оценивает модель на независимом test.
    """

    final_model, best_params = tune_model(
        X_train,
        y_train,
        feature_config,
        model_name,
    )

    # Модель с выбранными параметрами
    # для получения OOF-вероятностей.
    threshold_model = create_pipeline(
        feature_config,
        model_name,
    )

    threshold_model.set_params(
        **best_params
    )

    cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = cross_val_predict(
        threshold_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=N_JOBS,
    )[:, 1]

    threshold = find_best_threshold(
        y_train,
        oof_probabilities,
    )

    # GridSearchCV с refit=True уже обучил
    # лучшую модель на всей train-выборке.
    test_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = predict_with_threshold(
        test_probabilities,
        threshold,
    )

    metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities,
    )

    return {
        "model": final_model,
        "threshold": threshold,
        "probabilities": test_probabilities,
        "predictions": test_predictions,
        "metrics": metrics,
        "best_params": best_params,
    }


# ==========================================================
# 13. МАТРИЦЫ ОШИБОК
# ==========================================================

def save_confusion_matrices(results, y_test):

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(5 * len(results), 4),
    )

    for ax, (model_name, result) in zip(
        axes,
        results.items(),
    ):

        cm = confusion_matrix(
            y_test,
            result["predictions"],
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["ham", "spam"],
        )

        display.plot(
            ax=ax,
            colorbar=False,
            values_format="d",
        )

        ax.set_title(model_name)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=150,
    )

    plt.close()


# ==========================================================
# 14. ВАЖНОСТЬ ПРИЗНАКОВ
# ==========================================================

def get_importance_pipeline(
    model_name,
    feature_config,
    best_params,
    final_model,
    X_train,
    y_train,
):
    """
    Получает Pipeline для анализа коэффициентов.

    Для LinearSVC создаётся отдельная диагностическая
    модель без калибровки. Она нужна только для графика.

    В приложение сохраняется калиброванная модель.
    """

    if model_name != "LinearSVC":
        return final_model

    diagnostic_pipeline = create_base_pipeline(
        feature_config,
        "LinearSVC",
    )

    # Передаём подобранное значение C.
    diagnostic_pipeline.set_params(
        **{
            "classifier__C":
                best_params["estimator__classifier__C"]
        }
    )

    diagnostic_pipeline.fit(
        X_train,
        y_train,
    )

    return diagnostic_pipeline


def save_feature_importance(
    pipeline,
    model_name,
):

    feature_transformer = pipeline.named_steps[
        "features"
    ]

    feature_names = (
        feature_transformer.get_feature_names_out()
    )

    classifier = pipeline.named_steps[
        "classifier"
    ]

    if isinstance(classifier, LogisticRegression):

        coefficients = classifier.coef_[0]

    elif isinstance(classifier, LinearSVC):

        coefficients = classifier.coef_[0]

    elif isinstance(classifier, MultinomialNB):

        coefficients = (
            classifier.feature_log_prob_[1]
            - classifier.feature_log_prob_[0]
        )

    else:

        print(
            "Не удалось получить коэффициенты модели."
        )

        return

    if len(feature_names) != len(coefficients):

        print(
            "Количество имён признаков "
            "не совпадает с количеством коэффициентов."
        )

        return

    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "importance": np.abs(coefficients),
    })

    top_features = importance.nlargest(
        20,
        "importance",
    ).sort_values("coefficient")

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["feature"],
        top_features["coefficient"],
    )

    plt.axvline(0, linewidth=0.8)

    plt.title(
        f"Top 20 features: {model_name}"
    )

    plt.xlabel(
        "Coefficient / log probability difference"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "feature_importance.png",
        dpi=150,
    )

    plt.close()


# ==========================================================
# 15. БЕЗОПАСНОЕ СОХРАНЕНИЕ АРТЕФАКТОВ
# ==========================================================

def save_artifacts(
    model,
    threshold,
    X_test,
):
    """
    Сначала сохраняет кандидатов во временную папку,
    проверяет их и только потом заменяет рабочие файлы.

    При ошибке замены восстанавливает старые артефакты.
    """

    candidate_dir = MODELS_DIR / "candidate"
    backup_dir = MODELS_DIR / "backup"

    final_model_path = MODELS_DIR / "model.pkl"
    final_threshold_path = MODELS_DIR / "threshold.pkl"

    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    candidate_dir.mkdir()
    backup_dir.mkdir()

    candidate_model = candidate_dir / "model.pkl"
    candidate_threshold = candidate_dir / "threshold.pkl"

    old_model_exists = final_model_path.exists()
    old_threshold_exists = final_threshold_path.exists()

    model_replaced = False
    threshold_replaced = False

    try:

        # Сохраняем новые артефакты.
        joblib.dump(
            model,
            candidate_model,
        )

        joblib.dump(
            float(threshold),
            candidate_threshold,
        )

        # Проверяем возможность загрузки.
        check_model = joblib.load(candidate_model)

        check_threshold = joblib.load(
            candidate_threshold
        )

        if not hasattr(check_model, "predict_proba"):

            raise RuntimeError(
                "Модель не поддерживает predict_proba."
            )

        if not 0 <= check_threshold <= 1:

            raise RuntimeError(
                "Некорректный порог классификации."
            )

        # Проверяем реальные предсказания.
        sample = X_test.iloc[:min(5, len(X_test))]

        check_probabilities = check_model.predict_proba(
            sample
        )

        if (
            check_probabilities.ndim != 2
            or check_probabilities.shape[1] != 2
            or not np.isfinite(check_probabilities).all()
        ):

            raise RuntimeError(
                "Проверка предсказаний модели не пройдена."
            )

        # Создаём резервные копии старых файлов.
        if old_model_exists:

            shutil.copy2(
                final_model_path,
                backup_dir / "model.pkl",
            )

        if old_threshold_exists:

            shutil.copy2(
                final_threshold_path,
                backup_dir / "threshold.pkl",
            )

        # Заменяем рабочие файлы.
        os.replace(
            candidate_model,
            final_model_path,
        )

        model_replaced = True

        os.replace(
            candidate_threshold,
            final_threshold_path,
        )

        threshold_replaced = True

    except Exception:

        # Восстанавливаем предыдущую модель.
        if old_model_exists:

            shutil.copy2(
                backup_dir / "model.pkl",
                final_model_path,
            )

        elif model_replaced and final_model_path.exists():

            final_model_path.unlink()

        # Восстанавливаем предыдущий порог.
        if old_threshold_exists:

            shutil.copy2(
                backup_dir / "threshold.pkl",
                final_threshold_path,
            )

        elif threshold_replaced and final_threshold_path.exists():

            final_threshold_path.unlink()

        raise

    finally:

        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)

        if backup_dir.exists():
            shutil.rmtree(backup_dir)


# ==========================================================
# 16. ГЛАВНАЯ ФУНКЦИЯ
# ==========================================================

def main():

    print("\nЗагрузка данных...")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Не найден датасет: {DATA_PATH}"
        )

    df = load_data(DATA_PATH)

    print(
        f"Всего уникальных сообщений: {len(df)}"
    )

    print(
        f"Ham: {(df['label'] == 0).sum()}"
    )

    print(
        f"Spam: {(df['label'] == 1).sum()}"
    )

    # Получаем обучающую и тестовую выборки.
    train_df, test_df = prepare_train_test_data(df)

    print("\nРазмер train:", len(train_df))
    print("Размер test:", len(test_df))

    print(
        "Ham в train:",
        (train_df["label"] == 0).sum(),
    )

    print(
        "Spam в train:",
        (train_df["label"] == 1).sum(),
    )

    print(
        "Ham в test:",
        (test_df["label"] == 0).sum(),
    )

    print(
        "Spam в test:",
        (test_df["label"] == 1).sum(),
    )

    # Извлекаем признаки.
    X_train = prepare_features(
        train_df["text"]
    )

    X_test = prepare_features(
        test_df["text"]
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    # Проверяем достаточное количество объектов
    # для вложенной кросс-валидации и калибровки.
    min_class_count = y_train.value_counts().min()

    if min_class_count < 10:

        raise ValueError(
            "Недостаточно объектов одного из классов "
            "для выбранной схемы кросс-валидации."
        )

    # ------------------------------------------------------
    # ЭКСПЕРИМЕНТЫ: 4 НАБОРА ПРИЗНАКОВ x 3 МОДЕЛИ
    # ------------------------------------------------------

    print("\nЗапуск экспериментов...")

    experiments_df = run_feature_experiments(
        X_train,
        y_train,
    )

    print("\nРезультаты экспериментов:")

    print(
        experiments_df.to_string(index=False)
    )

    # Выбираем комбинацию по mean F1 nested CV.
    best_result = experiments_df.iloc[0]

    best_feature_config = best_result[
        "feature_config"
    ]

    best_model_name = best_result[
        "model"
    ]

    print("\nЛучшая комбинация по nested CV:")

    print(
        "Набор признаков:",
        best_feature_config,
    )

    print(
        "Модель:",
        best_model_name,
    )

    # ------------------------------------------------------
    # ФИНАЛЬНОЕ СРАВНЕНИЕ ТРЁХ МОДЕЛЕЙ
    # ------------------------------------------------------

    print("\nФинальное сравнение моделей...")

    comparison_results = {}

    for model_name in MODEL_NAMES:

        print("\n" + "-" * 50)

        print(
            "Обучение:",
            model_name,
        )

        result = train_final_model(
            X_train,
            y_train,
            X_test,
            y_test,
            best_feature_config,
            model_name,
        )

        comparison_results[model_name] = result

        print(
            f"Threshold: {result['threshold']:.4f}"
        )

        print(
            f"Test F1: {result['metrics']['f1']:.4f}"
        )

    # ------------------------------------------------------
    # СОХРАНЕНИЕ ТАБЛИЦЫ СРАВНЕНИЯ
    # ------------------------------------------------------

    comparison_rows = []

    for model_name, result in comparison_results.items():

        comparison_rows.append({
            "model": model_name,
            "feature_config": best_feature_config,
            "threshold": result["threshold"],
            **result["metrics"],
            "best_params": str(result["best_params"]),
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nСравнение моделей на тестовой выборке:")

    print(
        comparison_df.to_string(index=False)
    )

    # ------------------------------------------------------
    # СОХРАНЕНИЕ ВЫБРАННОЙ МОДЕЛИ
    # ------------------------------------------------------

    # Выбор модели основан на nested CV,
    # а не на тестовых метриках.
    selected_result = comparison_results[
        best_model_name
    ]

    save_artifacts(
        selected_result["model"],
        selected_result["threshold"],
        X_test,
    )

    # ------------------------------------------------------
    # СОХРАНЕНИЕ ГРАФИКОВ
    # ------------------------------------------------------

    save_confusion_matrices(
        comparison_results,
        y_test,
    )

    importance_pipeline = get_importance_pipeline(
        best_model_name,
        best_feature_config,
        selected_result["best_params"],
        selected_result["model"],
        X_train,
        y_train,
    )

    save_feature_importance(
        importance_pipeline,
        best_model_name,
    )

    # ------------------------------------------------------
    # ИТОГОВЫЙ ОТЧЁТ
    # ------------------------------------------------------

    print("\nИтоговый отчёт выбранной модели:")

    print(
        classification_report(
            y_test,
            selected_result["predictions"],
            labels=[0, 1],
            target_names=["ham", "spam"],
            zero_division=0,
        )
    )

    print("\nОбучение завершено.")

    print("\nСохранённые файлы:")

    print("Модель: models/model.pkl")
    print("Порог: models/threshold.pkl")

    print("Тестовая выборка: data/test.tsv")

    print(
        "Эксперименты: results/feature_experiments.csv"
    )

    print(
        "Сравнение: results/model_comparison.csv"
    )

    print(
        "Матрицы ошибок: results/confusion_matrix.png"
    )

    print(
        "Важность признаков: results/feature_importance.png"
    )


if __name__ == "__main__":
    main()
