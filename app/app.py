
import csv
import io
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import extract_structural_features, prepare_features


# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "sms.tsv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.pkl"

RESULTS_DIR = BASE_DIR / "results"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
EXPERIMENTS_PATH = RESULTS_DIR / "feature_experiments.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.png"

TRAIN_SCRIPT_PATH = BASE_DIR / "train_model.py"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide",
)

st.title("SMS Spam Detector")
st.caption(
    "Система классификации SMS-сообщений "
    "на нормальные и потенциально мошеннические."
)


# ============================================================
# DATASET HELPERS
# ============================================================

VALID_LABELS = {"spam", "ham"}


def normalize_text(text):
    """Приводит текст к строке и удаляет внешние пробелы."""
    if pd.isna(text):
        return ""

    return str(text).strip()


def read_dataset_records():
    """
    Читает текущий TSV-файл.

    Поддерживает файл с заголовком и без заголовка.
    Возвращает список пар (label, text).
    """
    if not DATA_PATH.exists():
        return []

    records = []

    with open(DATA_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) < 2:
                raise ValueError(
                    f"Некорректная строка {row_number} в sms.tsv: "
                    "ожидаются два столбца — label и text."
                )

            label = row[0].strip().lower()
            text = "\t".join(row[1:]).strip()

            # Пропускаем заголовок, если он присутствует.
            if row_number == 1 and label in {
                "label",
                "v1",
                "target",
                "class",
            }:
                continue

            if label not in VALID_LABELS:
                raise ValueError(
                    f"Неизвестная метка в строке {row_number}: {label}"
                )

            if text:
                records.append((label, text))

    return records


def get_dataset_size():
    """Возвращает количество корректных размеченных сообщений."""
    return len(read_dataset_records())


def check_message_in_dataset(text):
    """
    Проверяет, есть ли сообщение в датасете.

    Возвращает:
    - 'same' — сообщение уже есть с такой же меткой;
    - 'conflict' — сообщение есть, но с другой меткой;
    - None — сообщения ещё нет.
    """
    normalized = normalize_text(text)

    for label, existing_text in read_dataset_records():
        if existing_text == normalized:
            return label

    return None


def add_feedback(text, label):
    """
    Добавляет сообщение с пользовательской меткой в TSV.

    Используется csv.writer, чтобы корректно записывать
    табуляции, кавычки и специальные символы в тексте.
    """
    normalized = normalize_text(text)
    label = label.lower()

    if not normalized:
        return "empty"

    if label not in VALID_LABELS:
        return "invalid_label"

    existing_label = check_message_in_dataset(normalized)

    if existing_label == label:
        return "same"

    if existing_label is not None:
        return "conflict"

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(
            file,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerow([label, normalized])

    return "added"


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_artifacts():
    """
    Загружает обученную модель и порог классификации.

    В новой архитектуре TF-IDF и scaler находятся внутри Pipeline,
    поэтому отдельно загружать их не требуется.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена модель: {MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Не найден порог классификации: {THRESHOLD_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    threshold = float(threshold)

    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Некорректное значение порога: {threshold}"
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Сохранённая модель не поддерживает predict_proba()."
        )

    return model, threshold


def get_model_name(model):
    """Пытается определить название модели внутри Pipeline."""
    current = model

    # Pipeline
    if hasattr(current, "named_steps"):
        steps = list(current.named_steps.values())

        if steps:
            current = steps[-1]

    # CalibratedClassifierCV
    if hasattr(current, "estimator"):
        current = current.estimator

        if hasattr(current, "named_steps"):
            steps = list(current.named_steps.values())

            if steps:
                current = steps[-1]

    return type(current).__name__


# ============================================================
# PREDICTION
# ============================================================

def predict_message(text, model, threshold):
    """
    Классифицирует одно сообщение.

    Подготовка признаков выполняется через общий features.py.
    Все дальнейшие преобразования выполняет сохранённый Pipeline.
    """
    normalized = normalize_text(text)

    if not normalized:
        raise ValueError("Сообщение не должно быть пустым.")

    X = prepare_features([normalized])

    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    # В обучении spam кодируется единицей.
    if 1 in classes:
        spam_index = classes.index(1)
    elif "spam" in classes:
        spam_index = classes.index("spam")
    else:
        raise ValueError(
            "Не удалось определить класс spam в сохранённой модели."
        )

    spam_probability = float(probabilities[spam_index])
    ham_probability = 1.0 - spam_probability

    prediction = (
        "spam"
        if spam_probability >= threshold
        else "ham"
    )

    structural_features = extract_structural_features(normalized)

    return {
        "prediction": prediction,
        "spam_probability": spam_probability,
        "ham_probability": ham_probability,
        "threshold": threshold,
        "features": structural_features,
    }


# ============================================================
# STRUCTURAL FEATURES DISPLAY
# ============================================================

FEATURE_LABELS = {
    "text_length": "Длина сообщения",
    "word_count": "Количество слов",
    "uppercase_count": "Количество заглавных букв",
    "uppercase_ratio": "Доля заглавных букв",
    "digit_count": "Количество цифр",
    "digit_ratio": "Доля цифр",
    "has_url": "Есть ссылка",
    "url_count": "Количество ссылок",
    "has_long_number": "Есть длинная последовательность цифр",
    "exclamation_count": "Количество восклицательных знаков",
    "question_count": "Количество вопросительных знаков",
    "special_char_count": "Количество специальных символов",
    "currency_count": "Количество денежных символов",
    "trigger_word_count": "Количество триггерных слов",
    "trigger_win": "Слово win",
    "trigger_free": "Слово free",
    "trigger_urgent": "Слово urgent",
    "trigger_call": "Слово call",
    "trigger_text": "Слово text",
    "trigger_stop": "Слово stop",
    "trigger_claim": "Слово claim",
    "trigger_prize": "Слово prize",
    "trigger_offer": "Слово offer",
    "trigger_congratulations": "Слово congratulations",
    "trigger_click": "Слово click",
    "trigger_limited": "Слово limited",
}


def display_structural_features(features):
    """Показывает рассчитанные структурные признаки."""
    rows = []

    for feature_name, value in features.items():
        label = FEATURE_LABELS.get(feature_name, feature_name)

        if feature_name.endswith("_ratio"):
            display_value = f"{float(value):.3f}"
        else:
            display_value = value

        rows.append({
            "Признак": label,
            "Значение": display_value,
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# RETRAINING
# ============================================================

def retrain_model():
    """
    Запускает train_model.py отдельным процессом.

    Старые артефакты не удаляются приложением.
    Их сохранение и замена контролируются train_model.py.
    """
    if not TRAIN_SCRIPT_PATH.exists():
        st.error(
            f"Не найден скрипт обучения: {TRAIN_SCRIPT_PATH}"
        )
        return False

    with st.spinner("Обучение моделей. Это может занять некоторое время..."):
        try:
            result = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT_PATH)],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        except Exception as error:
            st.error(f"Не удалось запустить обучение: {error}")
            return False

    if result.stdout:
        with st.expander("Вывод скрипта обучения"):
            st.text(result.stdout)

    if result.returncode != 0:
        st.error(
            "Обучение завершилось с ошибкой. "
            "Текущие артефакты модели не следует заменять вручную."
        )

        if result.stderr:
            st.code(result.stderr)

        return False

    st.success("Обучение успешно завершено.")

    # Обновляем загруженную модель при следующем обращении.
    load_artifacts.clear()

    try:
        st.session_state["original_count"] = get_dataset_size()
    except Exception as error:
        st.warning(
            f"Модель обучена, но не удалось обновить счётчик датасета: {error}"
        )

    return True


# ============================================================
# SESSION STATE
# ============================================================

if "last_message" not in st.session_state:
    st.session_state["last_message"] = ""

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

if "original_count" not in st.session_state:
    try:
        st.session_state["original_count"] = get_dataset_size()
    except Exception:
        st.session_state["original_count"] = 0


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model, threshold = load_artifacts()
    model_loaded = True
    model_name = get_model_name(model)

except Exception as error:
    model_loaded = False
    model = None
    threshold = None
    model_name = "Не загружена"

    st.error(
        "Не удалось загрузить модель. "
        "Проверьте наличие и совместимость файлов model.pkl и threshold.pkl."
    )

    with st.expander("Подробности ошибки"):
        st.exception(error)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Информация о системе")

    if model_loaded:
        st.success("Модель загружена")
        st.write(f"**Модель:** {model_name}")
        st.write(f"**Порог классификации:** {threshold:.3f}")
    else:
        st.error("Модель недоступна")

    st.divider()

    try:
        dataset_size = get_dataset_size()
        st.metric("Сообщений в датасете", dataset_size)

        new_messages = max(
            0,
            dataset_size - st.session_state["original_count"],
        )

        st.metric("Новых сообщений", new_messages)

    except Exception as error:
        st.warning(f"Не удалось прочитать датасет: {error}")

    st.divider()

    st.subheader("Переобучение")

    st.caption(
        "Новые сообщения с обратной связью добавляются в датасет. "
        "Обучение запускается отдельно."
    )

    if st.button(
        "Переобучить модели",
        use_container_width=True,
        disabled=not DATA_PATH.exists(),
    ):
        success = retrain_model()

        if success:
            st.rerun()


# ============================================================
# TABS
# ============================================================

tab_single, tab_batch, tab_interpretation = st.tabs([
    "Проверка сообщения",
    "Массовая проверка",
    "Интерпретация модели",
])


# ============================================================
# TAB 1 — SINGLE MESSAGE
# ============================================================

with tab_single:
    st.subheader("Проверка SMS")

    st.write(
        "Введите текст сообщения, чтобы определить, "
        "похоже ли оно на мошенническое или спам."
    )

    examples = {
        "Пример мошеннического SMS": (
            "Congratulations! You have won a free prize. "
            "Call now to claim your reward!"
        ),
        "Пример обычного SMS": (
            "Hey, are we still meeting for lunch tomorrow?"
        ),
    }

    example_columns = st.columns(2)

    for column, (label, example_text) in zip(
        example_columns,
        examples.items(),
    ):
        with column:
            if st.button(label, use_container_width=True):
                st.session_state["message_input"] = example_text

    message = st.text_area(
        "Текст сообщения",
        value=st.session_state.get("message_input", ""),
        height=150,
        placeholder="Введите SMS...",
        key="message_text_area",
    )

    if st.button(
        "Проверить сообщение",
        type="primary",
        disabled=not model_loaded,
    ):
        if not normalize_text(message):
            st.warning("Введите текст сообщения.")

        else:
            try:
                result = predict_message(
                    message,
                    model,
                    threshold,
                )

                st.session_state["last_message"] = normalize_text(message)
                st.session_state["last_prediction"] = result

            except Exception as error:
                st.error(f"Ошибка при проверке сообщения: {error}")

    result = st.session_state["last_prediction"]

    if result is not None:
        st.divider()
        st.subheader("Результат классификации")

        prediction = result["prediction"]
        spam_probability = result["spam_probability"]
        ham_probability = result["ham_probability"]

        if prediction == "spam":
            st.error("Сообщение классифицировано как SPAM")
        else:
            st.success("Сообщение классифицировано как HAM")

        metric_columns = st.columns(2)

        with metric_columns[0]:
            st.metric(
                "Вероятность SPAM",
                f"{spam_probability:.1%}",
            )

        with metric_columns[1]:
            st.metric(
                "Вероятность HAM",
                f"{ham_probability:.1%}",
            )

        st.progress(spam_probability)

        st.caption(
            f"Порог классификации: {result['threshold']:.3f}. "
            "Если вероятность SPAM достигает или превышает порог, "
            "сообщение относится к SPAM."
        )

        st.divider()

        st.subheader("Анализ структурных признаков")

        display_structural_features(result["features"])

        st.divider()

        st.subheader("Обратная связь")

        st.write(
            "Если классификация оказалась неверной, "
            "вы можете указать правильную метку. "
            "Сообщение будет добавлено в датасет для следующего обучения."
        )

        feedback_columns = st.columns(2)

        with feedback_columns[0]:
            if st.button(
                "Это SPAM",
                use_container_width=True,
            ):
                feedback_status = add_feedback(
                    st.session_state["last_message"],
                    "spam",
                )

                if feedback_status == "added":
                    st.success(
                        "Сообщение добавлено в датасет с меткой SPAM."
                    )
                    st.rerun()

                elif feedback_status == "same":
                    st.info(
                        "Это сообщение уже есть в датасете с такой меткой."
                    )

                elif feedback_status == "conflict":
                    st.warning(
                        "Это сообщение уже присутствует в датасете "
                        "с другой меткой. Автоматически менять разметку нельзя."
                    )

                elif feedback_status == "empty":
                    st.warning("Пустое сообщение нельзя добавить.")

                else:
                    st.error("Не удалось добавить сообщение.")

        with feedback_columns[1]:
            if st.button(
                "Это HAM",
                use_container_width=True,
            ):
                feedback_status = add_feedback(
                    st.session_state["last_message"],
                    "ham",
                )

                if feedback_status == "added":
                    st.success(
                        "Сообщение добавлено в датасет с меткой HAM."
                    )
                    st.rerun()

                elif feedback_status == "same":
                    st.info(
                        "Это сообщение уже есть в датасете с такой меткой."
                    )

                elif feedback_status == "conflict":
                    st.warning(
                        "Это сообщение уже присутствует в датасете "
                        "с другой меткой. Автоматически менять разметку нельзя."
                    )

                elif feedback_status == "empty":
                    st.warning("Пустое сообщение нельзя добавить.")

                else:
                    st.error("Не удалось добавить сообщение.")


# ============================================================
# TAB 2 — BATCH PREDICTION
# ============================================================

with tab_batch:
    st.subheader("Массовая проверка сообщений")

    st.write(
        "Загрузите CSV или TXT-файл с сообщениями. "
        "Для CSV можно выбрать столбец, содержащий текст."
    )

    uploaded_file = st.file_uploader(
        "Загрузить файл",
        type=["csv", "txt"],
        key="batch_file",
    )

    if uploaded_file is not None and model_loaded:
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()

            if file_extension == ".csv":
                file_bytes = uploaded_file.getvalue()

                try:
                    uploaded_df = pd.read_csv(
                        io.BytesIO(file_bytes),
                        encoding="utf-8-sig",
                    )
                except UnicodeDecodeError:
                    uploaded_df = pd.read_csv(
                        io.BytesIO(file_bytes),
                        encoding="cp1251",
                    )

                if uploaded_df.empty:
                    st.warning("В файле нет сообщений.")

                else:
                    columns = uploaded_df.columns.tolist()

                    default_column = (
                        "text"
                        if "text" in columns
                        else columns[0]
                    )

                    selected_column = st.selectbox(
                        "Столбец с текстом сообщений",
                        options=columns,
                        index=columns.index(default_column),
                    )

                    messages = (
                        uploaded_df[selected_column]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )

            else:
                try:
                    file_text = uploaded_file.getvalue().decode("utf-8-sig")
                except UnicodeDecodeError:
                    file_text = uploaded_file.getvalue().decode("cp1251")

                messages = file_text.splitlines()

            messages = [
                normalize_text(message)
                for message in messages
                if normalize_text(message)
            ]

            if not messages:
                st.warning("В файле не найдено непустых сообщений.")

            else:
                st.write(
                    f"Найдено сообщений для проверки: **{len(messages)}**"
                )

                if st.button(
                    "Проверить все сообщения",
                    type="primary",
                ):
                    batch_results = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for index, text in enumerate(messages):
                        result = predict_message(
                            text,
                            model,
                            threshold,
                        )

                        batch_results.append({
                            "text": text,
                            "prediction": result["prediction"],
                            "spam_probability": result["spam_probability"],
                            "ham_probability": result["ham_probability"],
                        })

                        progress_bar.progress(
                            (index + 1) / len(messages)
                        )

                        status_text.text(
                            f"Обработано: {index + 1} из {len(messages)}"
                        )

                    batch_df = pd.DataFrame(batch_results)

                    batch_df = batch_df.rename(columns={
                        "text": "Сообщение",
                        "prediction": "Классификация",
                        "spam_probability": "Вероятность SPAM",
                        "ham_probability": "Вероятность HAM",
                    })

                    st.session_state["batch_results"] = batch_df

                    progress_bar.empty()
                    status_text.empty()

        except Exception as error:
            st.error(f"Не удалось обработать файл: {error}")

    elif uploaded_file is not None and not model_loaded:
        st.warning("Сначала необходимо загрузить обученную модель.")

    if "batch_results" in st.session_state:
        batch_df = st.session_state["batch_results"]

        st.divider()
        st.subheader("Результаты массовой проверки")

        spam_count = (
            batch_df["Классификация"] == "spam"
        ).sum()

        ham_count = (
            batch_df["Классификация"] == "ham"
        ).sum()

        metric_columns = st.columns(2)

        with metric_columns[0]:
            st.metric("SPAM", int(spam_count))

        with metric_columns[1]:
            st.metric("HAM", int(ham_count))

        st.dataframe(
            batch_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = batch_df.to_csv(
            index=False,
            encoding="utf-8-sig",
        )

        st.download_button(
            label="Скачать результаты CSV",
            data=csv_data,
            file_name="sms_classification_results.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 3 — MODEL INTERPRETATION
# ============================================================

with tab_interpretation:
    st.subheader("Интерпретация и оценка модели")

    st.write(
        "Здесь представлены результаты сравнения моделей, "
        "экспериментов с признаками и оценки качества классификации."
    )

    if COMPARISON_PATH.exists():
        st.markdown("### Сравнение моделей")

        try:
            comparison_df = pd.read_csv(COMPARISON_PATH)

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as error:
            st.error(
                f"Не удалось прочитать таблицу сравнения моделей: {error}"
            )

    else:
        st.info(
            "Таблица сравнения пока отсутствует. "
            "Она появится после успешного обучения."
        )

    if EXPERIMENTS_PATH.exists():
        st.markdown("### Эксперименты с наборами признаков")

        try:
            experiments_df = pd.read_csv(EXPERIMENTS_PATH)

            st.dataframe(
                experiments_df,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as error:
            st.error(
                f"Не удалось прочитать результаты экспериментов: {error}"
            )

    else:
        st.info(
            "Результаты экспериментов пока отсутствуют."
        )

    st.divider()

    st.markdown("### Матрицы ошибок")

    if CONFUSION_MATRIX_PATH.exists():
        st.image(
            str(CONFUSION_MATRIX_PATH),
            caption="Confusion matrix",
            use_container_width=True,
        )
    else:
        st.info("Матрица ошибок появится после обучения.")

    st.divider()

    st.markdown("### Важность признаков")

    if FEATURE_IMPORTANCE_PATH.exists():
        st.image(
            str(FEATURE_IMPORTANCE_PATH),
            caption="Feature importance",
            use_container_width=True,
        )
    else:
        st.info("График важности признаков появится после обучения.")

    st.divider()

    st.markdown("### Методология эксперимента")

    st.write(
        """
        **Сравниваемые модели:**
        - Logistic Regression
        - LinearSVC с калибровкой вероятностей
        - Multinomial Naive Bayes

        **Конфигурации признаков:**
        1. TF-IDF only — только текстовые признаки.
        2. Structural only — только структурные признаки.
        3. TF-IDF + basic structural — текстовые и базовые структурные признаки.
        4. TF-IDF + all structural — текстовые и все структурные признаки,
           включая триггерные слова.

        **Схема оценки:**
        - Nested cross-validation: 5 внешних и 3 внутренних фолда.
        - Подбор гиперпараметров с помощью GridSearchCV.
        - Подбор порога классификации по OOF-предсказаниям.
        - Финальная оценка на отдельной тестовой выборке,
          которая не участвует в обучении и подборе порога.

        TF-IDF и масштабирование признаков находятся внутри Pipeline,
        поэтому преобразования обучаются только на соответствующих
        обучающих частях выборки.
        """
    )
