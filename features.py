
import re

import pandas as pd


# ============================================================
# TRIGGER WORDS
# ============================================================

TRIGGER_WORDS = [
    "win",
    "free",
    "urgent",
    "call",
    "text",
    "stop",
    "claim",
    "prize",
    "offer",
    "congratulations",
    "click",
    "limited",
]


# ============================================================
# FEATURE GROUPS
# ============================================================

BASIC_FEATURES = [
    "text_length",
    "word_count",
    "uppercase_count",
    "uppercase_ratio",
    "digit_count",
    "digit_ratio",
    "has_url",
    "url_count",
    "has_long_number",
    "exclamation_count",
    "question_count",
    "special_char_count",
    "currency_count",
]

TRIGGER_FEATURES = [
    "trigger_word_count",
    "trigger_win",
    "trigger_free",
    "trigger_urgent",
    "trigger_call",
    "trigger_text",
    "trigger_stop",
    "trigger_claim",
    "trigger_prize",
    "trigger_offer",
    "trigger_congratulations",
    "trigger_click",
    "trigger_limited",
]

ALL_STRUCTURAL_FEATURES = BASIC_FEATURES + TRIGGER_FEATURES


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Очищает текст для последующей векторизации TF-IDF.

    Приводит текст к нижнему регистру и удаляет
    лишние пробелы.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# STRUCTURAL FEATURE EXTRACTION
# ============================================================

def extract_structural_features(text):
    """
    Извлекает структурные признаки SMS.

    Признаки рассчитываются на исходном тексте,
    чтобы не потерять информацию о заглавных буквах.
    """
    if pd.isna(text):
        text = ""

    text = str(text)

    text_lower = text.lower()

    # Длина и количество слов
    text_length = len(text)
    words = re.findall(r"\b\w+\b", text_lower)
    word_count = len(words)

    # Заглавные буквы
    uppercase_count = sum(
        1 for char in text if char.isupper()
    )

    uppercase_ratio = (
        uppercase_count / text_length
        if text_length > 0
        else 0
    )

    # Цифры
    digit_count = sum(
        1 for char in text if char.isdigit()
    )

    digit_ratio = (
        digit_count / text_length
        if text_length > 0
        else 0
    )

    # Ссылки
    urls = re.findall(
        r"(?:https?://|www\.)\S+",
        text_lower,
    )

    url_count = len(urls)
    has_url = int(url_count > 0)

    # Длинная последовательность цифр (7 и более)
    has_long_number = int(
        bool(re.search(r"\d{7,}", text))
    )

    # Знаки препинания
    exclamation_count = text.count("!")
    question_count = text.count("?")

    # Специальные символы
    special_char_count = sum(
        1 for char in text
        if not char.isalnum() and not char.isspace()
    )

    # Денежные символы
    currency_symbols = ["$", "£", "€", "¥"]

    currency_count = sum(
        text.count(symbol)
        for symbol in currency_symbols
    )

    # Триггерные слова
    trigger_counts = {}

    for word in TRIGGER_WORDS:
        pattern = rf"\b{re.escape(word)}\b"

        count = len(
            re.findall(pattern, text_lower)
        )

        trigger_counts[f"trigger_{word}"] = int(count > 0)

    trigger_word_count = sum(
        len(
            re.findall(
                rf"\b{re.escape(word)}\b",
                text_lower,
            )
        )
        for word in TRIGGER_WORDS
    )

    features = {
        "text_length": text_length,
        "word_count": word_count,
        "uppercase_count": uppercase_count,
        "uppercase_ratio": uppercase_ratio,
        "digit_count": digit_count,
        "digit_ratio": digit_ratio,
        "has_url": has_url,
        "url_count": url_count,
        "has_long_number": has_long_number,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "special_char_count": special_char_count,
        "currency_count": currency_count,
        "trigger_word_count": trigger_word_count,
    }

    features.update(trigger_counts)

    return features


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_features(texts):
    """
    Подготавливает DataFrame для ML Pipeline.

    Возвращает:
    - clean_text — очищенный текст для TF-IDF;
    - структурные признаки.
    """
    normalized_texts = [
        "" if pd.isna(text) else str(text)
        for text in texts
    ]

    cleaned_texts = [
        clean_text(text)
        for text in normalized_texts
    ]

    structural_features = [
        extract_structural_features(text)
        for text in normalized_texts
    ]

    features_df = pd.DataFrame(structural_features)

    features_df.insert(
        0,
        "clean_text",
        cleaned_texts,
    )

    return features_df
