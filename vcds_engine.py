"""
vcds_engine.py — Парсер VCDS, генератор тестовых логов, анализ данных
ВСЕ ЛОГИ ГЕНЕРИРУЮТСЯ ДЛЯ РЕЖИМА БЕНЗИНА (базовый режим диагностики)
"""

import pandas as pd
import numpy as np
import re
import io
from io import StringIO
from typing import Tuple, Optional, Dict, List

# ==================== ПАРСЕР VCDS CSV ====================

def parse_vcds_csv(file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Парсит CSV/TXT лог VCDS, возвращает DataFrame и VIN."""
    try:
        for enc in ['cp1251', 'utf-8', 'latin-1']:
            try:
                text = file_bytes.decode(enc, errors='strict')
                break
            except UnicodeDecodeError:
                continue
        else:
            text = file_bytes.decode('cp1251', errors='ignore')
    except Exception:
        return None, None

    delimiter = '\t' if '\t' in text else ','
    lines = text.splitlines()

    # Ищем VIN
    vin_pattern = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b', re.IGNORECASE)
    extracted_vin = None
    for line in lines[:30]:
        match = vin_pattern.search(line)
        if match:
            extracted_vin = match.group(1).upper()
            break

    # Ищем заголовок
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(('"', 'STAMP', 'TIME', 'Marker', 'Address', 'Группа', 'Group')):
            header_idx = i
            break
    if header_idx is None:
        for i, line in enumerate(lines):
            if line.count(delimiter) > 2:
                header_idx = i
                break
    if header_idx is None:
        return None, extracted_vin

    csv_text = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(StringIO(csv_text), delimiter=delimiter, skip_blank_lines=True)
    except Exception:
        return None, extracted_vin

    df.columns = [col.strip().strip('"').strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='ignore')

    df = df.dropna(how='all')
    if len(df) > 180:
        df = df.iloc[:180]

    return df, extracted_vin


# ==================== ГЕНЕРАТОР ТЕСТОВЫХ ЛОГОВ (ТОЛЬКО БЕНЗИН) ====================

def generate_test_log_df(scenario="normal", diagnostic_mode="Механика (Группы 001-063)", 
                         is_base_trim=False, mods=None) -> pd.DataFrame:
    """
    Генерирует тестовый DataFrame с заданным сценарием неисправности.
    ВСЕ ПАРАМЕТРЫ — ДЛЯ БЕНЗИНА (базовый режим).
    ГБО-поправки НЕ применяются.
    """
    if mods is None:
        mods = {"tuned": False, "decatted": False, "lpg": False}

    np.random.seed(42)
    time = np.arange(0, 100, 1.0)
    n_points = len(time)

    # Электрика/CAN
    if diagnostic_mode.startswith("Электрика"):
        akpp = np.ones(n_points) if not is_base_trim else np.full(n_points, -1)
        abs_block = np.ones(n_points) if not is_base_trim else np.full(n_points, -1)
        klimat = np.ones(n_points) if not is_base_trim else np.full(n_points, -1)

        if scenario == "can_loss_abs" and not is_base_trim:
            abs_block = np.zeros(n_points)
        elif scenario == "immo_conflict":
            priborka = np.random.choice([0, 1], p=[0.8, 0.2], size=n_points)
        else:
            priborka = np.ones(n_points)

        srs_block = np.ones(n_points)
        fan = np.zeros(n_points)

        return pd.DataFrame({
            "Отметка Времени (сек)": time,
            "АКПП (Группа 125-1)": akpp,
            "АБС (Группа 125-2)": abs_block,
            "Приборка (Группа 125-3)": priborka if 'priborka' in locals() else np.ones(n_points),
            "SRS (Группа 125-4)": srs_block,
            "Климат (Группа 126-1)": klimat,
            "Запрос Вентилятора % (Группа 135-1)": fan
        })

    # === БАЗОВЫЕ ПАРАМЕТРЫ (БЕНЗИН, НОРМАЛЬНЫЙ РЕЖИМ) ===
    # Все значения — для бензина, без ГБО-поправок
    rpm = 840 + np.random.normal(0, 8, n_points)
    coolant_temp = np.clip(20.0 + time * 0.7, 20.0, 90.0)
    iat = np.clip(25.0 + np.random.normal(0, 2, n_points) + time * 0.1, 5.0, 70.0)
    map_base = 290.0 if is_base_trim else 305.0
    map_vals = map_base + np.random.normal(0, 5, n_points)

    # Время впрыска на бензине: 2.0-3.0 мс на ХХ
    injector = 2.25 + np.random.normal(0, 0.04, n_points)

    # Коррекции на бензине: ±5%
    stft = np.random.normal(0, 1.0, n_points)
    ltft = np.zeros(n_points) + 0.8

    misfire_c1 = np.zeros(n_points)
    misfire_c2 = np.zeros(n_points)
    misfire_c3 = np.zeros(n_points)
    misfire_c4 = np.zeros(n_points)
    misfire_status = np.ones(n_points)

    g187 = np.random.normal(7.0, 1.0, n_points)
    g188 = 100.0 - g187 + np.random.normal(0, 0.3, n_points)
    g79 = np.ones(n_points) * 14.0
    g185 = g79 / 2.0
    throttle = g187

    # УОЗ на бензине: 4-10° на ХХ
    uoz = np.random.normal(6.0, 2.0, n_points)

    phase_position = np.random.normal(0.0, 0.3, n_points)
    knock_c1 = knock_c2 = knock_c3 = knock_c4 = np.zeros(n_points)

    dd_base = 0.5 + (rpm - 840) / 5000 * 2.5
    dd_c1 = dd_base + np.random.normal(0, 0.05, n_points)
    dd_c2 = dd_base + np.random.normal(0, 0.05, n_points)
    dd_c3 = dd_base + np.random.normal(0, 0.05, n_points)
    dd_c4 = dd_base + np.random.normal(0, 0.05, n_points)

    o2_heater_resistance = np.random.normal(8.0, 2.0, n_points)
    cat_conversion = np.random.normal(0.2, 0.1, n_points)
    cat_status = np.ones(n_points)
    o2_voltage = np.random.normal(1.50, 0.05, n_points)
    adaptation_status = np.full(n_points, "ADP. OK")

    # decatted влияет на катализатор (физически, независимо от топлива)
    if mods.get("decatted", False):
        cat_status = np.full(n_points, -1)
        cat_conversion = np.zeros(n_points)

    # === СЦЕНАРИИ НЕИСПРАВНОСТЕЙ (ВСЕ НА БЕНЗИНЕ) ===

    if scenario == "detonation":
        # ДЕТОНАЦИЯ: УОЗ ОТКАТЫВАЕТСЯ (уменьшается), откат УОЗ растёт
        rpm = np.linspace(2000, 5600, n_points)
        map_vals = np.linspace(800, 980, n_points)
        injector = np.linspace(6.0, 11.5, n_points)  # Бензин, под нагрузкой
        stft = np.random.normal(0, 1.5, n_points)
        ltft = np.ones(n_points) + 1.5
        g79 = np.linspace(14.0, 90.0, n_points)
        g185 = g79 / 2.0
        g187 = np.linspace(7.0, 90.0, n_points)
        g188 = 100.0 - g187 + np.random.normal(0, 0.3, n_points)
        throttle = g187
        # УОЗ ОТКАТЫВАЕТСЯ назад — признак детонации
        uoz = np.clip(np.linspace(3, 6, n_points) + np.random.normal(0, 1, n_points), 0, 10)
        # Откат УОЗ (knock retard) — высокий, это детонация
        knock_all = np.clip(np.random.normal(6.0, 1.5, n_points), 4.0, 10.0)
        knock_c1 = knock_c2 = knock_c3 = knock_c4 = knock_all
        dd_base = 0.5 + (rpm - 840) / 5000 * 2.5 + 0.2
        dd_c1 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c2 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c3 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c4 = dd_base + np.random.normal(0, 0.05, n_points)
        cat_conversion = np.random.normal(0.5, 0.2, n_points)
        iat = np.clip(35.0 + time * 0.5 + np.random.normal(0, 3, n_points), 5.0, 90.0)
        o2_voltage = np.clip(1.50 + stft * 0.02, 1.0, 2.5)

    elif scenario == "leak":
        # Подсос воздуха
        map_vals = 385.0 + np.random.normal(0, 6, n_points)
        injector = 3.10 + np.random.normal(0, 0.04, n_points)  # Бензин, бедная смесь
        stft = 15.2 + np.random.normal(0, 1.2, n_points)
        ltft = np.ones(n_points) + 6.5
        uoz = np.random.normal(7.0, 1.5, n_points)
        throttle = np.random.normal(1.0, 0.2, n_points)
        g187 = throttle
        g188 = 100.0 - g187 + np.random.normal(0, 0.3, n_points)
        phase_position = -4.5 + np.random.normal(0, 0.3, n_points)
        misfire_c1 = np.random.choice([0,1], p=[0.95,0.05], size=n_points)
        misfire_c2 = np.random.choice([0,1], p=[0.95,0.05], size=n_points)
        misfire_c3 = np.random.choice([0,1], p=[0.95,0.05], size=n_points)
        misfire_c4 = np.random.choice([0,1], p=[0.95,0.05], size=n_points)
        o2_voltage = np.clip(1.50 + stft * 0.02, 1.0, 2.5)

    elif scenario == "rich":
        # Богатая смесь (засорённый ДМРВ)
        map_vals = 300.0 + np.random.normal(0, 5, n_points)
        injector = 1.85 + np.random.normal(0, 0.03, n_points)  # Бензин, короткий впрыск
        stft = -18.5 + np.random.normal(0, 1.0, n_points)
        ltft = np.ones(n_points) - 9.0
        cat_conversion = np.random.normal(0.7, 0.2, n_points)
        o2_voltage = np.clip(1.50 + stft * 0.02, 1.0, 2.5)

    elif scenario == "fuel_pump_death":
        # Умирающий бензонасос
        rpm = np.linspace(840, 4500, n_points)
        map_vals = np.linspace(300.0, 850.0, n_points)
        injector = np.linspace(2.5, 4.8, n_points)  # Бензин, недостаточный впрыск
        stft = np.linspace(2.0, 24.0, n_points)
        ltft = np.ones(n_points) + 8.5
        g79 = np.linspace(14.0, 70.0, n_points)
        g185 = g79 / 2.0
        g187 = np.linspace(7.0, 60.0, n_points)
        g188 = 100.0 - g187 + np.random.normal(0, 0.3, n_points)
        throttle = g187
        misfire_c1 = np.clip(np.cumsum(np.random.choice([0,1], p=[0.9,0.1], size=n_points)), 0, 10)
        misfire_c4 = np.clip(np.cumsum(np.random.choice([0,1], p=[0.9,0.1], size=n_points)), 0, 10)
        uoz = np.clip(np.linspace(15, 30, n_points) + np.random.normal(0, 2, n_points), 10, 35)
        knock_all = np.clip(np.linspace(0.0, 5.0, n_points) + np.random.normal(0, 0.5, n_points), 0.0, 7.0)
        knock_c1 = knock_c2 = knock_c3 = knock_c4 = knock_all
        dd_base = 0.5 + (rpm - 840) / 5000 * 2.5 + 0.1
        dd_c1 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c2 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c3 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c4 = dd_base + np.random.normal(0, 0.05, n_points)
        cat_conversion = np.linspace(0.4, 0.8, n_points)
        iat = np.clip(30.0 + time * 0.8, 5.0, 100.0)
        o2_voltage = np.clip(1.50 + stft * 0.02, 1.0, 2.5)

    elif scenario == "misfire_coil":
        # Локальный пропуск (катушка 2 цил.)
        misfire_c2 = np.clip(np.cumsum(np.random.choice([0,1,2], p=[0.7,0.2,0.1], size=n_points)), 0, 45)
        stft = np.linspace(0, 12.0, n_points)
        knock_c2 = np.clip(np.random.normal(4.0, 1.0, n_points), 2.0, 7.0)
        dd_c2 = dd_base + 0.4 + np.random.normal(0, 0.1, n_points)
        o2_heater_resistance = np.full(n_points, 99.9)
        o2_voltage = np.clip(1.50 + stft * 0.02, 1.0, 2.5)

    elif scenario == "compression_loss":
        # Потеря компрессии (цил 4)
        rpm = np.concatenate([np.ones(50)*840, np.linspace(840, 2500, 50)])
        map_vals = np.concatenate([np.ones(50)*375.0, np.linspace(375.0, 500.0, 50)])
        misfire_c4 = np.concatenate([np.cumsum(np.random.choice([0,1], p=[0.5,0.5], size=50)), np.ones(50)*25])
        uoz = np.concatenate([np.random.normal(6.0, 2.0, 50), np.clip(np.linspace(10, 20, 50), 5, 25)])
        g187 = np.concatenate([np.random.normal(7.0, 1.0, 50), np.linspace(10, 25, 50)])
        g188 = 100.0 - g187 + np.random.normal(0, 0.3, n_points)
        throttle = g187
        dd_c4[:50] = 1.3 + np.random.normal(0, 0.1, 50)
        phase_position = -6.0 + np.random.normal(0, 0.5, n_points)
        cat_conversion[:50] = 0.7
        iat = np.concatenate([np.random.normal(55, 3, 50), np.linspace(55, 35, 50)])
        o2_voltage = np.concatenate([np.random.normal(1.50, 0.05, 50), np.random.normal(1.60, 0.1, 50)])

    total_misfires = misfire_c1 + misfire_c2 + misfire_c3 + misfire_c4

    return pd.DataFrame({
        "Время (сек)": time,
        "Обороты (об/мин)": np.round(rpm, 0),
        "Температура ОЖ (°C)": np.round(coolant_temp, 1),
        "Температура впуска (°C)": np.round(iat, 1),
        "Нагрузка (%)": np.round(rpm * 0.018, 1),
        "Давление ДАД (mbar)": np.round(map_vals, 1),
        "Время впрыска (мс)": np.round(injector, 2),
        "Краткосрочная коррекция (%)": np.round(stft, 2),
        "Долговременная коррекция (%)": np.round(ltft, 2),
        "Угол дросселя (%)": np.round(throttle, 1),
        "Датчик дросселя 1 (G187) %": np.round(g187, 1),
        "Датчик дросселя 2 (G188) %": np.round(g188, 1),
        "Педаль газа 1 (G79) %": np.round(g79, 1),
        "Педаль газа 2 (G185) %": np.round(g185, 1),
        "УОЗ (°ПКВ)": np.round(uoz, 1),
        "Пропуски Ц1": np.round(misfire_c1, 0),
        "Пропуски Ц2": np.round(misfire_c2, 0),
        "Пропуски Ц3": np.round(misfire_c3, 0),
        "Пропуски Ц4": np.round(misfire_c4, 0),
        "Суммарные пропуски": np.round(total_misfires, 0),
        "Статус пропусков": misfire_status.astype(int),
        "Откат УОЗ Ц1 (°KW)": np.round(knock_c1, 1),
        "Откат УОЗ Ц2 (°KW)": np.round(knock_c2, 1),
        "Откат УОЗ Ц3 (°KW)": np.round(knock_c3, 1),
        "Откат УОЗ Ц4 (°KW)": np.round(knock_c4, 1),
        "Напряжение ДД Ц1 (В)": np.round(dd_c1, 2),
        "Напряжение ДД Ц2 (В)": np.round(dd_c2, 2),
        "Напряжение ДД Ц3 (В)": np.round(dd_c3, 2),
        "Напряжение ДД Ц4 (В)": np.round(dd_c4, 2),
        "Фазовое положение (°)": np.round(phase_position, 2),
        "Лямбда-регулирование (%)": np.round(stft, 2),
        "Напряжение Зонда 1 (В)": np.round(o2_voltage, 2),
        "Сопротивление Зонда 1 (Ом)": np.round(o2_heater_resistance, 1),
        "Конверсия катализатора": np.round(cat_conversion, 2),
        "Статус катализатора": cat_status.astype(int),
        "Статус адаптации": adaptation_status,
    })


# ==================== СТАТИСТИКА ЛОГА ====================

def generate_log_summary(df: pd.DataFrame, vin: str = "", is_base_trim: bool = False) -> str:
    """Генерирует текстовую сводку по логу для отправки в ИИ."""
    summary = []
    summary.append(f"Данные CSV-лога (сняты на БЕНЗИНЕ). VIN: {vin if vin else 'Не указан'}")
    time_col = df.columns[0]
    summary.append(f"Точек: {len(df)}. Длительность: {df[time_col].max() - df[time_col].min():.1f} сек.")

    for col in df.columns[1:]:
        if is_base_trim and any(x in col for x in ["АКПП", "АБС", "Климат"]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            min_val = df[col].min()
            max_val = df[col].max()
            std_val = df[col].std()
            summary.append(f"- {col}: среднее={mean_val:.2f}, мин={min_val:.2f}, макс={max_val:.2f}, СКО={std_val:.2f}")

    return "\n".join(summary)


def encode_image_to_base64(uploaded_file) -> Optional[str]:
    """Кодирует загруженный файл в base64 data URL."""
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        encoded_string = base64.b64encode(file_bytes).decode('utf-8')
        return f"data:{uploaded_file.type};base64,{encoded_string}"
    return None
