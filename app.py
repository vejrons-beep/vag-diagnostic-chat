import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import io
import re
import base64
import os
import copy
from PIL import Image
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="VAG Expert Chat + Vision", page_icon="🚗", layout="wide")

MODEL_NAME = "google/gemini-2.5-flash"
API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

# --- ПРОВЕРКА ПИН-КОДА (с автоматическим входом через URL) ---
def check_password():
    query_params = st.query_params
    if "auth" in query_params:
        st.session_state.authenticated = True
        st.query_params.clear()
        return True
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    st.title("🔒 Доступ ограничен")
    st.write("Введите пин-код для продолжения")
    with st.form("auth_form"):
        password = st.text_input("Пин-код", type="password")
        submit = st.form_submit_button("Войти")
        if submit:
            correct_password = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", "1234"))
            if password == correct_password:
                st.session_state.authenticated = True
                st.query_params["auth"] = "1"
                st.rerun()
            else:
                st.error("Неверный пин-код")
    return False

if not check_password():
    st.stop()

# --- ПОДКЛЮЧЕНИЕ К GOOGLE SHEETS (через base64-секрет) ---
def _get_gsheet():
    try:
        b64_str = st.secrets["GSPREAD_SERVICE_ACCOUNT_BASE64"]
        creds_json = base64.b64decode(b64_str).decode()
        creds_dict = json.loads(creds_json)
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(
            "https://docs.google.com/spreadsheets/d/1ALFMvWYfjJsT_OeLWQRDXIzuvoA-8Xvn9CB5tm8qH1w/edit#gid=0"
        )
        return sheet.sheet1
    except Exception as e:
        st.error(f"❌ Ошибка подключения к Google Sheets: {e}")
        return None

def save_history_to_disk(history, vin_code):
    sheet = _get_gsheet()
    if sheet is None:
        return
    try:
        payload = json.dumps({"vin_code": vin_code, "chat_history": history}, ensure_ascii=False)
        sheet.update("A1", [[payload]])
    except Exception:
        pass

def load_history_from_disk():
    sheet = _get_gsheet()
    if sheet is None:
        st.warning("⚠️ Не удалось подключиться к Google Sheets при загрузке истории.")
        return [], ""
    try:
        data_str = sheet.acell("A1").value
        if data_str:
            data = json.loads(data_str)
            st.success("✅ История загружена из облака.")
            return data.get("chat_history", []), data.get("vin_code", "")
        else:
            st.info("ℹ️ Ячейка пуста, начинаем новую историю.")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки истории: {e}")
    return [], ""

def clear_history_on_disk():
    sheet = _get_gsheet()
    if sheet is None:
        return
    try:
        sheet.update("A1", [[""]])
    except Exception:
        pass

# --- ПАРСЕР VCDS ---
@st.cache_data(show_spinner=False)
def parse_vcds_csv(file_bytes):
    try:
        for enc in ['cp1251', 'utf-8', 'latin-1']:
            try:
                text = file_bytes.decode(enc, errors='strict')
                break
            except UnicodeDecodeError:
                continue
        else:
            text = file_bytes.decode('cp1251', errors='ignore')
    except:
        return None, None
    delimiter = '\t' if '\t' in text else ','
    lines = text.splitlines()
    vin_pattern = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b', re.IGNORECASE)
    extracted_vin = None
    for line in lines[:30]:
        match = vin_pattern.search(line)
        if match:
            extracted_vin = match.group(1).upper()
            break
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(('"','STAMP','TIME','Marker','Address','Группа','Group')):
            header_idx = i
            break
    if header_idx is None:
        for i, line in enumerate(lines):
            if line.count(delimiter) > 2:
                header_idx = i
                break
    if header_idx is None:
        return None, extracted_vin
    from io import StringIO
    csv_text = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(StringIO(csv_text), delimiter=delimiter, skip_blank_lines=True)
    except:
        return None, extracted_vin
    df.columns = [col.strip().strip('"').strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='ignore')
    df = df.dropna(how='all')
    if len(df) > 180:
        df = df.iloc[:180]
    return df, extracted_vin

# --- СТАТИСТИКА ---
def generate_log_summary(df, vin="", is_base_trim=False):
    summary = []
    summary.append(f"Данные CSV-лога. VIN: {vin if vin else 'Не указан'}")
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

def encode_image_to_base64(uploaded_file):
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        encoded_string = base64.b64encode(file_bytes).decode('utf-8')
        return f"data:{uploaded_file.type};base64,{encoded_string}"
    return None

def ask_ai_chat(api_key, model_name, messages, max_tokens=2500):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://share.streamlit.io",
    }
    cleaned_messages = []
    for m in messages:
        new_content = []
        for item in m["content"]:
            if item["type"] == "text":
                new_content.append(item)
            elif item["type"] == "image_url":
                new_content.append({"type": "text", "text": "[Ранее отправленный скриншот экрана диагностики]"})
        cleaned_messages.append({"role": m["role"], "content": new_content})
    if messages and messages[-1]["content"]:
        cleaned_messages[-1]["content"] = messages[-1]["content"]
    data = {
        "model": model_name,
        "messages": cleaned_messages,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Ошибка API OpenRouter: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Ошибка сети: {e}"

# --- ГЕНЕРАТОР ТЕСТОВЫХ ЛОГОВ (с добавлением суммарных пропусков и статуса) ---
def generate_test_log_df(scenario="normal", diagnostic_mode="Механика (Группы 001-063)", is_base_trim=False, mods=None):
    if mods is None:
        mods = {"tuned": False, "decatted": False, "lpg": False}
    np.random.seed(42)
    time = np.arange(0, 100, 1.0)
    n_points = len(time)

    if diagnostic_mode.startswith("Электрика"):
        # ... (CAN-режим не менялся)
        return pd.DataFrame()

    rpm = 840 + np.random.normal(0, 8, n_points)
    coolant_temp = np.clip(20.0 + time * 0.7, 20.0, 90.0)
    iat = np.clip(25.0 + np.random.normal(0, 2, n_points) + time * 0.1, 5.0, 70.0)
    map_base = 290.0 if is_base_trim else 305.0
    map_vals = map_base + np.random.normal(0, 5, n_points)
    injector = 2.25 + np.random.normal(0, 0.04, n_points)
    stft = np.random.normal(0, 1.0, n_points)
    ltft = np.zeros(n_points) + 0.8
    misfire_c1 = np.zeros(n_points)
    misfire_c2 = np.zeros(n_points)
    misfire_c3 = np.zeros(n_points)
    misfire_c4 = np.zeros(n_points)
    total_misfires = misfire_c1 + misfire_c2 + misfire_c3 + misfire_c4
    misfire_status = np.ones(n_points)  # 1=ON (активен)
    g79 = np.ones(n_points) * 14.5
    g187 = np.ones(n_points) * 4.5
    uoz = np.random.normal(6.0, 2.0, n_points)
    throttle = np.random.normal(2.2, 0.2, n_points)
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

    if mods.get("decatted", False):
        cat_status = np.full(n_points, -1)
        cat_conversion = np.zeros(n_points)

    if scenario == "detonation":
        rpm = np.linspace(2000, 5600, n_points)
        map_vals = np.linspace(800, 980, n_points)
        injector = np.linspace(6.0, 11.5, n_points)
        stft = np.random.normal(0, 1.5, n_points)
        ltft = np.ones(n_points) + 1.5
        g79 = np.linspace(14.5, 90.0, n_points)
        g187 = np.linspace(4.5, 88.0, n_points)
        uoz = np.clip(np.linspace(15, 28, n_points) + np.random.normal(0, 2, n_points), 10, 35)
        throttle = np.clip(g79 * 0.7 + np.random.normal(0, 2, n_points), 0, 100)
        knock_all = np.clip(np.random.normal(5.0, 1.5, n_points), 3.0, 8.0)
        knock_c1 = knock_c2 = knock_c3 = knock_c4 = knock_all
        dd_base = 0.5 + (rpm - 840) / 5000 * 2.5 + 0.2
        dd_c1 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c2 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c3 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c4 = dd_base + np.random.normal(0, 0.05, n_points)
        cat_conversion = np.random.normal(0.5, 0.2, n_points)
        iat = np.clip(35.0 + time * 0.5 + np.random.normal(0, 3, n_points), 5.0, 90.0)
    elif scenario == "leak":
        map_vals = 385.0 + np.random.normal(0, 6, n_points)
        injector = 3.10 + np.random.normal(0, 0.04, n_points)
        stft = 15.2 + np.random.normal(0, 1.2, n_points)
        ltft = np.ones(n_points) + 6.5
        uoz = np.random.normal(7.0, 1.5, n_points)
        throttle = np.random.normal(0.8, 0.2, n_points)
        phase_position = -4.5 + np.random.normal(0, 0.3, n_points)
        # При подсосе могут появляться случайные пропуски из-за обеднения
        misfire_c1 = np.random.choice([0,1], p=[0.95,0.05], size=n_points)
        misfire_c2 = np.random.choice([0,1], p=[0.95,0.05], size=n_points)
        misfire_c3 = np.random.choice([0,1], p=[0.95,0.05], size=n_points)
        misfire_c4 = np.random.choice([0,1], p=[0.95,0.05], size=n_points)
    elif scenario == "rich":
        map_vals = 300.0 + np.random.normal(0, 5, n_points)
        injector = 1.85 + np.random.normal(0, 0.03, n_points)
        stft = -18.5 + np.random.normal(0, 1.0, n_points)
        ltft = np.ones(n_points) - 9.0
        cat_conversion = np.random.normal(0.7, 0.2, n_points)
    elif scenario == "fuel_pump_death":
        rpm = np.linspace(840, 4500, n_points)
        map_vals = np.linspace(300.0, 850.0, n_points)
        injector = np.linspace(2.5, 4.8, n_points)
        stft = np.linspace(2.0, 24.0, n_points)
        ltft = np.ones(n_points) + 8.5
        g79 = np.linspace(14.5, 70.0, n_points)
        g187 = np.linspace(4.5, 60.0, n_points)
        misfire_c1 = np.clip(np.cumsum(np.random.choice([0,1], p=[0.9,0.1], size=n_points)), 0, 10)
        misfire_c4 = np.clip(np.cumsum(np.random.choice([0,1], p=[0.9,0.1], size=n_points)), 0, 10)
        uoz = np.clip(np.linspace(15, 30, n_points) + np.random.normal(0, 2, n_points), 10, 35)
        throttle = np.clip(g79 * 0.8 + np.random.normal(0, 1.5, n_points), 0, 100)
        knock_all = np.clip(np.linspace(0.0, 5.0, n_points) + np.random.normal(0, 0.5, n_points), 0.0, 7.0)
        knock_c1 = knock_c2 = knock_c3 = knock_c4 = knock_all
        dd_base = 0.5 + (rpm - 840) / 5000 * 2.5 + 0.1
        dd_c1 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c2 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c3 = dd_base + np.random.normal(0, 0.05, n_points)
        dd_c4 = dd_base + np.random.normal(0, 0.05, n_points)
        cat_conversion = np.linspace(0.4, 0.8, n_points)
        iat = np.clip(30.0 + time * 0.8, 5.0, 100.0)
    elif scenario == "misfire_coil":
        misfire_c2 = np.clip(np.cumsum(np.random.choice([0,1,2], p=[0.7,0.2,0.1], size=n_points)), 0, 45)
        stft = np.linspace(0, 12.0, n_points)
        knock_c2 = np.clip(np.random.normal(4.0, 1.0, n_points), 2.0, 7.0)
        dd_c2 = dd_base + 0.4 + np.random.normal(0, 0.1, n_points)
        o2_heater_resistance = np.full(n_points, 99.9)
    elif scenario == "compression_loss":
        rpm = np.concatenate([np.ones(50)*840, np.linspace(840, 2500, 50)])
        map_vals = np.concatenate([np.ones(50)*375.0, np.linspace(375.0, 500.0, 50)])
        misfire_c4 = np.concatenate([np.cumsum(np.random.choice([0,1], p=[0.5,0.5], size=50)), np.ones(50)*25])
        uoz = np.concatenate([np.random.normal(6.0, 2.0, 50), np.clip(np.linspace(10, 20, 50), 5, 25)])
        throttle = np.concatenate([np.random.normal(2.2, 0.2, 50), np.linspace(5, 15, 50)])
        dd_c4[:50] = 1.3 + np.random.normal(0, 0.1, 50)
        phase_position = -6.0 + np.random.normal(0, 0.5, n_points)
        cat_conversion[:50] = 0.7
        iat = np.concatenate([np.random.normal(55, 3, 50), np.linspace(55, 35, 50)])

    total_misfires = misfire_c1 + misfire_c2 + misfire_c3 + misfire_c4

    df = pd.DataFrame({
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
        "G187 (%)": np.round(g187, 1),
        "G79 (%)": np.round(g79, 1),
        "Сопротивление Зонда 1 (Ом)": np.round(o2_heater_resistance, 1),
        "Конверсия катализатора": np.round(cat_conversion, 2),
        "Статус катализатора": cat_status.astype(int),
    })
    return df

# --- ДИНАМИЧЕСКИЙ СИСТЕМНЫЙ ПРОМПТ (обновлены группы 014–016) ---
def get_system_prompt(mode="Механика (Группы 001-063)", is_base_trim=False, ecu_type="Magneti Marelli 7GV", mods=None):
    if mods is None:
        mods = {"tuned": False, "decatted": False, "lpg": False}

    base_prompt = f"""Ты — профессиональный автодиагност концерна VAG (уровень дилерского центра), специализирующийся на работе с логами VCDS (Вася Диагност).
Текущий блок управления двигателем: {ecu_type}.
Твоя задача — анализировать логи и выдавать точные технические диагнозы по базе параметров.
"""
    config_note = ""
    if is_base_trim:
        config_note = """
[!] ВАЖНО: АВТОМОБИЛЬ В БАЗОВОЙ КОМПЛЕКТАЦИИ (МКПП, БЕЗ КОНДИЦИОНЕРА, БЕЗ ABS).
В группах 125 и 126 значения -1 для ABS, Климата и АКПП являются АБСОЛЮТНОЙ НОРМОЙ (блок физически отсутствует).
Нагрузка на ХХ должна быть строго 15-18.5%, а MAP строго 280-300 мбар (паразитных нагрузок нет).
ЖЕСТКОЕ ПРАВИЛО: Если параметр is_base_trim=True (базовая комплектация), то любые значения MAP выше 315 мбар и Нагрузки выше 20% на холостом ходу ТРАКТУЙ КАК АНОМАЛИЮ.
"""
    mods_note = "\n--- УЧЕТ МОДИФИКАЦИЙ АВТОМОБИЛЯ ---"
    if mods.get("decatted", False):
        mods_note += """
- УДАЛЕН КАТАЛИЗАТОР (Евро-2): Вторая лямбда (Группа 041-1) может быть прямой линией, повторять первую или висеть в ошибке. Игнорируй P0420, не предлагай замену ката.
"""
    else:
        mods_note += """
- СТОКОВЫЙ ВЫПУСК: Вторая лямбда (041-1) должна быть стабильна 0.6-0.7 В. Скачки за первой — катализатор разрушен.
"""
    if mods.get("lpg", False):
        mods_note += """
- УСТАНОВЛЕНО ГБО (Газ): LTFT (032) в пределах ±8% — НОРМА. Не ставь диагноз 'подсос' или 'умирающий бензонасос' при таких коррекциях.
"""
    if mods.get("tuned", False):
        mods_note += """
- ЧИП-ТЮНИНГ (Агрессивная прошивка): УОЗ (003-4) может быть более ранним, время впрыска (002-3) выше стоковых.
"""

    mechanics_rules = """
--- БАЗА ЭТАЛОНОВ CFNA 1.6 MPI (Полный набор групп) ---

ГРУППА 001 (Базовые параметры ХХ):
- 001-1 (Обороты): 650–750 об/мин на прогретом ДВС.
- 001-2 (Температура ОЖ): Рабочая = 85–98 °C.
- 001-3 (Лямбда-регулирование): Норма = ±10.0% (должно циклически колебаться). Застывание = отказ зонда.
- 001-4 (Бинарный статус): 11111111 — все тесты пройдены.

ГРУППА 002 (Воздух, Впрыск и Нагрузка):
- 002-2 (Нагрузка): Для базовой комплектации 15.0–18.5%. >20% — аномалия.
- 002-3 (Время впрыска): 2.00–3.00 мс. >3.20 мс при нормальном MAP — загрязнены форсунки.
- 002-4 (ДАД / MAP): Для базовой 280–300 мбар. >315 мбар — триггер неисправности.

ГРУППЫ 032, 033 (Топливные коррекции):
- 032-1 (LTFT ХХ): ±3.0%. > +4.0% = подсос.
- 032-2 (LTFT нагрузка): ±5.0%. > +6.0% = нехватка топлива.
- 033-1 (STFT онлайн): ±10.0%.

ГРУППА 003 (Зажигание и дроссель):
- 003-3 (Угол дросселя): 1.0–3.5% на ХХ. >3.5% — нагар, <1.0% — подсос.
- 003-4 (УОЗ): 4.0–12.0° до ВМТ, скачет — норма.

ГРУППЫ 014, 015, 016 (Пропуски воспламенения):
- 014-1 (Обороты): 650–750 об/мин.
- 014-2 (Нагрузка): 15.0–18.5%.
- 014-3 (Суммарные пропуски): В норме строго 0.
- 014-4 (Статус распознавания): ON (система активна). OFF — ЭБУ временно отключил подсчёт (например, при низком уровне топлива).
- 015-1...015-3, 016-1 (Счётчики по цилиндрам 1–4): В норме 0.
- Если пропуски растут только в одном цилиндре под нагрузкой — виновна катушка или свеча этого цилиндра. Используй метод перестановки катушек для проверки.
- Если пропуски идут хаотично по всем цилиндрам в сочетании с коррекциями 032 > +8% — смесь слишком бедная (проверяй бензонасос, фильтр или ГБО).
- Если пропуски строго в одном цилиндре только на ХХ, а на оборотах пропадают — проверяй механику (потеря компрессии, клапаны).

ГРУППЫ 062–063 (Дроссель/Педаль): 062-1+2 = 100%, 062-3 = 2*062-4.

ГРУППА 020 (Контроль детонации): Откаты 0.0°, допуск до -4.5°.

ГРУППЫ 021–022: Резерв/Дубли 020.

ГРУППА 026 (Напряжения ДД): ХХ 0.40–0.90 В, нагрузка 1.50–3.50 В.

ГРУППА 093 (Фазы ГРМ): 0.00° (±3.00°). <-5.50° или >+5.50° с MAP > 360 мбар — проскок цепи.

ГРУППА 099 (Лямбда-регулирование): 099-4 = ON на горячую. OFF — аварийный режим.

ГРУППЫ 041, 046 (Подогрев и Катализатор):
- 041-1 (Сопротивление Зонда 1): 2.0–15.0 Ом. >50 Ом — обрыв.
- 046-3 (Конверсия): 0.00–0.45. >0.60 — кат неэффективен.
- 046-4 (Статус): Cat B1 - OK.

ГРУППА 134 (Температура воздуха на впуске):
- 134-2 (IAT): В движении = улица + 5–15°C. На ХХ до +65°C (допустимо).
- Если значение зависло в -40°C или +140°C — датчик G42 неисправен (обрыв/КЗ).
- На холодном НЕзапущенном моторе показания 134-2 и 001-2 должны совпадать с температурой воздуха (±5°C).
- Если IAT > +70°C в пробках и есть откаты УОЗ (020) — ЭБУ позднит зажигание из-за перегрева впуска.

--- ЛОГИКА КРОСС-ВАЛИДАЦИИ (расширенная) ---
1. ЛОКАЛЬНЫЙ ПРОБОЙ КАТУШКИ/СВЕЧИ: ЕСЛИ пропуски быстро растут только в одном цилиндре (015-1...016-1), особенно под нагрузкой, И мгновенная лямбда (033-1) кратковременно уходит в плюс, ТОГДА: проблема в катушке или свече этого цилиндра. Переставить катушку с соседним цилиндром и проверить, переместятся ли пропуски. Если да — менять катушку, если нет — свечу или форсунку.
2. ХАОТИЧНЫЕ ПРОПУСКИ ИЗ-ЗА БЕДНОЙ СМЕСИ: ЕСЛИ пропуски хаотично возникают по всем цилиндрам И коррекции (032-1/2) > +8.0%, ТОГДА: системное обеднение смеси (бензонасос, топливный фильтр, подсос воздуха). Катушки и свечи исправны.
3. ПОТЕРЯ КОМПРЕССИИ (МЕХАНИКА): ЕСЛИ постоянные пропуски в одном цилиндре ТОЛЬКО на ХХ, а при повышении оборотов (>2000) пропуски прекращаются, И MAP > 320 мбар, ТОГДА: потеря компрессии в этом цилиндре (кольца, клапаны). Замерить компрессометром.
4. СКРЫТЫЙ ПОДСОС: ЕСЛИ аддитив > +5.0% И мультипликатив в норме И MAP > 350 мбар, ТОГДА: Подсос за дросселем (клапан ВКГ, кольца форсунок).
5. БЕНЗОНАСОС: ЕСЛИ аддитив в норме И мультипликатив > +7.0% И мгновенная уходит в +25% под нагрузкой, ТОГДА: Дефицит топлива. Замер давления (норма 4.0 бар).
6. РАССИНХРОН ДРОССЕЛЯ/ПЕДАЛИ: Нарушение пропорций 100% или 2:1 → EPC. Замена узла или адаптация (060).

7. ЗАГРЯЗНЕНИЕ ДРОССЕЛЬНОЙ ЗАСЛОНКИ (Паттерн А): ЕСЛИ угол дросселя (003-3) > 4.0% на прогретом ХХ И MAP на грани нормы (~310-315 мбар), ТОГДА: Заслонка забита нагаром. Требуется демонтаж, промывка и адаптация (060).
8. ПОДСОС ВОЗДУХА МИМО ДРОССЕЛЯ (Паттерн Б): ЕСЛИ угол дросселя (003-3) < 1.0% (стремится к нулю) И MAP > 320 мбар на ХХ, ТОГДА: Неучтённый воздух. Проверить дымогенератором впускной коллектор, кольца форсунок, трубку ВКГ.
9. ИЗНОС ШЕСТЕРЕН ДРОССЕЛЯ (Паттерн В): ЕСЛИ при нажатии педали (062-3 растет) угол дросселя (003-3) реагирует с задержкой, зависает или изменяется рывками, ТОГДА: Механический износ шестерен редуктора. Снять крышку, осмотреть зубья, при необходимости замена узла.

10. ПЛОХОЕ ТОПЛИВО (СИСТЕМНАЯ ДЕТОНАЦИЯ): ЕСЛИ откаты УОЗ (020) есть во всех цилиндрах одновременно (например, -4.5°, -3.0°, -4.5°, -3.0°) И топливные коррекции (032) в норме (±2%), ТОГДА: Низкое октановое число бензина. Рекомендация: заправиться АИ-95, проверить датчик детонации.
11. ДЕТОНАЦИЯ ИЗ-ЗА БЕДНОЙ СМЕСИ: ЕСЛИ откаты > -6.0° по всем цилиндрам И мультипликативная коррекция (032-2) > +7.0%, ТОГДА: Перегрев смеси из-за обеднения. Срочно искать причину бедной смеси (насос, фильтр, подсос).
12. ЛОКАЛЬНЫЙ СТУК (ПРОБЛЕМА ОДНОГО ЦИЛИНДРА): ЕСЛИ откаты идут только в одном цилиндре (например, Цил.3: -6.0°) И в этом же цилиндре есть пропуски (015/016), ТОГДА: Нагар на поршне/клапанах, неисправная свеча или форсунка этого цилиндра. Эндоскопия, замена свечи, проверка форсунки.

13. МЕХАНИЧЕСКИЙ СТУК (ПОРШНИ, ГИДРИКИ, НАВЕСНОЕ): ЕСЛИ напряжение ДД (026) на ХХ > 1.20 В в одном или нескольких цилиндрах, НО в Группе 020 откаты УОЗ = 0.0°, ТОГДА: Посторонний механический шум (перекладка поршней CFNA, стук гидрокомпенсатора, износ навесного). Проверить эндоскопом, уровнем масла, навесное оборудование.
14. НЕИСПРАВНОСТЬ ДАТЧИКА ДЕТОНАЦИИ (G61): ЕСЛИ во всех полях 026 напряжение жёстко 0.00 В или 4.95 В и не меняется с оборотами, ТОГДА: Обрыв, КЗ или неверный момент затяжки датчика. Проверить разъём, затяжку (20 Нм), заменить G61.
15. АНОМАЛЬНЫЙ ВНЕШНИЙ ШУМ: ЕСЛИ напряжение по всем цилиндрам на ХХ > 1.20 В, но откатов нет, ТОГДА: Шум от навесного (генератор, помпа) или низкого давления масла. Проверить подшипники и давление масла.

16. ЗАГРЯЗНЕННЫЕ ФОРСУНКИ: ЕСЛИ время впрыска (002-3) > 3.20 мс на ХХ при нормальном MAP (280-300 мбар) и нормальных коррекциях, ТОГДА: Форсунки забиты, промыть на стенде.
17. ДИФФЕРЕНЦИАЦИЯ ПОДСОС/ГРМ: 
   - Если MAP высокий (>325 мбар) И угол дросселя низкий (<1.0%) И коррекция LTFT высокая → ПОДСОС.
   - Если MAP высокий (>350 мбар) И угол дросселя высокий (>4.0%) И коррекции в норме → ПРОБЛЕМА ГРМ (проверить 093-3).

18. ЗАКЛИНИВШИЙ ТЕРМОСТАТ: ЕСЛИ температура ОЖ (001-2) длительно держится в диапазоне 70-78°C и не растёт, ТОГДА: Термостат открыт, замена.
19. НЕИСПРАВНЫЙ ЛЯМБДА-ЗОНД: ЕСЛИ лямбда-регулирование (001-3) застыло на одном значении (0%, +25%, -25%) и не колеблется, ТОГДА: Датчик кислорода неисправен или цепь оборвана.
20. КРИТИЧЕСКИЙ ИЗНОС ЦЕПИ ГРМ: ЕСЛИ фазовое положение (093-3) < -5.50° или > +5.50° в сочетании с MAP > 360 мбар, ТОГДА: Проскок цепи, немедленная замена ГРМ.
21. ЛЮФТ ШПОНКИ КОЛЕНВАЛА: ЕСЛИ 093-3 нестабильно и меняется после перегазовок, ТОГДА: Износ шпонки шестерни.
22. АВАРИЙНЫЙ РЕЖИМ ЛЯМБДА-РЕГУЛИРОВАНИЯ: ЕСЛИ 099-4 = OFF на прогретом моторе, ТОГДА: ЭБУ в Open Loop из-за неисправности первого лямбда-зонда или ошибки прошивки.

23. ОБРЫВ ПОДОГРЕВА ЛЯМБДА-ЗОНДА: ЕСЛИ сопротивление подогрева Зонда 1 (041-1) > 50 Ом (или 99.9) И статус 099-4 = OFF, ТОГДА: Обрыв цепи подогрева. Проверить предохранитель, проводку, заменить датчик.
24. НЕЭФФЕКТИВНЫЙ КАТАЛИЗАТОР (СТОК): ЕСЛИ конверсия (046-3) > 0.60 И статус ката NOT OK (0), ТОГДА: Катализатор разрушен. Замена или удаление с перепрошивкой на Евро-2.
25. НЕКОРРЕКТНАЯ ПРОШИВКА ЕВРО-2: ЕСЛИ включён флаг decatted=True, но статус ката (046-4) = NOT OK (0) или присутствует ошибка P0420, ТОГДА: Ошибка в прошивке – не отключён тест катализатора. Требуется коррекция софта.

26. НЕИСПРАВНОСТЬ ДАТЧИКА ТЕМПЕРАТУРЫ ВПУСКА (G42):
- ЕСЛИ 134-2 = -40°C или +140°C постоянно ИЛИ на холодном моторе разница с 001-2 > 5°C, ТОГДА: датчик неисправен (обрыв/КЗ/уход калибровки). Заменить.
- ЕСЛИ 134-2 > +70°C в движении И в группе 020 есть откаты, ТОГДА: подкапотный перегрев, ЭБУ защитно позднит зажигание.
"""

    can_rules = """--- БАЗА ЭТАЛОНОВ CAN-ШИНЫ (Группы 125-135) ---
Норма связи: 1 = блок отвечает, 0 = нет связи.
- 125-1 (АКПП), 125-2 (АБС/ESP), 126-1 (Климат). В базовой комплектации эти блоки отсутствуют, их значение = -1.
- 125-3 (Приборка): Если 0 или скачет 0/1 — иммо блокирует пуск.
- 135-1 (Запрос вентилятора): 100% при молчащем кулере = сгорел БУВ.
"""
    common_rules = "\nОБЩИЕ ПРАВИЛА: Отвечай профессионально, структурированно. Учитывай модификации и комплектацию."

    if mode.startswith("Электрика"):
        return base_prompt + config_note + mods_note + can_rules + common_rules
    else:
        return base_prompt + config_note + mods_note + mechanics_rules + common_rules

# --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЙ ---
saved_history, saved_vin = load_history_from_disk()
if "is_base_trim" not in st.session_state:
    st.session_state.is_base_trim = True
if "mods" not in st.session_state:
    st.session_state.mods = {"tuned": False, "decatted": False, "lpg": False}
if "diagnostic_mode" not in st.session_state:
    st.session_state.diagnostic_mode = "Механика (Группы 001-063)"

if "chat_history" not in st.session_state:
    if saved_history:
        st.session_state.chat_history = saved_history
    else:
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt(
                st.session_state.diagnostic_mode, st.session_state.is_base_trim,
                mods=st.session_state.mods
            )}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Привет! Я твой виртуальный диагност VAG VCDS. 🚗"}]}
        ]
if "vin_code" not in st.session_state:
    st.session_state.vin_code = saved_vin if saved_vin else ""
if "reference_map" not in st.session_state:
    st.session_state.reference_map = {
        "Давление ДАД (mbar)": (280.0, 340.0, "green", "red"),
        "Время впрыска (мс)": (2.0, 3.0, "green", "red"),
        "Краткосрочная коррекция (%)": (-10.0, 10.0, "blue", "orange"),
        "Долговременная коррекция (%)": (-10.0, 10.0, "blue", "orange"),
        "Пропуски": (0.0, 0.0, "green", "red"),
        "Угол дросселя (%)": (1.0, 3.5, "green", "red"),
        "Откат УОЗ": (0.0, 4.5, "green", "red"),
        "Напряжение ДД": (0.4, 3.5, "green", "red"),
        "Температура ОЖ (°C)": (85.0, 98.0, "green", "red"),
        "Температура впуска (°C)": (5.0, 65.0, "green", "red"),
        "Фазовое положение": (-3.0, 3.0, "green", "red"),
        "Сопротивление Зонда 1 (Ом)": (2.0, 15.0, "green", "red"),
        "Конверсия катализатора": (0.0, 0.45, "green", "red"),
    }
if "generated_log_df" not in st.session_state:
    st.session_state.generated_log_df = None
if "uploaded_image_key" not in st.session_state:
    st.session_state.uploaded_image_key = 0

# --- БОКОВАЯ ПАНЕЛЬ (как раньше, но с передачей mods в генератор) ---
with st.sidebar:
    st.header("⚙️ Конфигурация автомобиля")
    st.write("Настройте параметры машины для точной работы ИИ.")
    st.markdown("---")

    st.subheader("📦 Комплектация")
    st.session_state.is_base_trim = st.checkbox(
        "Базовая комплектация (CFNA BASE)",
        value=st.session_state.is_base_trim,
        help="МКПП, без кондиционера, без ABS. ИИ сузит допуски MAP до 315 мбар и проигнорирует отсутствие блоков по CAN."
    )

    st.markdown("---")
    st.subheader("🛠️ Модификации и тюнинг")

    decatted = st.checkbox(
        "Катализатор удален (Евро-2)",
        value=st.session_state.mods.get("decatted", False),
        help="ИИ проигнорирует просадки и прямые линии по второй лямбде (Группа 041) и не будет советовать замену ката."
    )
    lpg = st.checkbox(
        "Установлено ГБО (Газ)",
        value=st.session_state.mods.get("lpg", False),
        help="ИИ расширит допуски по долговременным коррекциям (LTFT) до ±8% и сделает поправку на специфику смеси пропан-бутана."
    )
    tuned = st.checkbox(
        "Чип-тюнинг (Stage 1/Custom)",
        value=st.session_state.mods.get("tuned", False),
        help="ИИ сделает скидку на более ранние углы зажигания (УОЗ) и повышенное время впрыска под нагрузкой."
    )

    st.session_state.mods = {"tuned": tuned, "decatted": decatted, "lpg": lpg}

    st.markdown("---")
    st.subheader("🔍 Режим диагностики")
    st.session_state.diagnostic_mode = st.radio(
        "Выберите контур проверки:",
        ["Механика (Группы 001-063)", "Электрика и CAN (Группы 125-135)"],
        index=0 if st.session_state.diagnostic_mode.startswith("Механика") else 1
    )

    if st.session_state.get("vin_code"):
        st.markdown("---")
        st.info(f"🆔 **Распознанный VIN:**\n`{st.session_state.vin_code}`")

    st.markdown("---")
    st.subheader("🧪 Симулятор")
    if st.session_state.diagnostic_mode.startswith("Электрика"):
        test_scenario = st.selectbox(
            "Сценарий:",
            ["Шина ОК (Все блоки на связи)", "Потеря связи с ABS", "Отвал приборки (Иммо)"]
        )
        mapping = {
            "Шина ОК (Все блоки на связи)": "normal",
            "Потеря связи с ABS": "can_loss_abs",
            "Отвал приборки (Иммо)": "immo_conflict"
        }
    else:
        test_scenario = st.selectbox(
            "Сценарий:",
            ["Исправный мотор", "Подсос воздуха", "Локальный пропуск (Катушка 2 цил.)",
             "Потеря компрессии (Цил 4 на ХХ)", "Умирающий бензонасос"]
        )
        mapping = {
            "Исправный мотор": "normal",
            "Подсос воздуха": "leak",
            "Локальный пропуск (Катушка 2 цил.)": "misfire_coil",
            "Потеря компрессии (Цил 4 на ХХ)": "compression_loss",
            "Умирающий бензонасос": "fuel_pump_death"
        }

    if st.button("⚡ Сгенерировать лог"):
        st.session_state.generated_log_df = generate_test_log_df(
            mapping[test_scenario],
            st.session_state.diagnostic_mode,
            st.session_state.is_base_trim,
            mods=st.session_state.mods
        )
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ Очистить историю"):
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt(
                st.session_state.diagnostic_mode, st.session_state.is_base_trim,
                mods=st.session_state.mods
            )}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Привет! Я твой виртуальный диагност VAG. 🚗"}]}
        ]
        st.session_state.vin_code = ""
        st.session_state.generated_log_df = None
        st.session_state.uploaded_image_key += 1
        clear_history_on_disk()
        st.rerun()

    if st.button("🚪 Выйти"):
        if "auth" in st.query_params:
            st.query_params.clear()
        if "chat_history" in st.session_state:
            del st.session_state.chat_history
        st.session_state.authenticated = False
        st.rerun()

# --- ОСНОВНОЙ ЭКРАН (без изменений) ---
st.title("VAG Expert Chat + Vision 💬")

for msg in st.session_state.chat_history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            for content_item in msg["content"]:
                if content_item["type"] == "text":
                    st.write(content_item["text"])
                elif content_item["type"] == "image_url":
                    st.image(content_item["image_url"]["url"], width=300)

st.markdown("---")
st.subheader("📁 Загрузка данных")
uploaded_file = st.file_uploader(
    "Лог VCDS (.csv/.txt) или скриншот (.png/.jpg)",
    type=["csv","txt","png","jpg","jpeg"],
    key=f"file_uploader_{st.session_state.uploaded_image_key}"
)

log_df = None
image_base64 = None

if uploaded_file is not None:
    if "text" in uploaded_file.type or "csv" in uploaded_file.type:
        file_bytes = uploaded_file.read()
        log_df, extracted_vin = parse_vcds_csv(file_bytes)
        if extracted_vin and extracted_vin != st.session_state.vin_code:
            st.session_state.vin_code = extracted_vin
            st.sidebar.info(f"📍 Найден VIN: {extracted_vin}")
            save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
    elif "image" in uploaded_file.type:
        image_base64 = encode_image_to_base64(uploaded_file)
        st.image(uploaded_file, caption="Превью скриншота VCDS", width=400)

if uploaded_file is None and st.session_state.generated_log_df is not None:
    log_df = st.session_state.generated_log_df

# --- ГРАФИКИ ---
if log_df is not None and not log_df.empty:
    st.success("📊 Данные лога успешно распознаны.")
    time_col = log_df.columns[0]

    with st.expander("📊 Посмотреть график параметров", expanded=True):
        if st.session_state.diagnostic_mode.startswith("Электрика"):
            selected_cols = [c for c in log_df.columns if "АКПП" in c or "АБС" in c or "Приборка" in c or "SRS" in c]
            if selected_cols:
                fig = px.line(log_df, x=time_col, y=selected_cols,
                              title="Статус CAN-связи (1=ОК, 0=обрыв, -1=блок отсутствует)",
                              template="plotly_dark")
                fig.update_yaxes(range=[-1.2, 1.2], tickvals=[-1, 0, 1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            selected_cols = [c for c in log_df.columns if "Давление" in c or "коррекция" in c
                             or "Пропуски" in c or "Откат" in c or "G187" in c or "G79" in c
                             or "Угол опережения" in c or "Угол дросселя" in c or "Напряжение ДД" in c
                             or "Температура ОЖ" in c or "Температура впуска" in c or "Фазовое положение" in c
                             or "Сопротивление Зонда" in c or "Конверсия катализатора" in c]
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Scatter(x=log_df[time_col], y=log_df[col], mode='lines', name=col))
                    try:
                        for ref_key, values in st.session_state.reference_map.items():
                            if len(values) == 4 and ref_key.lower() in col.lower():
                                low, high, c_low, c_high = values
                                fig.add_hline(y=low, line_dash="dot", line_color=c_low,
                                              annotation_text=f"Мин {ref_key}")
                                fig.add_hline(y=high, line_dash="dot", line_color=c_high,
                                              annotation_text=f"Макс {ref_key}")
                    except Exception:
                        st.warning("Не удалось построить линии заводских допусков для некоторых колонок.")
                fig.update_layout(template="plotly_dark", title="Параметры с допусками VAG",
                                  xaxis_title="Время (сек)", yaxis_title="Значение")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(log_df.head(10))

    btn_label = "Запустить экспертный анализ лога"
    if st.button(btn_label):
        if not API_KEY:
            st.error("API-ключ не найден!")
        else:
            log_summary_text = generate_log_summary(log_df, st.session_state.vin_code, st.session_state.is_base_trim)
            current_mods = st.session_state.mods
            system_instruction = get_system_prompt(
                mode=st.session_state.diagnostic_mode,
                is_base_trim=st.session_state.is_base_trim,
                mods=current_mods
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Привет! Я загрузил CSV-лог диагностики своего автомобиля. Вот статистический срез данных:\n\n{log_summary_text}\n\nПожалуйста, проанализируй эти параметры, найди скрытые аномалии, поставь диагноз и распиши пошаговый план ремонта."}
                ]}
            ]
            with st.spinner("VAG Expert анализирует лог..."):
                ai_response = ask_ai_chat(API_KEY, MODEL_NAME, messages, max_tokens=3000)
                st.markdown(ai_response)

# --- СКРИНШОТ ---
if image_base64 is not None:
    st.success("🖼️ Скриншот готов.")
    if st.button("👁️ Отправить скриншот"):
        if not API_KEY:
            st.error("API-ключ не найден!")
        else:
            prompt_text = "Пользователь загрузил скриншот окна программы VCDS. Распознай коды ошибок или группы измерений и выдай структурированный диагностический вердикт."
            if st.session_state.vin_code:
                prompt_text = f"VIN автомобиля: {st.session_state.vin_code}. " + prompt_text
            msg = {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]}
            st.session_state.chat_history.append(msg)
            temp_hist = copy.deepcopy(st.session_state.chat_history)
            with st.chat_message("assistant"):
                with st.spinner("Gemini анализирует скриншот..."):
                    resp = ask_ai_chat(API_KEY, MODEL_NAME, temp_hist, max_tokens=2000)
                    st.write(resp)
            st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": resp}]})
            save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
            st.session_state.uploaded_image_key += 1
            st.rerun()

# --- ЧАТ-ВВОД ---
if user_input := st.chat_input("Напишите симптомы или задайте вопрос..."):
    if not API_KEY:
        st.error("API-ключ не найден!")
    else:
        with st.chat_message("user"):
            st.write(user_input)

        mods = st.session_state.mods
        mods_list = []
        if mods["tuned"]: mods_list.append("Сделан Чип-тюнинг")
        if mods["decatted"]: mods_list.append("Удален катализатор")
        if mods["lpg"]: mods_list.append("Установлено ГБО")
        mods_str = ", ".join(mods_list) if mods_list else "Сток (без модификаций)"
        vin_str = st.session_state.vin_code if st.session_state.vin_code else "Не указан"
        base_str = "Да" if st.session_state.is_base_trim else "Нет"

        ai_text = (f"[Контекст VCDS - VIN: {vin_str}. Моды: {mods_str}. Базовая компл.: {base_str}] {user_input}")

        st.session_state.chat_history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
        temp_history = copy.deepcopy(st.session_state.chat_history)
        temp_history[-1]["content"][0]["text"] = ai_text

        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response = ask_ai_chat(API_KEY, MODEL_NAME, temp_history, max_tokens=1500)
                st.write(response)

        st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
        st.rerun()
