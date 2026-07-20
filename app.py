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

# --- ПРОВЕРКА ПИН-КОДА (автовход через ?auth=1) ---
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

# --- GOOGLE SHEETS: A1 профиль, B1 история ---
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

def save_profile(vin_code, is_base_trim, mods):
    sheet = _get_gsheet()
    if sheet is None:
        return
    try:
        profile = {"vin_code": vin_code, "is_base_trim": is_base_trim, "mods": mods}
        sheet.update("A1", [[json.dumps(profile, ensure_ascii=False)]])
    except Exception:
        pass

def load_profile():
    sheet = _get_gsheet()
    if sheet is None:
        return {"vin_code": "", "is_base_trim": True, "mods": {"tuned": False, "decatted": False, "lpg": False}}
    try:
        data_str = sheet.acell("A1").value
        if data_str:
            return json.loads(data_str)
    except Exception:
        pass
    return {"vin_code": "", "is_base_trim": True, "mods": {"tuned": False, "decatted": False, "lpg": False}}

def save_chat_history(history):
    sheet = _get_gsheet()
    if sheet is None:
        return
    try:
        sheet.update("B1", [[json.dumps(history, ensure_ascii=False)]])
    except Exception:
        pass

def load_chat_history():
    sheet = _get_gsheet()
    if sheet is None:
        return []
    try:
        data_str = sheet.acell("B1").value
        if data_str:
            return json.loads(data_str)
    except Exception:
        return []
    return []

def clear_chat_history():
    sheet = _get_gsheet()
    if sheet is None:
        return
    try:
        sheet.update("B1", [["[]"]])
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

# --- ДИНАМИЧЕСКИЙ REFERENCE MAP ---
def build_reference_map(is_base_trim=False, mods=None):
    mods = mods or {}
    map_low = 290 if is_base_trim else 315
    map_high = 340 if is_base_trim else 365
    ltft_limit = 8 if mods.get("lpg") else 5
    uoz_high = 12 if mods.get("tuned") else 10
    injector_high = 14 if mods.get("tuned") else 12

    return {
        "Давление ДАД (mbar)": (map_low, map_high, "green", "red"),
        "Время впрыска (мс)": (2.0, injector_high, "green", "red"),
        "Краткосрочная коррекция (%)": (-10.0, 10.0, "blue", "orange"),
        "Долговременная коррекция (%)": (-ltft_limit, ltft_limit, "blue", "orange"),
        "Пропуски": (0.0, 0.0, "green", "red"),
        "Угол дросселя (%)": (1.0, 3.5, "green", "red"),
        "Откат УОЗ": (0.0, 4.5, "green", "red"),
        "Напряжение ДД": (0.4, 3.5, "green", "red"),
        "Температура ОЖ (°C)": (85.0, 98.0, "green", "red"),
        "Температура впуска (°C)": (5.0, 65.0, "green", "red"),
        "Фазовое положение": (-3.0, 3.0, "green", "red"),
        "Сопротивление Зонда 1 (Ом)": (2.0, 15.0, "green", "red"),
        "Конверсия катализатора": (0.0, 0.45, "green", "red"),
        "Напряжение Зонда 1 (В)": (1.10, 2.10, "green", "red"),
        "Датчик дросселя 1 (G187) %": (3.0, 97.0, "green", "red"),
        "Педаль газа 1 (G79) %": (12.0, 94.0, "green", "red"),
    }

# --- ГЕНЕРАТОР ТЕСТОВЫХ ЛОГОВ ---
def generate_test_log_df(scenario="normal", diagnostic_mode="Механика (Группы 001-063)", is_base_trim=False, mods=None):
    if mods is None:
        mods = {"tuned": False, "decatted": False, "lpg": False}
    np.random.seed(42)
    time = np.arange(0, 100, 1.0)
    n_points = len(time)

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
        df = pd.DataFrame({
            "Отметка Времени (сек)": time,
            "АКПП (Группа 125-1)": akpp,
            "АБС (Группа 125-2)": abs_block,
            "Приборка (Группа 125-3)": priborka if 'priborka' in locals() else np.ones(n_points),
            "SRS (Группа 125-4)": srs_block,
            "Климат (Группа 126-1)": klimat,
            "Запрос Вентилятора % (Группа 135-1)": fan
        })
        return df

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
    misfire_status = np.ones(n_points)
    g187 = np.random.normal(7.0, 1.0, n_points)
    g188 = 100.0 - g187 + np.random.normal(0, 0.3, n_points)
    g79 = np.ones(n_points) * 14.0
    g185 = g79 / 2.0
    throttle = g187
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

    if mods.get("decatted", False):
        cat_status = np.full(n_points, -1)
        cat_conversion = np.zeros(n_points)

    if scenario == "detonation":
        rpm = np.linspace(2000, 5600, n_points)
        map_vals = np.linspace(800, 980, n_points)
        injector = np.linspace(6.0, 11.5, n_points)
        stft = np.random.normal(0, 1.5, n_points)
        ltft = np.ones(n_points) + 1.5
        g79 = np.linspace(14.0, 90.0, n_points)
        g185 = g79 / 2.0
        g187 = np.linspace(7.0, 90.0, n_points)
        g188 = 100.0 - g187 + np.random.normal(0, 0.3, n_points)
        throttle = g187
        uoz = np.clip(np.linspace(15, 28, n_points) + np.random.normal(0, 2, n_points), 10, 35)
        knock_all = np.clip(np.random.normal(5.0, 1.5, n_points), 3.0, 8.0)
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
        map_vals = 385.0 + np.random.normal(0, 6, n_points)
        injector = 3.10 + np.random.normal(0, 0.04, n_points)
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
        map_vals = 300.0 + np.random.normal(0, 5, n_points)
        injector = 1.85 + np.random.normal(0, 0.03, n_points)
        stft = -18.5 + np.random.normal(0, 1.0, n_points)
        ltft = np.ones(n_points) - 9.0
        cat_conversion = np.random.normal(0.7, 0.2, n_points)
        o2_voltage = np.clip(1.50 + stft * 0.02, 1.0, 2.5)
    elif scenario == "fuel_pump_death":
        rpm = np.linspace(840, 4500, n_points)
        map_vals = np.linspace(300.0, 850.0, n_points)
        injector = np.linspace(2.5, 4.8, n_points)
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
        misfire_c2 = np.clip(np.cumsum(np.random.choice([0,1,2], p=[0.7,0.2,0.1], size=n_points)), 0, 45)
        stft = np.linspace(0, 12.0, n_points)
        knock_c2 = np.clip(np.random.normal(4.0, 1.0, n_points), 2.0, 7.0)
        dd_c2 = dd_base + 0.4 + np.random.normal(0, 0.1, n_points)
        o2_heater_resistance = np.full(n_points, 99.9)
        o2_voltage = np.clip(1.50 + stft * 0.02, 1.0, 2.5)
    elif scenario == "compression_loss":
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
    return df

# --- ДИНАМИЧЕСКИЙ СИСТЕМНЫЙ ПРОМПТ ---
def get_system_prompt(mode="Механика (Группы 001-063)", is_base_trim=False, ecu_type="Magneti Marelli 7GV", mods=None):
    if mods is None:
        mods = {"tuned": False, "decatted": False, "lpg": False}

    base_prompt = f"""Ты — профессиональный автодиагност концерна VAG (уровень дилерского центра), специализирующийся на работе с логами VCDS (Вася Диагност).
Текущий блок управления двигателем: {ecu_type}.
Твоя задача — анализировать логи и выдавать точные технические диагнозы по базе параметров.
"""
    anti_hallucination_filter = """
[🚨] КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ АРХИТЕКТУРЫ ДВИГАТЕЛЯ:
Двигатель CFNA 1.6 — это исключительно атмосферный бензиновый мотор с обычным РАСПРЕДЕЛЕННЫМ впрыском (MPI).
ТЕБЕ КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО упоминать, рекомендовать к проверке или использовать в диагностике следующие сущности:
1. ТНВД и датчики высокого давления (никаких упоминаний систем TSI/TFSI/FSI). Группа 106 на этом моторе неинформативна.
2. Турбины, интеркулеры, вестгейты, байпасы и давление наддува (мотор атмосферный).
3. Дизельные форсунки, системы Common Rail, ТНВД дизеля, балансировку цилиндров (никаких групп 013, 072-077).
4. Фазорегуляторы, клапаны N205, муфты VVT и цепи с изменяемыми фазами (здесь жесткие фиксированные звезды, группы 091-092 отсутствуют).
5. Датчики массового расхода воздуха (ДМРВ / MAF) — воздух считается строго по ДАД (MAP, группа 002).

Если ты зафиксируешь пропуски, детонацию или бедную смесь, ты обязан предлагать проверку ТОЛЬКО тех узлов и групп (из списка 15 групп), которые физически существуют на данном CFNA. За предложение проверить ТНВД, ДМРВ или Фазорегулятор на этом моторе твой диагноз будет аннулирован.
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
- УСТАНОВЛЕНО ГБО (Газ): Для газа долговременные коррекции (032) в пределах ±8.0% являются допустимой нормой. Не ставь диагноз 'подсос' или 'умирающий бензонасос' только на основании этих значений.
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

ГРУППЫ 032, 033 (Топливные коррекции Лямбда):
- 032-1 (Аддитивная коррекция, LTFT ХХ): Норма = -3.0% … +3.0%. Если > +4.5% при нормальном мультипликативе — это 100% подсос воздуха на холостом ходу (клапан ВКГ, кольца форсунок). Если < -4.5% — перелив топлива.
- 032-2 (Мультипликативная коррекция, LTFT Нагрузка): Норма = -5.0% … +5.0%. Если > +6.5% — мотору не хватает бензина в движении (умирает насос или забит фильтр). Для машин с ГБО (lpg=True) допуск расширяется до ±8.0%.
- 033-1 (Мгновенная лямбда-коррекция, STFT): Норма = ±10.0% (должна быстро колебаться). Зависание в +25% — жесткое обеднение смеси, в -25% — жесткое переобогащение.
- 033-2 (Напряжение широкополосного Зонда 1): Стехиометрия ≈ 1.50 В. >2.0 В — бедная смесь, <1.1 В — богатая смесь.

ГРУППА 003 (Зажигание и дроссель):
- 003-3 (Угол дросселя): 1.0–3.5% на ХХ. >3.5% — нагар, <1.0% — подсос.
- 003-4 (УОЗ): 4.0–12.0° до ВМТ, скачет — норма.

ГРУППЫ 014, 015, 016 (Пропуски воспламенения):
- 014-1 (Обороты): 650–750 об/мин.
- 014-2 (Нагрузка): 15.0–18.5%.
- 014-3 (Суммарные пропуски): В норме строго 0.
- 014-4 (Статус распознавания пропусков): В норме **ON / Activated**. Если статус **OFF / Deactivated**, система намеренно отключена ЭБУ.
   **Важное примечание для CFNA:** При горящей лампе низкого уровня топлива (< 7–8 л) ЭБУ принудительно выключает распознавание пропусков, чтобы избежать ложных срабатываний из‑за «отлива» топлива в баке.
   Если 014-4 = OFF и все счётчики 015/016 равны 0, **обязательно предупреди пользователя**, что диагностика пропусков временно невозможна, и порекомендуй заправить автомобиль (минимум 1/4 бака) и повторить запись лога.
- 015-1...015-3, 016-1 (Счётчики по цилиндрам 1–4): В норме 0.
- Если пропуски растут только в одном цилиндре под нагрузкой — виновна катушка или свеча этого цилиндра. Используй метод перестановки катушек для проверки.
- Если пропуски идут хаотично по всем цилиндрам в сочетании с коррекциями 032 > +8% — смесь слишком бедная (проверяй бензонасос, фильтр или ГБО).
- Если пропуски строго в одном цилиндре только на ХХ, а на оборотах пропадают — проверяй механику (потеря компрессии, клапаны).

ГРУППЫ 062, 063 (Дроссель и Педаль газа):
- 062-1 (Датчик 1 дросселя G187): Растет от ~3-12% (ХХ) до ~88-97% (полный газ).
- 062-2 (Датчик 2 дросселя G188): Падает от ~88-97% (ХХ) до ~3-12% (полный газ). Сумма 062-1 + 062-2 всегда должна быть строго около 100% (допуск ±2%). Нарушение суммы = износ дорожек дросселя или окисление контактов.
- 062-3 (Датчик 1 педали G79): Норма ХХ = 12-16%. Полный газ = 88-94%.
- 062-4 (Датчик 2 педали G185): Должен быть строго в 2 раза меньше, чем 062-3 (соотношение 2:1 на всем ходу). Нарушение пропорции уводит мотор в аварию EPC.
- 063-3 (Статус адаптации): Должен быть строго 'ADP. OK'. Статус 'Error' требует принудительной адаптации через Группу 060 (инструкция ниже).

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
- Если IAT > +70°C в пробках и есть откаты УОЗ (020) — ЭБУ позднит зажигание из-за перегрева впуска.

--- ЛОГИКА КРОСС-ВАЛИДАЦИИ (расширенная) ---
1. ЛОКАЛЬНЫЙ ПРОБОЙ КАТУШКИ/СВЕЧИ: ЕСЛИ пропуски быстро растут только в одном цилиндре (015-1...016-1), особенно под нагрузкой, И мгновенная лямбда (033-1) кратковременно уходит в плюс, ТОГДА: проблема в катушке или свече этого цилиндра. Переставить катушку с соседним цилиндром и проверить, переместятся ли пропуски.
2. ХАОТИЧНЫЕ ПРОПУСКИ ИЗ-ЗА БЕДНОЙ СМЕСИ: ЕСЛИ пропуски хаотично возникают по всем цилиндрам И коррекции (032-1/2) > +8.0%, ТОГДА: системное обеднение смеси (бензонасос, топливный фильтр, подсос воздуха). Катушки и свечи исправны.
3. ПОТЕРЯ КОМПРЕССИИ: ЕСЛИ постоянные пропуски в одном цилиндре ТОЛЬКО на ХХ, а при повышении оборотов (>2000) пропуски прекращаются, И MAP > 320 мбар, ТОГДА: потеря компрессии в этом цилиндре. Замерить компрессометром.
4. СКРЫТЫЙ ПОДСОС: ЕСЛИ аддитив (032-1) > +4.5% И мультипликатив (032-2) в норме (±1.5%) И MAP > 315 мбар, ТОГДА: Подсос за дросселем (клапан ВКГ, кольца форсунок). Проверить дымогенератором.
5. УМИРАЮЩИЙ БЕНЗОНАСОС / ЗАБИТЫЙ ФИЛЬТР: ЕСЛИ аддитив в норме И мультипликатив > +6.5% И мгновенная лямбда (033-1) под нагрузкой зависает в +20…+25%, а напряжение зонда (033-2) > 2.10 В, ТОГДА: дефицит топлива. Замерить давление в рампе (норма 4.0 бар).
6. ПЕРЕЛИВ ТОПЛИВА: ЕСЛИ аддитив < -4.5% ИЛИ мультипликатив < -5.0% И напряжение зонда (033-2) < 1.10 В, ТОГДА: переобогащение смеси. Проверить форсунки на утечку, воздушный фильтр, давление в рампе (не должно быть >4.5 бар).
7. РАССИНХРОН ДРОССЕЛЯ/ПЕДАЛИ: Нарушение пропорций 100% или 2:1 → EPC. Замена узла или адаптация (060).
8. ИЗНОС ДОРОЖЕК ДРОССЕЛЯ (ОШИБКА EPC): ЕСЛИ сумма 062-1 + 062-2 выходит за пределы 98-102%, ТОГДА: износ потенциометров или окисление контактов. Почистить разъём, при повторении ошибки — замена дроссельного узла.
9. СБОЙ ДАТЧИКА ПЕДАЛИ ГАЗА: ЕСЛИ соотношение 062-3 и 062-4 не равно 2:1 (допуск ±5%), ТОГДА: отказ датчиков педали. Проверить проводку, при необходимости заменить модуль педали.
10. СРЫВ АДАПТАЦИИ ЗАСЛОНКИ: ЕСЛИ 063-3 = 'Error' или после чистки дросселя плавают обороты, ТОГДА: выполнить адаптацию через Группу 060 (зажигание вкл, мотор заглушен, температура ОЖ <90°C, педаль отпущена). Дождаться статуса ADP. OK, затем выключить зажигание на 20 сек.
11. ПРЕДУПРЕЖДЕНИЕ О НИЗКОМ УРОВНЕ ТОПЛИВА: ЕСЛИ в Группе 014-4 статус равен OFF / Deactivated И все счётчики пропусков (015-1…016-1) равны 0, ТОГДА: Вероятно, включена лампа низкого уровня топлива. ЭБУ намеренно отключил мониторинг пропусков. Диагностика пропусков невозможна. Порекомендуй заправить автомобиль (минимум до 1/4 бака) и повторить съём лога после того, как статус 014-4 станет ON / Activated.

--- ИНСТРУКЦИЯ: АДАПТАЦИЯ ДРОССЕЛЬНОЙ ЗАСЛОНКИ (Группа 060) ---
Если требуется адаптация:
1. Убедись, что ошибки в ЭБУ удалены, двигатель заглушен, зажигание включено.
2. **Температура ОЖ должна быть от 5°C до 90°C (лучше выполнять на холодном или чуть тёплом моторе). Адаптация дросселя (Гр. 060) блокируется, если температура ОЖ > 90°C.**
3. Педаль газа полностью отпущена.
4. В VCDS: 01-Двигатель → Базовые параметры (04) → Группа 060 → Нажать «Прочитать» (Go!).
5. Дождаться, пока статус сменится с ADP. RUN на ADP. OK (заслонка будет жужжать).
6. Выйти из группы, выключить зажигание на 20 секунд, затем завести двигатель.
"""

    can_rules = """--- БАЗА ЭТАЛОНОВ CAN-ШИНЫ (Группы 125-135) ---
Норма связи: 1 = блок отвечает, 0 = нет связи.
- 125-1 (АКПП), 125-2 (АБС/ESP), 126-1 (Климат). В базовой комплектации эти блоки отсутствуют, их значение = -1.
- 125-3 (Приборка): Если 0 или скачет 0/1 — иммо блокирует пуск.
- 135-1 (Запрос вентилятора): 100% при молчащем кулере = сгорел БУВ.
"""
    common_rules = "\nОБЩИЕ ПРАВИЛА: Отвечай профессионально, структурированно. Учитывай модификации и комплектацию."

    if mode.startswith("Электрика"):
        return base_prompt + anti_hallucination_filter + config_note + mods_note + can_rules + common_rules
    else:
        return base_prompt + anti_hallucination_filter + config_note + mods_note + mechanics_rules + common_rules

# --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЙ ---
profile = load_profile()
saved_history = load_chat_history()
saved_vin = profile.get("vin_code", "")

if "is_base_trim" not in st.session_state:
    st.session_state.is_base_trim = profile.get("is_base_trim", True)
if "mods" not in st.session_state:
    st.session_state.mods = profile.get("mods", {"tuned": False, "decatted": False, "lpg": False})
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
    st.session_state.vin_code = saved_vin
if "reference_map" not in st.session_state:
    st.session_state.reference_map = build_reference_map(
        st.session_state.is_base_trim, st.session_state.mods
    )
if "generated_log_df" not in st.session_state:
    st.session_state.generated_log_df = None
if "uploaded_image_key" not in st.session_state:
    st.session_state.uploaded_image_key = 0

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Конфигурация автомобиля")
    st.write("Настройте параметры машины для точной работы ИИ.")
    st.markdown("---")
    st.subheader("📦 Комплектация")
    prev_base = st.session_state.is_base_trim
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
    prev_mods = dict(st.session_state.mods)
    st.session_state.mods = {"tuned": tuned, "decatted": decatted, "lpg": lpg}

    # Обновляем system prompt и reference map при изменении настроек
    settings_changed = (prev_base != st.session_state.is_base_trim or 
                        prev_mods != st.session_state.mods)
    if settings_changed and len(st.session_state.chat_history) > 0:
        st.session_state.reference_map = build_reference_map(
            st.session_state.is_base_trim, st.session_state.mods
        )
        st.session_state.chat_history[0] = {
            "role": "system",
            "content": [{"type": "text", "text": get_system_prompt(
                st.session_state.diagnostic_mode,
                st.session_state.is_base_trim,
                mods=st.session_state.mods
            )}]
        }

    save_profile(st.session_state.vin_code, st.session_state.is_base_trim, st.session_state.mods)

    st.markdown("---")
    st.subheader("🔍 Режим диагностики")
    prev_mode = st.session_state.diagnostic_mode
    st.session_state.diagnostic_mode = st.radio(
        "Выберите контур проверки:",
        ["Механика (Группы 001-063)", "Электрика и CAN (Группы 125-135)"],
        index=0 if st.session_state.diagnostic_mode.startswith("Механика") else 1
    )
    if prev_mode != st.session_state.diagnostic_mode and len(st.session_state.chat_history) > 0:
        st.session_state.chat_history[0] = {
            "role": "system",
            "content": [{"type": "text", "text": get_system_prompt(
                st.session_state.diagnostic_mode,
                st.session_state.is_base_trim,
                mods=st.session_state.mods
            )}]
        }
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
        clear_chat_history()
        st.rerun()
    if st.button("🚪 Выйти"):
        if "auth" in st.query_params:
            st.query_params.clear()
        if "chat_history" in st.session_state:
            del st.session_state.chat_history
        st.session_state.authenticated = False
        st.rerun()

# --- ОСНОВНОЙ ЭКРАН ---
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
            save_profile(st.session_state.vin_code, st.session_state.is_base_trim, st.session_state.mods)
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
                             or "Сопротивление Зонда" in c or "Конверсия катализатора" in c
                             or "Напряжение Зонда" in c or "Датчик дросселя" in c or "Педаль газа" in c]
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
            user_msg = {"role": "user", "content": [{"type": "text", "text": f"Привет! Я загрузил CSV-лог диагностики своего автомобиля. Вот статистический срез данных:\n\n{log_summary_text}\n\nПожалуйста, проанализируй эти параметры, найди скрытые аномалии, поставь диагноз и распиши пошаговый план ремонта."}]}
            st.session_state.chat_history.append(user_msg)

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
                user_msg
            ]
            with st.spinner("VAG Expert анализирует лог..."):
                ai_response = ask_ai_chat(API_KEY, MODEL_NAME, messages, max_tokens=3000)
                st.markdown(ai_response)

            st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": ai_response}]})
            save_chat_history(st.session_state.chat_history)
            st.rerun()

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
            save_chat_history(st.session_state.chat_history)
            st.session_state.uploaded_image_key += 1
            st.rerun()

# --- ЧАТ-ВВОД ---
if user_input := st.chat_input("Напишите симптомы или задайте вопрос..."):
    if not API_KEY:
        st.error("API-ключ не найден!")
    else:
        with st.chat_message("user"):
            st.write(user_input)

        vin_match = re.search(r'\b([A-HJ-NPR-Z0-9]{17})\b', user_input, re.IGNORECASE)
        if vin_match and vin_match.group(1).upper() != st.session_state.vin_code:
            st.session_state.vin_code = vin_match.group(1).upper()
            save_profile(st.session_state.vin_code, st.session_state.is_base_trim, st.session_state.mods)
            st.sidebar.info(f"📍 VIN обновлён из сообщения: {st.session_state.vin_code}")

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
        save_chat_history(st.session_state.chat_history)
        st.rerun()
