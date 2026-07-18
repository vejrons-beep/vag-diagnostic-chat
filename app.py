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
from PIL import Image
import numpy as np

# Настройка страницы Streamlit
st.set_page_config(page_title="VAG Expert Chat + Vision", page_icon="🚗", layout="wide")

# --- КОНСТАНТЫ И НАСТРОЙКИ ---
MODEL_NAME = "google/gemini-2.5-flash"
API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
CACHE_FILE = "chat_history_cache.json"

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С ПОСТОЯННОЙ ПАМЯТЬЮ ---
def save_history_to_disk(history, vin_code):
    try:
        data = {"vin_code": vin_code, "chat_history": history}
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        pass

def load_history_from_disk():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("chat_history", []), data.get("vin_code", "")
        except Exception as e:
            return [], ""
    return [], ""

def clear_history_on_disk():
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception as e:
            pass

# --- УЛУЧШЕННЫЙ ПАРСЕР РЕАЛЬНЫХ ЛОГОВ VCDS ---
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

# --- СТАТИСТИЧЕСКИЙ СЖАТЫЙ АНАЛИЗ С УЧЕТОМ РЕЖИМА ---
def generate_log_summary(df, vin="", mode="Механика (Группы 001-063)"):
    summary = []
    summary.append(f"Данные CSV-лога (Статистический срез). VIN: {vin if vin else 'Не указан'}")
    time_col = df.columns[0]
    summary.append(f"Всего точек измерения: {len(df)}. Длительность: {df[time_col].max() - df[time_col].min():.1f} сек.")
    
    for col in df.columns[1:]:
        # Фильтрация параметров для экономии контекста токенов ИИ
        if mode == "Механика (Группы 001-063)" and any(x in col for x in ["АКПП", "АБС", "Приборка", "SRS", "Климат"]):
            continue
        if mode == "Электрика и CAN (Группы 125-135)" and any(x in col for x in ["Давление", "коррекция", "впрыска", "Откат", "Пропуски"]):
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
        mime_type = uploaded_file.type
        return f"data:{mime_type};base64,{encoded_string}"
    return None

def ask_ai_chat(api_key, model_name, messages, max_tokens=2000):
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

# --- УЛУЧШЕННАЯ СВЯЗКА НАГРУЗКИ И MAP В СИМУЛЯТОРЕ ---
def generate_test_log_df(scenario="normal", diagnostic_mode="Механика (Группы 001-063)", is_base_trim=True):
    np.random.seed(42)
    time = np.arange(0, 100, 1.0)
    n_points = len(time)
    
    if diagnostic_mode == "Электрика и CAN (Группы 125-135)":
        base_can = 0 if is_base_trim else 1
        if scenario == "can_loss_abs":
            df = pd.DataFrame({
                "Отметка Времени (сек)": time,
                "АКПП (Группа 125-1)": np.ones(n_points) * base_can,
                "АБС (Группа 125-2)": np.zeros(n_points),
                "Приборка (Группа 125-3)": np.ones(n_points),
                "SRS (Группа 125-4)": np.ones(n_points),
                "Климат (Группа 126-1)": np.ones(n_points) * base_can,
                "Запрос Вентилятора % (Группа 135-1)": np.zeros(n_points)
            })
        elif scenario == "immo_conflict":
             df = pd.DataFrame({
                "Отметка Времени (сек)": time,
                "АКПП (Группа 125-1)": np.ones(n_points) * base_can,
                "АБС (Группа 125-2)": np.ones(n_points) * base_can, 
                "Приборка (Группа 125-3)": np.random.choice([0, 1], p=[0.8, 0.2], size=n_points),
                "SRS (Группа 125-4)": np.ones(n_points),
                "Климат (Группа 126-1)": np.ones(n_points) * base_can,
                "Запрос Вентилятора % (Группа 135-1)": np.zeros(n_points)
            })
        else:
            df = pd.DataFrame({
                "Отметка Времени (сек)": time,
                "АКПП (Группа 125-1)": np.ones(n_points) * base_can,
                "АБС (Группа 125-2)": np.ones(n_points) * base_can, 
                "Приборка (Группа 125-3)": np.ones(n_points),
                "SRS (Группа 125-4)": np.ones(n_points),
                "Климат (Группа 126-1)": np.ones(n_points) * base_can,
                "Запрос Вентилятора % (Группа 135-1)": np.zeros(n_points)
            })
        return df

    # Механика
    g79 = np.ones(n_points) * 14.5
    g185 = g79 / 2
    g187 = np.ones(n_points) * 4.5
    g188 = 100 - g187

    rpm = 840 + np.random.normal(0, 8, n_points)
    map_vals = (290.0 if is_base_trim else 305.0) + np.random.normal(0, 5, n_points)
    
    # Прямая физическая связь Нагрузки и давления MAP
    load_vals = (15.0 if is_base_trim else 18.5) + (map_vals - (290.0 if is_base_trim else 305.0)) * 0.1
    
    injector = 2.25 + np.random.normal(0, 0.04, n_points)
    stft = np.random.normal(0, 1.0, n_points)
    ltft = np.zeros(n_points) + 0.8
    misfire_c1 = np.zeros(n_points)
    misfire_c2 = np.zeros(n_points)
    misfire_c3 = np.zeros(n_points)
    misfire_c4 = np.zeros(n_points)

    if scenario == "detonation":
        rpm = np.linspace(2000, 5600, n_points) + np.random.normal(0, 15, n_points)
        map_vals = np.linspace(800, 980, n_points) + np.random.normal(0, 5, n_points)
        load_vals = np.linspace(45.0, 92.0, n_points) + np.random.normal(0, 1, n_points)
        injector = np.linspace(6.0, 11.5, n_points) + np.random.normal(0, 0.1, n_points)
        stft = np.random.normal(0, 1.5, n_points)
        ltft = np.zeros(n_points) + 1.5
        g79 = np.linspace(14.5, 90.0, n_points)
        g185 = g79 / 2
        g187 = np.linspace(4.5, 88.0, n_points)
        g188 = 100 - g187
    elif scenario == "leak":
        map_vals = 385.0 + np.random.normal(0, 6, n_points)
        load_vals = 27.5 + np.random.normal(0, 0.8, n_points) # Нагрузка растет из-за падения вакуума
        injector = 3.10 + np.random.normal(0, 0.04, n_points)
        stft = 15.2 + np.random.normal(0, 1.2, n_points)
        ltft = np.zeros(n_points) + 6.5
    elif scenario == "rich":
        map_vals = 300.0 + np.random.normal(0, 5, n_points)
        load_vals = 18.0 + np.random.normal(0, 0.5, n_points)
        injector = 1.85 + np.random.normal(0, 0.03, n_points)
        stft = -18.5 + np.random.normal(0, 1.0, n_points)
        ltft = np.zeros(n_points) - 9.0
    elif scenario == "fuel_pump_death":
        rpm = np.linspace(840, 4500, n_points)
        map_vals = np.linspace(300.0, 850.0, n_points) + np.random.normal(0, 5, n_points)
        load_vals = np.linspace(18.0, 85.0, n_points)
        injector = np.linspace(2.5, 4.8, n_points) + np.random.normal(0, 0.1, n_points)
        stft = np.linspace(2.0, 24.0, n_points) + np.random.normal(0, 1.0, n_points) 
        ltft = np.zeros(n_points) + 8.5 
        g79 = np.linspace(14.5, 70.0, n_points)
        g185 = g79 / 2
        g187 = np.linspace(4.5, 60.0, n_points)
        g188 = 100 - g187
        misfire_c1 = np.clip(np.cumsum(np.random.choice([0, 1], p=[0.9, 0.1], size=n_points)), 0, 10)
        misfire_c4 = np.clip(np.cumsum(np.random.choice([0, 1], p=[0.9, 0.1], size=n_points)), 0, 10)
    elif scenario == "misfire_coil":
        misfire_c2 = np.clip(np.cumsum(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1], size=n_points)), 0, 45)
        stft = np.linspace(0, 12.0, n_points) + np.random.normal(0, 1.0, n_points) 
    elif scenario == "compression_loss":
        rpm = np.concatenate([np.ones(50)*840, np.linspace(840, 2500, 50)])
        map_vals = np.concatenate([np.ones(50)*375.0, np.linspace(375.0, 500.0, 50)]) + np.random.normal(0, 3, n_points)
        load_vals = np.concatenate([np.ones(50)*28.0, np.linspace(28.0, 55.0, 50)])
        misfire_c4 = np.concatenate([np.cumsum(np.random.choice([0, 1], p=[0.5, 0.5], size=50)), np.ones(50)*25])

    df = pd.DataFrame({
        "Отметка Времени (сек)": time,
        "Обороты двигателя (об/мин)": np.round(rpm, 0),
        "Нагрузка двигателя (%)": np.round(load_vals, 1),
        "Давление ДАД (mbar)": np.round(map_vals, 1),
        "Время впрыска (мс)": np.round(injector, 2),
        "Краткосрочная коррекция (%)": np.round(stft, 2),
        "Долговременная коррекция (%)": np.round(ltft, 2),
        "Дроссель 1 (G187) %": np.round(g187, 1),
        "Дроссель 2 (G188) %": np.round(g188, 1),
        "Педаль 1 (G79) %": np.round(g79, 1),
        "Педаль 2 (G185) %": np.round(g185, 1),
        "Пропуски Цилиндр 1": np.round(misfire_c1, 0),
        "Пропуски Цилиндр 2": np.round(misfire_c2, 0),
        "Пропуски Цилиндр 3": np.round(misfire_c3, 0),
        "Пропуски Цилиндр 4": np.round(misfire_c4, 0)
    })

    return df

# --- ДИНАМИЧЕСКИЙ СИСТЕМНЫЙ ПРОМПТ ---
def get_system_prompt(mode="Механика (Группы 001-063)", is_base_trim=False, ecu_type="Magneti Marelli 7GV"):
    base_prompt = f"""Ты — профессиональный автодиагност концерна VAG (уровень дилерского центра), специализирующийся на работе с логами VCDS (Вася Диагност).
Текущий блок управления двигателем: {ecu_type}.
Твоя задача — анализировать логи и выдавать точные технические диагнозы по базе параметров.
"""
    
    config_note = ""
    if is_base_trim:
        config_note = """
[!] ВАЖНО: АВТОМОБИЛЬ В БАЗОВОЙ КОМПЛЕКТАЦИИ (МКПП, БЕЗ КОНДИЦИОНЕРА, БЕЗ ABS).
В группах 125 и 126 значения 0 или N/A для ABS, Климата и АКПП являются АБСОЛЮТНОЙ НОРМОЙ. Не фиксируй это как потерю связи!
Нагрузка на ХХ должна быть строго 15-18%, а MAP строго 280-300 мбар (паразитных нагрузок нет).
"""

    mechanics_rules = """--- БАЗА ЭТАЛОНОВ CFNA 1.6 MPI ---
ГРУППА 002 (Воздух и Нагрузка):
- 002-2 (Нагрузка): 15.0–25.0%.
- 002-4 (ДАД / MAP): 280–340 мбар. Если >360 мбар — подсос воздуха, проскок цепи ГРМ, потеря компрессии.

ГРУППЫ 032, 033 (Топливо):
- 032-1 (Аддитив / Краткосрочная): ±3.0%. > +4.0% = подсос. < -4.0% = перелив форсунок.
- 032-2 (Мультипликатив / Долговременная): ±5.0%. > +6.0% = нехватка топлива (насос/фильтр).
- 033-1 (Мгновенная лямбда): Зависание в +25% = крайне бедная смесь.

ГРУППЫ 014, 015, 016 (ЗАЖИГАНИЕ И ПРОПУСКИ):
- 014-3 (Суммарные пропуски): 0.
- 015-1, 015-2, 015-3, 016-1 (Счетчики Цил 1-4): Строго 0.

ГРУППЫ 062, 063 (Дроссель/Педаль):
- 062-1 + 062-2 = 100% (зеркальность датчиков G187 и G188).
- 062-3 = 062-4 * 2 (педаль датчики G79 и G185 имеют соотношение 2:1).

--- ЛОГИКА КРОСС-ВАЛИДАЦИИ ---
1. ПРОБОЙ КАТУШКИ/СВЕЧИ: ЕСЛИ Пропуски (015/016) быстро растут ТОЛЬКО в одном цилиндре И Мгновенная лямбда (033-1) уходит в плюс, ТОГДА: Локальный пропуск. Рекомендация: переставить катушку на другой цилиндр.
2. ПРОПУСКИ ИЗ-ЗА БЕДНОЙ СМЕСИ: ЕСЛИ Пропуски хаотичны по всем цилиндрам И Коррекции (032-1/2) > +8.0%, ТОГДА: Системное обеднение. Катушки целы, проблема в бензонасосе или подсосе.
3. ПОТЕРЯ КОМПРЕССИИ: ЕСЛИ постоянные пропуски в ОДНОМ цилиндре ТОЛЬКО на холостом ходу (на оборотах >2000 счетчик стоит) И MAP > 360 мбар, ТОГДА: Механическая потеря компрессии (клапан/кольца). Замер компрессометром.
4. СКРЫТЫЙ ПОДСОС: ЕСЛИ Аддитив > +5.0% И Мультипликатив в норме И MAP > 350 мбар, ТОГДА: Подсос за дросселем (клапан ВКГ).
5. БЕНЗОНАСОС: ЕСЛИ Аддитив в норме И Мультипликатив > +7.0% И Мгновенная уходит в +25% под нагрузкой, ТОГДА: Дефицит топлива (замер давления, норма 4.0 бар).
6. РАССИНХРОН ДРОССЕЛЯ/ПЕДАЛИ: Нарушение пропорций 100% или 2:1 ведет к EPC. Замена узла или адаптация (060)."""

    can_rules = """--- БАЗА ЭТАЛОНОВ CAN-ШИНЫ (Группы 125-135) ---
Норма связи: 1. 0 = Нет связи.
- 125-1 (АКПП), 125-2 (АБС/ESP), 126-1 (Климат).
- 125-3 (Приборка): Если 0 или скачет 0/1 — иммо блокирует пуск, авто глохнет.
- 135-1 (Запрос вентилятора): 100% при молчащем кулере = сгорел БУВ.
ЛОГИКА:
1. ТОТАЛЬНЫЙ СБОЙ: ЕСЛИ все параметры = 0, ТОГДА Обрыв шины CAN-Drive или обесточивание ЭБУ.
2. ЛОКАЛЬНЫЙ ОБРЫВ: ЕСЛИ только один блок = 0, ТОГДА обрыв провода к блоку или предохранитель."""

    common_rules = """\nОБЩИЕ ПРАВИЛА: Отвечай профессионально, структурированно. Учитывай модификации и комплектацию."""

    if mode == "Электрика и CAN (Группы 125-135)":
        return base_prompt + config_note + can_rules + common_rules
    else:
        return base_prompt + config_note + mechanics_rules + common_rules

# --- ИНИЦИАЛИЗАЦИЯ СЕССИИ ---
saved_history, saved_vin = load_history_from_disk()

if "diagnostic_mode" not in st.session_state:
    st.session_state.diagnostic_mode = "Механика (Группы 001-063)"
if "is_base_trim" not in st.session_state:
    st.session_state.is_base_trim = True
if "ecu_type" not in st.session_state:
    st.session_state.ecu_type = "Magneti Marelli 7GV"

if "chat_history" not in st.session_state:
    if saved_history:
        st.session_state.chat_history = saved_history
    else:
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt(st.session_state.diagnostic_mode, st.session_state.is_base_trim, st.session_state.ecu_type)}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"Привет! Я твой виртуальный диагност VAG VCDS. 🚗\nОпиши симптомы, загрузи CSV-лог или скинь скриншот."}]}
        ]

if "vin_code" not in st.session_state:
    st.session_state.vin_code = saved_vin if saved_vin else ""

if "generated_log_df" not in st.session_state:
    st.session_state.generated_log_df = None

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    
    st.markdown("---")
    st.subheader("🧠 Режим ИИ-Диагноста")
    
    # Исправление бага: Запрос на очистку при смене режима
    new_mode = st.radio(
        "Выберите профиль диагностики:",
        ("Механика (Группы 001-063)", "Электрика и CAN (Группы 125-135)")
    )
    
    new_trim = st.checkbox("Базовая комплектация (МКПП, без ABS и A/C)", value=st.session_state.is_base_trim)
    
    new_ecu = st.selectbox(
        "Блок управления двигателем (ЭБУ):",
        ["Magneti Marelli 7GV"]
    )
    
    if new_mode != st.session_state.diagnostic_mode or new_trim != st.session_state.is_base_trim or new_ecu != st.session_state.ecu_type:
        st.session_state.diagnostic_mode = new_mode
        st.session_state.is_base_trim = new_trim
        st.session_state.ecu_type = new_ecu
        
        if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[0]["role"] == "system":
            st.session_state.chat_history[0]["content"][0]["text"] = get_system_prompt(new_mode, new_trim, new_ecu)
            
        st.warning("⚠️ Профиль изменен! Рекомендуется очистить историю для точного контекста.")
        if st.button("🗑️ Сбросить диалог сейчас"):
            st.session_state.chat_history = [
                {"role": "system", "content": [{"type": "text", "text": get_system_prompt(new_mode, new_trim, new_ecu)}]},
                {"role": "assistant", "content": [{"type": "text", "text": "История сброшена под новый профиль. Слушаю вас!"}]}
            ]
            st.rerun()
        st.rerun()

    st.markdown("---")
    st.subheader("🚗 Идентификация автомобиля")
    vin_input = st.text_input("Ввести VIN-код:", value=st.session_state.vin_code, max_chars=17)
    if vin_input != st.session_state.vin_code:
        st.session_state.vin_code = vin_input.upper()

    st.markdown("---")
    st.subheader("🔧 Модификации")
    is_tuned = st.checkbox("⚙️ Чип-тюнинг")
    is_decatted = st.checkbox("💨 Удален катализатор")
    is_lpg = st.checkbox("🔥 Установлено ГБО")
    st.session_state.mods = {"tuned": is_tuned, "decatted": is_decatted, "lpg": is_lpg}

    st.markdown("---")
    st.subheader("🧪 Симулятор")
    
    if st.session_state.diagnostic_mode == "Электрика и CAN (Группы 125-135)":
        test_scenario = st.selectbox(
            "Сценарий (Электрика):",
            ["Шина ОК (Все блоки на связи)", "Потеря связи с ABS", "Отвал приборки (Иммо)"]
        )
        mapping = {
            "Шина ОК (Все блоки на связи)": "normal",
            "Потеря связи with ABS": "can_loss_abs",
            "Отвал приборки (Иммо)": "immo_conflict"
        }
    else:
        test_scenario = st.selectbox(
            "Сценарий (Механика):",
            ["Исправный мотор", "Подсос воздуха", "Локальный пропуск (Катушка 2 цил.)", "Потеря компрессии (Цил 4 на ХХ)", "Умирающий бензонасос"]
        )
        mapping = {
            "Исправный мотор": "normal",
            "Подсос воздуха": "leak",
            "Локальный пропуск (Катушка 2 цил.)": "misfire_coil",
            "Потеря компрессии (Цил 4 на ХХ)": "compression_loss",
            "Умирающий бензонасос": "fuel_pump_death"
        }

    if st.button("⚡ Сгенерировать лог"):
        st.session_state.generated_log_df = generate_test_log_df(mapping[test_scenario], st.session_state.diagnostic_mode, st.session_state.is_base_trim)
        st.rerun()

    st.markdown("---")
    if st.button("📋 Показать карту эталонов VAG"):
        st.info("""**Заводская карта допусков CFNA:**
- Давление ДАД (MAP): 280-340 mbar (База: 280-300)
- Нагрузка двигателя: 15-25% (База: 15-18%)
- Время впрыска: 2.0-3.0 мс
- Краткосрочная (Аддитив): ±3%
- Долговременная (Мультипликатив): ±5%
- Дроссель G187 + G188: строго 100%
- Педаль G79 : G185: строго 2:1""")

    if st.button("🗑️ Полная очистка"):
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt(st.session_state.diagnostic_mode, st.session_state.is_base_trim, st.session_state.ecu_type)}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Привет! Я твой виртуальный диагност VAG. 🚗"}]}
        ]
        st.session_state.vin_code = ""
        st.session_state.generated_log_df = None
        clear_history_on_disk()
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
st.subheader("📁 Загрузка данных для анализа")
uploaded_file = st.file_uploader("Загрузи лог VCDS (.csv, .txt) ИЛИ Скриншот (.png, .jpg)", type=["csv", "txt", "png", "jpg", "jpeg"])

log_df = None
image_base64 = None

if uploaded_file is not None:
    if "text" in uploaded_file.type or "csv" in uploaded_file.type:
        file_bytes = uploaded_file.read()
        log_df, extracted_vin = parse_vcds_csv(file_bytes)
        if extracted_vin and extracted_vin != st.session_state.vin_code:
            st.session_state.vin_code = extracted_vin
            save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
    elif "image" in uploaded_file.type:
        image_base64 = encode_image_to_base64(uploaded_file)
        st.image(uploaded_file, caption="Превью скриншота", width=400)

if uploaded_file is None and st.session_state.generated_log_df is not None:
    log_df = st.session_state.generated_log_df

# --- ВЫВОД АВТОМАТИЧЕСКОГО ЭКСПРЕСС-ВЕРДИКТА И ГРАФИКОВ ---
if log_df is not None and not log_df.empty:
    st.success("📊 Данные лога успешно распознаны.")
    time_col = log_df.columns[0]
    
    # --- УЛУЧШЕНИЕ: ЛОКАЛЬНЫЙ АВТОМАТИЧЕСКИЙ ВЕРДИКТ ---
    st.markdown("### 🔍 Предварительный автоматический скрининг лога:")
    alerts = []
    if "Давление ДАД (mbar)" in log_df.columns:
        max_map = log_df["Давление ДАД (mbar)"].max()
        mean_map = log_df["Давление ДАД (mbar)"].mean()
        if mean_map > 360:
            alerts.append("❌ **Критическая аномалия ДАД:** Среднее давление во впуске значительно выше нормы (>360 мбар). Возможен сильный подсос воздуха или проскок цепи ГРМ.")
    if "Краткосрочная коррекция (%)" in log_df.columns:
        max_stft = log_df["Краткосрочная коррекция (%)"].max()
        if max_stft > 12.0:
            alerts.append("⚠️ ** Warning по смеси:** Краткосрочная топливная коррекция уходит в сильный плюс. ЭБУ фиксирует нехватку топлива.")
    if "Пропуски Цилиндр 2" in log_df.columns:
        if log_df["Пропуски Цилиндр 2"].max() > 5:
            alerts.append("⚡ ** Обнаружены пропуски:** Зафиксирован лавинообразный рост пропусков зажигания во 2-м цилиндре.")
            
    if alerts:
        for alert in alerts:
            st.markdown(alert)
    else:
        st.markdown("✅ Локальные экспресс-тесты пройдены. Грубых аномалий в статике не найдено.")
    
    # --- ОТРИСОВКА ГРАФИКОВ ---
    with st.expander("📊 Посмотреть интерактивный график параметров", expanded=True):
        if st.session_state.diagnostic_mode == "Электрика и CAN (Группы 125-135)":
            selected_cols = [c for c in log_df.columns if "АКПП" in c or "АБС" in c or "Приборка" in c or "SRS" in c]
            if selected_cols:
                fig = px.line(log_df, x=time_col, y=selected_cols, title="Статус связи по CAN-шине (1 = ОК, 0 = Обрыв)", template="plotly_dark")
                fig.update_yaxes(range=[-0.2, 1.2], tickvals=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            selected_cols = [c for c in log_df.columns if "Давление" in c or "коррекция" in c or "Пропуски" in c or "Откат" in c or "Нагрузка" in c]
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Scatter(x=log_df[time_col], y=log_df[col], mode='lines', name=col))
                    
                    # Раздельные линии допусков под конкретные параметры
                    if "Давление" in col:
                        low, high = (280.0 if st.session_state.is_base_trim else 305.0), 340.0
                        fig.add_hline(y=low, line_dash="dot", line_color="green", annotation_text="Мин ДАД")
                        fig.add_hline(y=high, line_dash="dot", line_color="red", annotation_text="Макс ДАД")
                    elif "Нагрузка" in col:
                        low, high = (15.0 if st.session_state.is_base_trim else 18.5), 25.0
                        fig.add_hline(y=low, line_dash="dot", line_color="green", annotation_text="Мин Нагрузка")
                        fig.add_hline(y=high, line_dash="dot", line_color="red", annotation_text="Макс Нагрузка")
                    elif "Краткосрочная" in col:
                        fig.add_hline(y=-3.0, line_dash="dot", line_color="blue", annotation_text="-3% Аддитив")
                        fig.add_hline(y=3.0, line_dash="dot", line_color="orange", annotation_text="+3% Аддитив")
                    elif "Долговременная" in col:
                        fig.add_hline(y=-5.0, line_dash="dot", line_color="cyan", annotation_text="-5% Мультипликатив")
                        fig.add_hline(y=5.0, line_dash="dot", line_color="magenta", annotation_text="+5% Мультипликатив")
                
                fig.update_layout(template="plotly_dark", title="Динамика параметров", xaxis_title="Время (сек)", yaxis_title="Значение")
                st.plotly_chart(fig, use_container_width=True)
            
        st.dataframe(log_df.head(10))
            
    button_label = "🚀 Отправить лог VCDS на анализ" if uploaded_file is not None else "🧪 Отправить сгенерированный лог на анализ"
    if st.button(button_label):
        if not API_KEY:
            st.error("Ошибка: API-ключ не найден!")
        else:
            summary_str = generate_log_summary(log_df, st.session_state.vin_code, st.session_state.diagnostic_mode)
            raw_samples = f"\n\nПервые 3 строки:\n{log_df.head(3).to_csv(index=False)}\nПоследние 3 строки:\n{log_df.tail(3).to_csv(index=False)}"
            log_text_payload = f"[{st.session_state.diagnostic_mode}] [ЭБУ: {st.session_state.ecu_type}] " + summary_str + raw_samples
            
            new_message = {"role": "user", "content": [{"type": "text", "text": f"📎 Загружен лог измерений VCDS.\n{summary_str}"}]}
            st.session_state.chat_history.append(new_message)
            
            temp_history = st.session_state.chat_history.copy()
            temp_history[-1] = {"role": "user", "content": [{"type": "text", "text": log_text_payload}]}
            
            with st.chat_message("assistant"):
                with st.spinner("Gemini анализирует..."):
                    response = ask_ai_chat(API_KEY, MODEL_NAME, temp_history, max_tokens=3000)
                    st.write(response)
                    
            st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
            st.rerun()

# Ввод текста пользователя
if user_input := st.chat_input("Напишите симптомы или задайте вопрос..."):
    if not API_KEY:
        st.error("Ошибка: API-ключ не найден!")
    else:
        with st.chat_message("user"):
            st.write(user_input)
            
        mods = st.session_state.get("mods", {"tuned": False, "decatted": False, "lpg": False})
        mods_str = ", ".join([k for k, v in mods.items() if v]) if any(mods.values()) else "Сток"
        vin_str = st.session_state.vin_code if st.session_state.vin_code else "Не указан"
        base_trim_str = "Да" if st.session_state.is_base_trim else "Нет"
        
        ai_text_payload = f"[Контекст: {st.session_state.diagnostic_mode}. ЭБУ: {st.session_state.ecu_type}. VIN: {vin_str}. Моды: {mods_str}. Базовая компл.: {base_trim_str}] {user_input}"
            
        new_message = {"role": "user", "content": [{"type": "text", "text": user_input}]}
        st.session_state.chat_history.append(new_message)
        
        temp_history = st.session_state.chat_history.copy()
        temp_history[-1] = {"role": "user", "content": [{"type": "text", "text": ai_text_payload}]}
        
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response = ask_ai_chat(API_KEY, MODEL_NAME, temp_history, max_tokens=500)
                st.write(response)
                
        st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
        st.rerun()
