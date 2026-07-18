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
    """Специализированный парсер логов VCDS (.CSV с разделителями \t или ,)"""
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

# --- СТАТИСТИЧЕСКИЙ СЖАТЫЙ АНАЛИЗ ДЛЯ ИИ ---

def generate_log_summary(df, vin=""):
    """Создает компактный статистический отчет по логу для отправки в ИИ"""
    summary = []
    summary.append(f"Данные CSV-лога (Статистический срез). VIN: {vin if vin else 'Не указан'}")
    time_col = df.columns[0]
    summary.append(f"Всего точек измерения: {len(df)}. Длительность: {df[time_col].max() - df[time_col].min():.1f} сек.")
    
    for col in df.columns[1:]:
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

# --- УЛУЧШЕНИЕ: ДИНАМИЧЕСКИЙ ЛИМИТ MAX_TOKENS ---
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

# --- РАСШИРЕННАЯ ГЕНЕРАЦИЯ РЕАЛИСТИЧНЫХ ТЕСТОВЫХ ЛОГОВ ---

def generate_test_log_df(scenario="normal", diagnostic_mode="Механика (Группы 001-041)"):
    """Генерирует реалистичные сценарии"""
    np.random.seed(42)
    time = np.arange(0, 100, 1.0)
    n_points = len(time)
    
    if diagnostic_mode == "Электрика и CAN (Группы 125-135)":
        # Генерируем лог для электрики
        if scenario == "can_loss_abs":
            # Локальный обрыв ABS
            df = pd.DataFrame({
                "Отметка Времени (сек)": time,
                "АКПП (Группа 125-1)": np.ones(n_points),
                "АБС (Группа 125-2)": np.zeros(n_points), # АБС отвалился
                "Приборка (Группа 125-3)": np.ones(n_points),
                "SRS (Группа 125-4)": np.ones(n_points),
                "Климат (Группа 126-1)": np.ones(n_points),
                "Запрос Вентилятора % (Группа 135-1)": np.zeros(n_points)
            })
        elif scenario == "immo_conflict":
             df = pd.DataFrame({
                "Отметка Времени (сек)": time,
                "АКПП (Группа 125-1)": np.ones(n_points),
                "АБС (Группа 125-2)": np.ones(n_points), 
                "Приборка (Группа 125-3)": np.random.choice([0, 1], p=[0.8, 0.2], size=n_points), # Приборка скачет
                "SRS (Группа 125-4)": np.ones(n_points),
                "Климат (Группа 126-1)": np.ones(n_points),
                "Запрос Вентилятора % (Группа 135-1)": np.zeros(n_points)
            })
        else: # normal
            df = pd.DataFrame({
                "Отметка Времени (сек)": time,
                "АКПП (Группа 125-1)": np.ones(n_points),
                "АБС (Группа 125-2)": np.ones(n_points), 
                "Приборка (Группа 125-3)": np.ones(n_points),
                "SRS (Группа 125-4)": np.ones(n_points),
                "Климат (Группа 126-1)": np.ones(n_points),
                "Запрос Вентилятора % (Группа 135-1)": np.zeros(n_points)
            })
        return df

    # Базовая генерация механики
    if scenario == "detonation":
        rpm = np.linspace(2000, 5600, n_points) + np.random.normal(0, 15, n_points)
        map_vals = np.linspace(800, 980, n_points) + np.random.normal(0, 5, n_points)
        injector = np.linspace(6.0, 11.5, n_points) + np.random.normal(0, 0.1, n_points)
        stft = np.random.normal(0, 1.5, n_points)
        ltft = np.zeros(n_points) + 1.5
    else:
        rpm = 840 + np.random.normal(0, 8, n_points)
        if scenario == "leak":
            map_vals = 385.0 + np.random.normal(0, 6, n_points)
            injector = 3.10 + np.random.normal(0, 0.04, n_points)
            stft = 15.2 + np.random.normal(0, 1.2, n_points)
            ltft = np.zeros(n_points) + 6.5
        elif scenario == "rich":
            map_vals = 300.0 + np.random.normal(0, 5, n_points)
            injector = 1.85 + np.random.normal(0, 0.03, n_points)
            stft = -18.5 + np.random.normal(0, 1.0, n_points)
            ltft = np.zeros(n_points) - 9.0
        else: # normal & misfire
            map_vals = 305.0 + np.random.normal(0, 5, n_points)
            injector = 2.25 + np.random.normal(0, 0.04, n_points)
            stft = np.random.normal(0, 1.0, n_points)
            ltft = np.zeros(n_points) + 0.8

    df = pd.DataFrame({
        "Отметка Времени (сек)": time,
        "Обороты двигателя (об/мин)": np.round(rpm, 0),
        "Давление ДАД (mbar)": np.round(map_vals, 1),
        "Время впрыска (мс)": np.round(injector, 2),
        "Краткосрочная коррекция (%)": np.round(stft, 2),
        "Долговременная коррекция (%)": np.round(ltft, 2)
    })
    
    for cyl in range(1, 5):
        if scenario == "detonation" and cyl in [3, 4]:
            df[f"Откат УОЗ Цилиндр {cyl} (°KW)"] = np.round(np.linspace(0, 5.8, n_points) + np.random.normal(0, 0.2, n_points), 1)
        else:
            df[f"Откат УОЗ Цилиндр {cyl} (°KW)"] = 0.0

        if scenario == "misfire" and cyl == 2:
            df[f"Пропуски Цилиндр {cyl} (кол-во)"] = np.clip(np.cumsum(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1], size=n_points)), 0, 45)
        else:
            df[f"Пропуски Цилиндр {cyl} (кол-во)"] = 0

    return df

# --- ДИНАМИЧЕСКИЙ СИСТЕМНЫЙ ПРОМПТ В ЗАВИСИМОСТИ ОТ РЕЖИМА ---

def get_system_prompt(mode="Механика (Группы 001-041)"):
    base_prompt = """Ты — профессиональный автодиагност концерна VAG (уровень дилерского центра), специализирующийся на работе с логами VCDS (Вася Диагност).
Твоя главная специализация — двигатель 1.6 MPI CFNA (блок управления Magneti Marelli 7GV).
Твоя задача — анализировать загруженные пользователем логи (часто передаются в виде статистического среза) и выдавать точные технические диагнозы, опираясь СТРОГО на приведенную ниже базу эталонных параметров.
"""
    
    mechanics_rules = """--- БАЗА ЭТАЛОНОВ ДВИГАТЕЛЯ CFNA 1.6 MPI (МЕХАНИКА) ---
ГРУППА 001:
- 001-1 (Обороты): 650–750 об/мин. Если >800 — прогрев, подсос воздуха, или включен кондиционер.
- 001-2 (Темп. ОЖ): 85–98 °C. (Термостат открывается на 87°C. Если <80°C — недогрев).
- 001-3 (Лямбда-рег): Около 0%, быстрые колебания от -10% до +10%.

ГРУППА 002 (Воздух и Нагрузка):
- 002-2 (Нагрузка): 15.0–25.0%.
- 002-3 (Время впрыска): 2.0–3.0 мс. Если >3.2 мс — забиты форсунки или низкое давление топлива.
- 002-4 (ДАД / MAP): 280–340 мбар. КРИТИЧЕСКИЙ МАРКЕР! Если >360 мбар — подсос воздуха, проскок цепи ГРМ.

ГРУППА 003 (Дроссель и Зажигание):
- 003-3 (Угол заслонки): 1.0–3.5%. Если >4.0% — грязная, требует чистки.
- 003-4 (УОЗ): от 4.0° до 12.0° до ВМТ (постоянно скачет).

ГРУППЫ 015 и 016 (Пропуски зажигания):
- Пропуски Цилиндров 1,2,3,4: Строго 0.

ГРУППА 032 (Топливные коррекции - Долговременные):
- 032-1 (Аддитивная, ХХ): ±3% (Лимит ±10%). Если >+5% — подсос воздуха. Если <-5% — льет форсунка.
- 032-2 (Мультипликативная, Нагрузка): ±5% (Лимит ±10%).

ГРУППА 033 и 041 (Зонды):
- 033-1 (Мгновенная коррекция): Быстро меняется. Если "висит" в +25% — жесткий подсос или мертвый бензонасос.
- 033-2 (Напряжение зонд 1): 1.40–2.10 В.
- 041-1 (Напряжение зонд 2, за катом): 0.10–0.85 В.

--- ЛОГИКА КРОСС-ВАЛИДАЦИИ ДЛЯ ИИ ---
1. ДИАГНОСТИКА ПОДСОСА ВОЗДУХА: ЕСЛИ MAP (ДАД) > 360 мбар И Угол дросселя < 1.5% И Аддитивная коррекция > +4%, ТОГДА: Высокая вероятность неучтенного воздуха.
2. ДИАГНОСТИКА РАСТЯЖЕНИЯ ЦЕПИ ГРМ: ЕСЛИ MAP (ДАД) > 365 мбар И Нагрузка > 26% И Угол дросселя > 4.0% И Топливные коррекции в норме (±2%), ТОГДА: Смещение фаз газораспределения.
3. ПРОБЛЕМА С ТОПЛИВОМ / ФОРСУНКАМИ: ЕСЛИ Время впрыска > 3.3 мс И Мгновенная лямбда стремится к +15...+25%, ТОГДА: Низкое давление в рампе либо загрязнение форсунок.
4. ЗАГРЯЗНЕНИЕ ДРОССЕЛЯ: ЕСЛИ Угол дросселя > 4.5% на ХХ, при этом MAP и коррекции в пределах нормы, ТОГДА: Заслонка обросла нагаром."""

    can_rules = """--- БАЗА ЭТАЛОНОВ CAN-ШИНЫ CFNA 1.6 MPI (Группы 125-135) ---
Нормальное значение связи для всех блоков: 1 (Связь ОК). 0 = Нет связи.

ГРУППА 125:
- 125-1 (АКПП): 1. Если 0 (при наличии АКПП) — ЭБУ не видит коробку, машина не заведется.
- 125-2 (АБС/ESP): 1. КРИТИЧНО! Если 0 — загорятся ошибки ABS/ESP. ЭБУ не получает данные о скорости колес.
- 125-3 (Приборная панель): 1. Если 0 или скачет 0/1 — иммобилайзер заблокирует пуск. Тахометр упадет в ноль. Мотор заводится и глохнет.
- 125-4 (SRS / Подушки): 1. Если 0 — нет связи с блоком Airbag.

ГРУППА 126:
- 126-1 (Климат / AC): 1. Если 0 — кондиционер не включится.
- 126-2 (BCM / Блок бортовой сети): 1. Если 0 — проблемы с концевиками, реле стартера.
- 126-3 (Усилитель руля / ЭРУ): 1. Если 0 — руль станет "дубовым".

ГРУППА 127:
- 127-1 (Gateway / Диагностический интерфейс): 1. Если 0 — связь через VCDS обрывается.

ГРУППЫ 130-135 (Управление терморегулированием):
- 130-1 (Темп. ОЖ на выходе из радиатора): Норма 80-95°C.
- 135-1 (Запрос на включение вентилятора %): 0% - выкл. 10-50% - 1 скорость. >50% - 2 скорость. Если висит 100%, а вентилятор молчит — сгорел блок управления вентилятором (БУВ).

--- ЛОГИКА КРОСС-ВАЛИДАЦИИ ДЛЯ ИИ (CAN-ШИНА) ---
1. ТОТАЛЬНЫЙ СБОЙ CAN-ШИНЫ: ЕСЛИ в 125, 126, 127 все параметры = 0, ТОГДА: Обрыв шины CAN-Drive (оранжево-черный/оранжево-коричневый провода) или полное обесточивание ЭБУ (реле R4/K30).
2. ПРОБЛЕМА ЛОКАЛЬНОГО БЛОКА: ЕСЛИ 125-2 (ABS) = 0, а остальные (Приборка, BCM) = 1, ТОГДА: Локальный обрыв проводки к блоку ABS или сгорел предохранитель ABS.
3. КОНФЛИКТ ИММОБИЛАЙЗЕРА: ЕСЛИ 125-3 (Instruments) = 0 ИЛИ скачет 1/0, ТОГДА: Нарушена связь ЭБУ и приборки. Проверить разъем щитка приборов (болезнь Polo)."""

    common_rules = """
ОБЩИЕ ПРАВИЛА ОТВЕТА:
- Твои ответы должны быть профессиональными, без воды.
- Если данные подаются в виде статистического среза (среднее, мин, макс), опирайся на средние значения для вывода, а мин/макс используй для оценки стабильности (например, стабилен ли холостой ход).
- Всегда учитывай модификации (ГБО, чип-тюнинг), если пользователь их указал."""

    if mode == "Электрика и CAN (Группы 125-135)":
        return base_prompt + can_rules + common_rules
    else:
        return base_prompt + mechanics_rules + common_rules

saved_history, saved_vin = load_history_from_disk()

# Управление состоянием режима
if "diagnostic_mode" not in st.session_state:
    st.session_state.diagnostic_mode = "Механика (Группы 001-041)"

if "chat_history" not in st.session_state:
    if saved_history:
        st.session_state.chat_history = saved_history
    else:
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt(st.session_state.diagnostic_mode)}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"Привет! Я твой виртуальный диагност VAG VCDS. 🚗\nТекущий режим: **{st.session_state.diagnostic_mode}**.\nОпиши симптомы, загрузи CSV-лог или скинь скриншот."}]}
        ]

if "vin_code" not in st.session_state:
    st.session_state.vin_code = saved_vin if saved_vin else ""

if "reference_map" not in st.session_state:
    st.session_state.reference_map = {
        "Давление ДАД (mbar)": (280.0, 340.0, "green", "red"),
        "Время впрыска (мс)": (2.0, 3.0, "green", "red"),
        "Краткосрочная коррекция (%)": (-10.0, 10.0, "blue", "orange"),
        "Долговременная коррекция (%)": (-10.0, 10.0, "blue", "orange"),
        "Обороты двигателя (об/мин)": (650.0, 750.0, "green", "red")
    }

if "generated_log_df" not in st.session_state:
    st.session_state.generated_log_df = None

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    
    # ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМА ДИАГНОСТИКИ
    st.markdown("---")
    st.subheader("🧠 Режим ИИ-Диагноста")
    new_mode = st.radio(
        "Выберите профиль диагностики:",
        ("Механика (Группы 001-041)", "Электрика и CAN (Группы 125-135)")
    )
    
    if new_mode != st.session_state.diagnostic_mode:
        st.session_state.diagnostic_mode = new_mode
        # Обновляем системный промпт при смене режима
        if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[0]["role"] == "system":
            st.session_state.chat_history[0]["content"][0]["text"] = get_system_prompt(new_mode)
            st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": f"🔄 Режим ИИ переключен на: **{new_mode}**. База эталонов обновлена."}]})
        st.rerun()

    st.markdown("---")
    st.subheader("🚗 Идентификация автомобиля")

    vin_input = st.text_input("Ввести VIN-код:", value=st.session_state.vin_code, max_chars=17)
    if vin_input != st.session_state.vin_code:
        st.session_state.vin_code = vin_input.upper()

    detected_engine = "1.6 CFNA (Атмо)"
    
    if st.session_state.vin_code:
        if st.session_state.vin_code.startswith("XW8ZZZ61Z"):
            detected_engine = "1.6 CFNA (Атмо)"
            st.sidebar.success("🤖 Определен: Polo Sedan 1.6 CFNA")
            st.session_state.reference_map = {
                "Давление ДАД (mbar)": (280.0, 340.0, "green", "red"),
                "Время впрыска (мс)": (2.0, 3.0, "green", "red"),
                "Краткосрочная коррекция (%)": (-10.0, 10.0, "blue", "orange"),
                "Долговременная коррекция (%)": (-10.0, 10.0, "blue", "orange"),
                "Обороты двигателя (об/мин)": (650.0, 750.0, "green", "red")
            }
        elif any(x in st.session_state.vin_code for x in ["WVWZZZ", "XW8ZZZ1K"]):
            detected_engine = "1.4 TSI (Турбо)"
            st.sidebar.warning("🤖 Определен: VAG 1.4 TSI (Турбо)")
            st.session_state.reference_map = {
                "Давление ДАД (mbar)": (340.0, 390.0, "green", "red"),
                "Время впрыска (мс)": (1.4, 2.0, "green", "red"),
                "Краткосрочная коррекция (%)": (-5.0, 5.0, "blue", "orange"),
                "Долговременная коррекция (%)": (-7.0, 7.0, "blue", "orange"),
                "Обороты двигателя (об/мин)": (700.0, 900.0, "green", "red")
            }
    else:
        st.sidebar.info("ℹ️ VIN не указан. Применяются базовые нормы CFNA.")

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
            "Потеря связи с ABS": "can_loss_abs",
            "Отвал приборки (Иммо)": "immo_conflict"
        }
    else:
        test_scenario = st.selectbox(
            "Сценарий (Механика):",
            ["Исправный мотор", "Подсос воздуха", "Пропуски (Цилиндр 2)", "Детонация (Откаты УОЗ)", "Богатая смесь (-20%)"]
        )
        mapping = {
            "Исправный мотор": "normal",
            "Подсос воздуха": "leak",
            "Пропуски (Цилиндр 2)": "misfire",
            "Детонация (Откаты УОЗ)": "detonation",
            "Богатая смесь (-20%)": "rich"
        }

    if st.button("⚡ Сгенерировать лог"):
        st.session_state.generated_log_df = generate_test_log_df(mapping[test_scenario], st.session_state.diagnostic_mode)
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ Очистить историю"):
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt(st.session_state.diagnostic_mode)}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"Привет! Я твой виртуальный диагност VAG. Текущий режим: {st.session_state.diagnostic_mode}"}]}
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
            st.sidebar.info(f"📍 В файле найден VIN: {extracted_vin}")
            save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
    elif "image" in uploaded_file.type:
        image_base64 = encode_image_to_base64(uploaded_file)
        st.image(uploaded_file, caption="Превью скриншота", width=400)

if uploaded_file is None and st.session_state.generated_log_df is not None:
    log_df = st.session_state.generated_log_df

# --- ВЫВОД ГРАФИКА С ЭТАЛОННЫМИ ЛИНИЯМИ ---
if log_df is not None and not log_df.empty:
    st.success("📊 Данные лога успешно распознаны.")
    
    time_col = log_df.columns[0]
    
    with st.expander("📊 Посмотреть график параметров", expanded=True):
        if st.session_state.diagnostic_mode == "Электрика и CAN (Группы 125-135)":
            # Графики для электрики: статус 0 или 1
            selected_cols = [c for c in log_df.columns if "АКПП" in c or "АБС" in c or "Приборка" in c or "SRS" in c]
            if selected_cols:
                fig = px.line(log_df, x=time_col, y=selected_cols, title="Статус связи по CAN-шине (1 = ОК, 0 = Обрыв)", template="plotly_dark")
                fig.update_yaxes(range=[-0.2, 1.2], tickvals=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Графики для механики
            selected_cols = [c for c in log_df.columns if "Давление" in c or "коррекция" in c or "впрыска" in c or "Откат" in c or "Пропуски" in c]
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Scatter(x=log_df[time_col], y=log_df[col], mode='lines', name=col))
                    
                    for ref_key, (low, high, c_low, c_high) in st.session_state.reference_map.items():
                        if ref_key.lower() in col.lower():
                            fig.add_hline(y=low, line_dash="dot", line_color=c_low, annotation_text=f"Мин {ref_key}")
                            fig.add_hline(y=high, line_dash="dot", line_color=c_high, annotation_text=f"Макс {ref_key}")
                
                fig.update_layout(template="plotly_dark", title="Динамика параметров", xaxis_title="Время (сек)", yaxis_title="Значение")
                st.plotly_chart(fig, use_container_width=True)
            
        st.dataframe(log_df.head(10))
            
    button_label = "🚀 Отправить лог VCDS на анализ" if uploaded_file is not None else "🧪 Отправить сгенерированный лог на анализ"
    if st.button(button_label):
        if not API_KEY:
            st.error("Ошибка: API-ключ не найден!")
        else:
            summary_str = generate_log_summary(log_df, st.session_state.vin_code)
            raw_samples = f"\n\nПервые 3 строки:\n{log_df.head(3).to_csv(index=False)}\nПоследние 3 строки:\n{log_df.tail(3).to_csv(index=False)}"
            log_text_payload = f"[{st.session_state.diagnostic_mode}] " + summary_str + raw_samples
            
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

# Кнопка для скриншота
if image_base64 is not None:
    st.success("🖼️ Скриншот готов к отправке.")
    if st.button("👁️ Отправить Скриншот на анализ"):
        if not API_KEY:
            st.error("Ошибка: API-ключ не найден!")
        else:
            prompt_text = "Пользователь загрузил скриншот окна программы VCDS. Распознай коды ошибок или группы измерений и выдай структурированный диагностический вердикт."
            if st.session_state.vin_code:
                prompt_text = f"VIN автомобиля: {st.session_state.vin_code}. " + prompt_text
                
            new_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]
            }
            st.session_state.chat_history.append(new_message)
            
            with st.chat_message("assistant"):
                with st.spinner("Gemini анализирует скриншот VCDS..."):
                    response = ask_ai_chat(API_KEY, MODEL_NAME, st.session_state.chat_history, max_tokens=1500)
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
        
        ai_text_payload = f"[Контекст: {st.session_state.diagnostic_mode}. VIN: {vin_str}. Моды: {mods_str}] {user_input}"
            
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
