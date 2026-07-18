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

st.set_page_config(page_title="VAG Expert Chat + Vision", page_icon="🚗", layout="wide")

MODEL_NAME = "google/gemini-2.5-flash"
API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
CACHE_FILE = "chat_history_cache.json"

# --- ПРОВЕРКА ПИН-КОДА ---
def check_password():
    """Возвращает True, если пароль верен или уже были авторизованы."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒")
    st.write("Введите пин-код для продолжения")

    with st.form("auth_form"):
        password = st.text_input("Пин-код", type="password")
        submit = st.form_submit_button("Войти")

        if submit:
            correct_password = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", "1234"))
            if password == correct_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Неверный пин-код")
    return False

if not check_password():
    st.stop()

# --- ПОСТОЯННАЯ ПАМЯТЬ ---
def save_history_to_disk(history, vin_code):
    try:
        data = {"vin_code": vin_code, "chat_history": history}
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except:
        pass

def load_history_from_disk():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("chat_history", []), data.get("vin_code", "")
        except:
            return [], ""
    return [], ""

def clear_history_on_disk():
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except:
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

# --- ГЕНЕРАТОР ТЕСТОВЫХ ЛОГОВ ---
def generate_test_log_df(scenario="normal", diagnostic_mode="Механика (Группы 001-063)", is_base_trim=False):
    np.random.seed(42)
    time = np.arange(0, 100, 1.0)
    n_points = len(time)

    if diagnostic_mode.startswith("Электрика"):
        missing_val = -1 if is_base_trim else 1
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

    # Механика
    rpm = 840 + np.random.normal(0, 8, n_points)
    map_base = 290.0 if is_base_trim else 305.0
    map_vals = map_base + np.random.normal(0, 5, n_points)
    injector = 2.25 + np.random.normal(0, 0.04, n_points)
    stft = np.random.normal(0, 1.0, n_points)
    ltft = np.zeros(n_points) + 0.8
    misfire_c1 = np.zeros(n_points)
    misfire_c2 = np.zeros(n_points)
    misfire_c3 = np.zeros(n_points)
    misfire_c4 = np.zeros(n_points)
    g79 = np.ones(n_points) * 14.5
    g187 = np.ones(n_points) * 4.5

    if scenario == "detonation":
        rpm = np.linspace(2000, 5600, n_points)
        map_vals = np.linspace(800, 980, n_points)
        injector = np.linspace(6.0, 11.5, n_points)
        stft = np.random.normal(0, 1.5, n_points)
        ltft = np.ones(n_points) + 1.5
        g79 = np.linspace(14.5, 90.0, n_points)
        g187 = np.linspace(4.5, 88.0, n_points)
    elif scenario == "leak":
        map_vals = 385.0 + np.random.normal(0, 6, n_points)
        injector = 3.10 + np.random.normal(0, 0.04, n_points)
        stft = 15.2 + np.random.normal(0, 1.2, n_points)
        ltft = np.ones(n_points) + 6.5
    elif scenario == "rich":
        map_vals = 300.0 + np.random.normal(0, 5, n_points)
        injector = 1.85 + np.random.normal(0, 0.03, n_points)
        stft = -18.5 + np.random.normal(0, 1.0, n_points)
        ltft = np.ones(n_points) - 9.0
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
    elif scenario == "misfire_coil":
        misfire_c2 = np.clip(np.cumsum(np.random.choice([0,1,2], p=[0.7,0.2,0.1], size=n_points)), 0, 45)
        stft = np.linspace(0, 12.0, n_points)
    elif scenario == "compression_loss":
        rpm = np.concatenate([np.ones(50)*840, np.linspace(840, 2500, 50)])
        map_vals = np.concatenate([np.ones(50)*375.0, np.linspace(375.0, 500.0, 50)])
        misfire_c4 = np.concatenate([np.cumsum(np.random.choice([0,1], p=[0.5,0.5], size=50)), np.ones(50)*25])

    df = pd.DataFrame({
        "Отметка Времени (сек)": time,
        "Обороты двигателя (об/мин)": np.round(rpm, 0),
        "Давление ДАД (mbar)": np.round(map_vals, 1),
        "Время впрыска (мс)": np.round(injector, 2),
        "Краткосрочная коррекция (%)": np.round(stft, 2),
        "Долговременная коррекция (%)": np.round(ltft, 2),
        "Дроссель 1 (G187) %": np.round(g187, 1),
        "Педаль 1 (G79) %": np.round(g79, 1),
        "Пропуски Цилиндр 1": np.round(misfire_c1, 0),
        "Пропуски Цилиндр 2": np.round(misfire_c2, 0),
        "Пропуски Цилиндр 3": np.round(misfire_c3, 0),
        "Пропуски Цилиндр 4": np.round(misfire_c4, 0),
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

    config_note = ""
    if is_base_trim:
        config_note = """
[!] ВАЖНО: АВТОМОБИЛЬ В БАЗОВОЙ КОМПЛЕКТАЦИИ (МКПП, БЕЗ КОНДИЦИОНЕРА, БЕЗ ABS).
В группах 125 и 126 значения -1 для ABS, Климата и АКПП являются АБСОЛЮТНОЙ НОРМОЙ (блок физически отсутствует). Значение 0 для этих блоков означало бы обрыв связи, а -1 — заводское отсутствие.
Нагрузка на ХХ должна быть строго 15-18%, а MAP строго 280-300 мбар (паразитных нагрузок нет).
ЖЕСТКОЕ ПРАВИЛО: Если параметр is_base_trim=True (базовая комплектация), то любые значения MAP выше 315 мбар и Нагрузки выше 20% на холостом ходу (при оборотах ~700-800) ТРАКТУЙ КАК АНОМАЛИЮ (подсос воздуха или загрязнение дроссельной заслонки), игнорируя общие верхние допуски (340 мбар и 25%).
"""

    mods_note = "\n--- УЧЕТ МОДИФИКАЦИЙ АВТОМОБИЛЯ ---"
    if mods.get("decatted", False):
        mods_note += """
- УДАЛЕН КАТАЛИЗАТОР (Евро-2 / Стейдж 1): Вторая лямбда (Группа 041-1) может выдавать прямую линию, повторять первую лямбду или висеть в ошибке. Игнорируй любые аномалии и ошибки по катализатору (P0420) и второму зонду. Не предлагай замену катализатора.
"""
    else:
        mods_note += """
- СТОКОВЫЙ ВЫПУСК: Вторая лямбда (Группа 041-1) должна быть стабильна (0.6 - 0.7 В). Если она скачет вслед за первой — катализатор разрушен или забит.
"""

    if mods.get("lpg", False):
        mods_note += """
- УСТАНОВЛЕНО ГБО (Газ): На газе долговременная коррекция (Группа 032-1 и 032-2) в пределах ±8.0% является ДОПУСТИМОЙ нормой из-за разницы в теплотворной способности пропан-бутана и бензина. Не ставь диагноз 'подсос' или 'умирающий бензонасос', если отклонение коррекций в этих пределах. Обрати внимание на возможные пропуски зажигания из-за газовых форсунок.
"""

    if mods.get("tuned", False):
        mods_note += """
- ЧИП-ТЮНИНГ (Агрессивная прошивка): Угол опережения зажигания (УОЗ, Группа 003-4) может быть более ранним, а время впрыска (002-3) под нагрузкой — выше стандартных 3.0 мс. Оценивай стабильность, а не абсолютные заводские значения УОЗ.
"""

    mechanics_rules = """
--- БАЗА ЭТАЛОНОВ CFNA 1.6 MPI ---
ГРУППА 002 (Воздух и Нагрузка):
- 002-2 (Нагрузка): 15.0–25.0%.
- 002-4 (ДАД / MAP): 280–340 мбар. Если >360 мбар — подсос воздуха, проскок цепи ГРМ, потеря компрессии.

ГРУППЫ 032, 033 (Топливные коррекции):
- 032-1 (Аддитивная коррекция, LTFT на ХХ): ±3.0%. > +4.0% = подсос. < -4.0% = перелив форсунок.
- 032-2 (Мультипликативная коррекция, LTFT под нагрузкой): ±5.0%. > +6.0% = нехватка топлива (насос/фильтр).
- 033-1 (Мгновенная лямбда-коррекция, STFT онлайн): ±10.0%. Зависание в +25% или -25% указывает на выход регулирования за пределы физического лимита.

ГРУППЫ 014, 015, 016 (ЗАЖИГАНИЕ И ПРОПУСКИ):
- 014-3 (Суммарные пропуски): 0.
- 015-1, 015-2, 015-3, 016-1 (Счетчики Цил 1-4): Строго 0.

ГРУППЫ 062, 063 (Дроссель/Педаль):
- 062-1 + 062-2 = 100% (зеркальность).
- 062-3 = 062-4 * 2 (педаль 2:1).

--- ЛОГИКА КРОСС-ВАЛИДАЦИИ ---
1. ПРОБОЙ КАТУШКИ/СВЕЧИ: ЕСЛИ Пропуски (015/016) быстро растут ТОЛЬКО в одном цилиндре И Мгновенная лямбда (033-1) уходит в плюс, ТОГДА: Локальный пропуск. Рекомендация: переставить катушку на другой цилиндр.
2. ПРОПУСКИ ИЗ-ЗА БЕДНОЙ СМЕСИ: ЕСЛИ Пропуски хаотичны по всем цилиндрам И Коррекции (032-1/2) > +8.0%, ТОГДА: Системное обеднение. Катушки целы, проблема в бензонасосе или подсосе.
3. ПОТЕРЯ КОМПРЕССИИ: ЕСЛИ постоянные пропуски в ОДНОМ цилиндре ТОЛЬКО на холостом ходу (на оборотах >2000 счетчик стоит) И MAP > 360 мбар, ТОГДА: Механическая потеря компрессии (клапан/кольца). Замер компрессометром.
4. СКРЫТЫЙ ПОДСОС: ЕСЛИ Аддитив > +5.0% И Мультипликатив в норме И MAP > 350 мбар, ТОГДА: Подсос за дросселем (клапан ВКГ).
5. БЕНЗОНАСОС: ЕСЛИ Аддитив в норме И Мультипликатив > +7.0% И Мгновенная уходит в +25% под нагрузкой, ТОГДА: Дефицит топлива (замер давления, норма 4.0 бар).
6. РАССИНХРОН ДРОССЕЛЯ/ПЕДАЛИ: Нарушение пропорций 100% или 2:1 ведет к EPC. Замена узла или адаптация (060).
"""

    can_rules = """--- БАЗА ЭТАЛОНОВ CAN-ШИНЫ (Группы 125-135) ---
Норма связи: 1 = блок отвечает, 0 = нет связи.
- 125-1 (АКПП), 125-2 (АБС/ESP), 126-1 (Климат). В базовой комплектации эти блоки отсутствуют, их значение = -1 (физически нет).
- 125-3 (Приборка): Если 0 или скачет 0/1 — иммо блокирует пуск, авто глохнет.
- 135-1 (Запрос вентилятора): 100% при молчащем кулере = сгорел БУВ.
ЛОГИКА:
1. ТОТАЛЬНЫЙ СБОЙ: ЕСЛИ все параметры = 0, ТОГДА Обрыв шины CAN-Drive или обесточивание ЭБУ.
2. ЛОКАЛЬНЫЙ ОБРЫВ: ЕСЛИ только один блок = 0, ТОГДА обрыв провода к блоку или предохранитель.
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
            {"role": "assistant", "content": [{"type": "text", "text": "Привет! Я твой виртуальный диагност VAG VCDS. 🚗\n\nОпиши симптомы, загрузи CSV-лог или скинь скриншот экрана с ошибками."}]}
        ]
if "vin_code" not in st.session_state:
    st.session_state.vin_code = saved_vin if saved_vin else ""
if "reference_map" not in st.session_state:
    st.session_state.reference_map = {
        "Давление ДАД (mbar)": (280.0, 340.0, "green", "red"),
        "Время впрыска (мс)": (2.0, 3.0, "green", "red"),
        "Краткосрочная коррекция (%)": (-10.0, 10.0, "blue", "orange"),
        "Долговременная коррекция (%)": (-10.0, 10.0, "blue", "orange"),
        "Пропуски": (0.0, 0.0, "green", "red")
    }
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
            st.session_state.is_base_trim
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

    # Кнопка выхода (не очищает историю)
    if st.button("🚪 Выйти"):
        # Сбрасываем только авторизацию, история остаётся
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
                             or "Пропуски" in c or "Откат" in c or "G187" in c or "G79" in c]
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
