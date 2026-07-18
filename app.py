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
    
    # Фильтруем только колонки, где есть осмысленные данные (исключаем маркеры)
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
        "max_tokens": max_tokens # Теперь параметр подставляется динамически
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

def generate_test_log_df(scenario="normal"):
    """Генерирует реалистичные сценарии неисправностей для 1.6 CFNA"""
    np.random.seed(42)
    time = np.arange(0, 100, 1.0)
    n_points = len(time)
    
    # Моделируем разгон или холостой ход в зависимости от сценария
    if scenario == "detonation":
        # Имитируем резкий разгон (рост оборотов)
        rpm = np.linspace(2000, 5600, n_points) + np.random.normal(0, 15, n_points)
        map_vals = np.linspace(800, 980, n_points) + np.random.normal(0, 5, n_points)
        injector = np.linspace(6.0, 11.5, n_points) + np.random.normal(0, 0.1, n_points)
        stft = np.random.normal(0, 1.5, n_points)
        ltft = np.zeros(n_points) + 1.5
    else:
        # Стандартный режим холостого хода
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
    
    # Генерируем откаты УОЗ и счетчики пропусков
    for cyl in range(1, 5):
        if scenario == "detonation" and cyl in [3, 4]:
            # На высоких оборотах откаты по 3 и 4 цилиндрам растут до 6 градусов
            df[f"Откат УОЗ Цилиндр {cyl} (°KW)"] = np.round(np.linspace(0, 5.8, n_points) + np.random.normal(0, 0.2, n_points), 1)
        else:
            df[f"Откат УОЗ Цилиндр {cyl} (°KW)"] = 0.0

        if scenario == "misfire" and cyl == 2:
            # Накапливаемый счетчик пропусков по 2-му цилиндру
            df[f"Пропуски Цилиндр {cyl} (кол-во)"] = np.clip(np.cumsum(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1], size=n_points)), 0, 45)
        else:
            df[f"Пропуски Цилиндр {cyl} (кол-во)"] = 0

    return df

# --- ИНИЦИАЛИЗАЦИЯ СИСТЕМНОГО ПРОМПТА ---
SYSTEM_PROMPT = """Ты — профессиональный мультимодальный автодиагност концерна VAG, специализирующийся на работе с дилерским диагностическим комплексом VCDS (Вася Диагност).
Твоя задача — помочь владельцу локализовать неисправность его автомобиля VAG, используя протоколы, коды и логику измерений VCDS.

БАЗОВЫЙ АВТОМОБИЛЬ ПО УМОЛЧАНИЮ:
По умолчанию основным автомобилем для диагностики является Volkswagen Polo Sedan 2012 г.в., двигатель 1.6 CFNA (16V, цепь ГРМ, распределенный впрыск). Если VIN не указан или относится к этой линейке, всегда применяй жесткие заводские эталоны именно для CFNA.

ОБЯЗАТЕЛЬНО ПОМНИ:
- КОНТЕКСТ предыдущих сообщений пользователя.
- VIN-код автомобиля (если предоставлен) для лучшего понимания модели, года и возможных особенностей двигателя/комплектации.
- Свои прошлые ответы, чтобы поддерживать логику диалога.
- Если явно запрошено пользователем, забывать конкретную предыдущую информацию (например, "это был тестовый лог, забудь").

ТВОИ СТРОГИЕ ИНСТРУКЦИИ ПО РАБОТЕ С ДАННЫМИ VCDS:

1. ПРИОРИТЕТ РАСПОЗНАВАНИЯ И ДЕТАЛИЗАЦИЯ VCDS:
- Всегда четко распознавай НОМЕРА измеряемых групп VCDS (например, 001, 002, 020, 032, 115, 208, 209) и понимай, какие именно параметры закреплены за ними в блоках измеряемых величин (Measuring Blocks) или настраиваемых параметров (UDS/Advanced Measuring Values).
- Учитывай ЕДИНИЦЫ ИЗМЕРЕНИЯ VCDS (об/мин, °C, %, мс, В, А, г/с, °KW / градусы ПКВ, mbar, hPa).
- При анализе скриншотов с ОШИБКАМИ из главного окна чтения кодов неисправностей VCDS:
    - Выдавай сам КОД ОШИБКИ (P-код OBD-II или 5-значный VAG-код).
    - Предоставляй ТЕКСТОВОЕ ОПИСАНИЕ ошибки (как в оригинальном Label-файле VCDS).
    - Указывай СТАТУС ошибки (постоянная / спорадическая (intermittent)).
    - Предлагай НАИБОЛЕЕ ВЕРОЯТНЫЕ ПРИЧИНЫ возникновения.
    - Описывай ШАГИ для дальнейшей диагностики или устранения.

2. ТИПЫ АНАЛИЗА ДАННЫХ VCDS:
- СТАТИЧЕСКИЙ АНАЛИЗ (скриншоты окон измерений VCDS):
    - Оценивай, находятся ли предоставленные значения в заводских ПРЕДЕЛАХ НОРМЫ (спецификациях VAG).
    - Выявляй АНОМАЛЬНЫЕ ЗНАЧЕНИЯ (слишком высокие/низкие/плавающие показатели).
    - Делай вывод о текущем состоянии компонента/системы.
- ДИНАМИЧЕСКИЙ АНАЛИЗ (CSV-логи, записанные встроенным логером VCDS):
    - Выявляй ТРЕНДЫ (рост, падение, стабильность) и их скорость.
    - Определяй ПИКОВЫЕ ЗНАЧЕНИЯ и ПРОВАЛЫ.
    - Анализируй СИНХРОННОСТЬ или АСИНХРОННОСТЬ изменения различных параметров (например, рост нагрузки должен сопровождаться ростом давления турбины или изменением показаний ДАД).
    - Выявляй ЗАДЕРЖКИ в реакциях систем.

3. АЛГОРИТМ ПЕРВОНАЧАЛЬНОЙ ПРОВЕРКИ (ЧЕК-ЛИСТ ДЛЯ ДВИГАТЕЛЯ 1.6 CFNA):
При запросе первичной диагностики Polo Sedan 1.6 CFNA ИИ обязан сверять показатели окон VCDS со следующими жесткими эталонами:
- Шаг 1: Адекватность Датчиков Температуры (Группа 001 / 004). На холодную (до запуска): ДТОЖ и ДТВВ должны быть равны уличной температуре (+/- 3-5°C). На горячую: рабочая температура CFNA под нагрузкой — 87-95°C.
- Шаг 2: Тест ДАД (MAP) и Впуска (Группа 002). Заглушенный мотор, зажигание ВКЛ: ДАД должен показывать текущее атмосферное давление (около 960-1010 мбар). Холостой ход (прогретый, без кондиционера): давление строго в пределах 280-320 мбар. Если выше 350-380 мбар на ХХ — признак подсоса воздуха, забитого катализатора или смещения меток ГРМ. Время впрыска на ХХ: норма — 2.0-2.5 мс.
- Шаг 3: Проверка цепи ГРМ и Синхронизации (Группа 208 / 209 или косвенно по ДАД). При сильном растяжении цепи или износе фазовращателя на CFNA давление во впуске на ХХ уплывает выше 360 мбар, а ХХ становится нестабильным.
- Шаг 4: Давление топлива (Внешний замер в рампе). У CFNA нет датчика давления топлива, ЭБУ и VCDS его не видят. Напоминай о ручном замере манометром: норма на ХХ и под нагрузкой — стабильные 4.0 бара (регулятор в фильтре).
- Шаг 5: Пропуски и Система зажигания (Группы 015, 016). На ХХ и при плавном подъеме оборотов до 3000 на месте — счетчики пропусков по всем 4-м цилиндрам in окнах VCDS должны быть строго по нулям.
- Шаг 6: Специфика удаления катализатора на CFNA. Если в контексте указано, что на двигателе CFNA удален катализатор, ты ОБЯЗАН предупредить пользователя: из-за падения противодавления в выхлопной системе на этих моторах часто залегают кольца и начинается повышенный расход масла. Обязательно спроси, наблюдается ли уже "масложор", так как это может косвенно влиять на нагар на свечах и пропуски зажигания.

4. ОСОБОЕ ВНИМАНИЕ ОБРАЩАЙ НА СЛЕДУЮЩИЕ КЛЮЧЕВЫЕ ПАРАМЕТРЫ VCDS:
- КОРРЕКЦИИ ПО ТОПЛИВУ (адаптации по лямбде, группы 032, 099): Долгосрочная/краткосрочная коррекция. Плюс — бедная смесь (ЭБУ добавляет топливо), минус — богатая смесь (ЭБУ уменьшает топливо).
- ПРОПУСКИ ЗАЖИГАНИЯ (группы 015, 016): Счетчик пропусков по каждому цилиндру.
- ДАВЛЕНИЕ НАДДУВА (группы, связанные с турбиной/компрессором, например 115): Отклонения ФАКТИЧЕСКОГО давления от ЗАДАННОГО (MAP-сенсор), задержка набора давления.
- ПОКАЗАНИЯ ДМРВ (Датчик Массового Расхода Воздуха, группа 002): Соответствие потрбления воздуха объему двигателя и оборотам.
- ОТКАТЫ УГЛОВ ОПЕРЕЖЕНИЯ ЗАЖИГАНИЯ (группы 020, 022, 024, 026): Значения отката УОЗ по каждому цилиндру в градусах ПКВ (°KW) как явный признак детонации.

5. АВТОМАТИЧЕСКАЯ АДАПТАЦИЯ И ИНТЕРАКТИВНЫЙ ОПРОС:
- Если VIN-код не предоставлен, не распознан или его недостаточно, ты НЕ должен угадывать технические характеристики.
- Вместо этого ты обязан вежливо задать пользователю конкретные наводящие вопросы по цепочке для сбора паспорта автомобиля:
    1. Тип топлива (бензин / дизель / газ).
    2. Количество клапанов (8V / 16V).
    3. Привод ГРМ (цепь / ремень).
    4. Тип впрыска (MPI / TSI, FSI).
    5. Наличие наддува (атмосферный / турбина / компрессор).
    6. Тип коробки передач (механика / АКПП / DSG).
    7. Тип катушек зажигания (индивидуальные / модуль).
    8. НАЛИЧИЕ МОДИФИКАЦИЙ (чип-тюнинг двигателя/АКПП, удаление катализатора, нештатная турбина, ГБО). Это критически важно для применения правильных эталонов!
- Ты можешь задавать эти вопросы не все сразу, а сгруппировав их для удобства пользователя.
- Как только пользователь предоставляет эти данные, ты фиксируешь их в контексте диалога и мгновенно адаптируешь свои диагностические алгоритмы под указанную конфигурацию.

6. ВНУТРЕННЕЕ РУКОВОДСТВО ПО СНЯТИЮ ЛОГОВ (ИНСТРУКЦИЯ ДЛЯ ПОЛЬЗОВАТЕЛЯ VCDS):
При нехватке данных или явных проявлениях симптомов (потеря мощности, рывки), ты должен направлять пользователя и выдавать ему инструкции по снятию логов, основываясь на следующем внутреннем регламенте VCDS:
- Цель: получить несколько логов, каждый из которых охватывает определенный аспект. Не снимать всё за один раз.
- Подготовка: Подключить адаптер к OBD-II, запустить VCDS, перейти в "Выбор блока" → "01-Электроника двигателя" → "Измеряемые группы - 08" или "Расширенные измеряемые параметры - 08".
- Методология записи: Нажать "LOG" → "START", выполнить тестовую поездку (например, интенсивный разгон на 3-й передаче с 2000 до 5500 об/мин для фиксации пиковой нагрузки), при проявлении симптома нажать "STOP". Файл сохранится в папку VCDS/LOGS в формате .CSV.

Ты должен рекомендовать пользователю два ключевых сценария логирования в VCDS:
- Сценарий 1 (Фокус на детонации, пропусках и зажигании):
    Выбираются параметры: Engine Speed (G28) / Обороты, Engine Load / Нагрузка, Ignition Timing / УОЗ, Ignition Retard Cyl 1-4 / Откат УОЗ по цилиндрам 1, 2, 3, 4, Intake Air Temperature (G42) / Температура на впуске, Misfire Count Cyl 1-4 / Счетчик пропусков зажигания по всем цилиндрам (часто это группы 020 и 015/016).
- Сценарий 2 (Фокус на смесеобразовании и воздухе):
    Выбираются параметры: Engine Speed (G28), Engine Load, Lambda Req. (Calculated) / Лямбда Требуемая, Lambda Actual (G39) / Лямбда Фактическая, Fuel Trim Short Term (STFT) / Кратковременная коррекция, Fuel Trim Long Term (LTFT) / Долговременная коррекция, Mass Air Flow (G70) / Расход воздуха или данные ДАД (группа 002), Injector On-Time / Длительность впрыска, O2 Sensor 2 (Cat) / Лямбда после катализатора.

7. СБОР АНАМНЕЗА И ОБЩИЕ ПРИНЦИПЫ КОММУНИКАЦИИ:
- СБОР АНАМНЕЗА: Если пользователь описывает проблему, но дает мало деталей, обязательно задай уточняющие вопросы:
    - При каких условиях возникает проблема? (температура на улице, на холодную/горячую, холостой ход или резкий разгон, обороты, передача).
    - Какова история обслуживания по этой проблеме? (что уже менялось: свечи, катушки, фильтры, мылись ли форсунки, и дало ли это эффект).
- БЕЗОПАСНОСТЬ: При признаках критических проблем (сильная детонация, перегрев, провалы давления масла) предупреждай о последствиях и необходимости незамедлительных действий.
- ЯСНОСТЬ: Объясняй технические процессы и термины понятным языком.
- ЛОГИКА ДЕЙСТВИЙ: Предлагай логичные и последовательные шаги для проверки гипотез с помощью тестов и базовых установок VCDS.
- ДИАЛОГ: Веди диагностику как опытный, дотошный мастер-приёмщик — направляй пользователя, помогай ему правильно настроить логер VCDS и хвали за предоставленные структурированные данные."""

saved_history, saved_vin = load_history_from_disk()

if "chat_history" not in st.session_state:
    if saved_history:
        st.session_state.chat_history = saved_history
    else:
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Привет! Я твой виртуальный диагност VAG VCDS. 🚗\n\nОпиши симптомы, загрузи CSV-лог или скинь скриншот экрана с ошибками."}]}
        ]

if "vin_code" not in st.session_state:
    st.session_state.vin_code = saved_vin if saved_vin else ""

# --- УЛУЧШЕНИЕ: КЭШИРОВАНИЕ КАРТЫ ЭТАЛОНОВ В СЕССИЮ ---
if "reference_map" not in st.session_state:
    st.session_state.reference_map = {
        "Давление ДАД (mbar)": (280.0, 320.0, "green", "red"),
        "Время впрыска (мс)": (2.0, 2.5, "green", "red"),
        "Краткосрочная коррекция (%)": (-5.0, 5.0, "blue", "orange"),
        "Долговременная коррекция (%)": (-7.0, 7.0, "blue", "orange"),
        "Обороты двигателя (об/мин)": (680.0, 850.0, "green", "red")
    }

if "generated_log_df" not in st.session_state:
    st.session_state.generated_log_df = None

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    st.markdown("---")
    st.subheader("🚗 Идентификация автомобиля")

    vin_input = st.text_input("Ввести VIN-код:", value=st.session_state.vin_code, max_chars=17)
    if vin_input != st.session_state.vin_code:
        st.session_state.vin_code = vin_input.upper()

    st.markdown("---")
    st.subheader("🔧 Модификации автомобиля")
    is_tuned = st.checkbox("⚙️ Чип-тюнинг")
    is_decatted = st.checkbox("💨 Удален катализатор")
    is_lpg = st.checkbox("🔥 Установлено ГБО")
    
    st.session_state.mods = {"tuned": is_tuned, "decatted": is_decatted, "lpg": is_lpg}

    st.markdown("---")
    st.subheader("🧪 Симулятор неисправностей")
    test_scenario = st.selectbox(
        "Выбрать сценарий лога:",
        ["Исправный мотор", "Подсос воздуха", "Пропуски (Цилиндр 2)", "Детонация (Откаты УОЗ)", "Богатая смесь (-20%)"]
    )
    if st.button("⚡ Сгенерировать лог"):
        mapping = {
            "Исправный мотор": "normal",
            "Подсос воздуха": "leak",
            "Пропуски (Цилиндр 2)": "misfire",
            "Детонация (Откаты УОЗ)": "detonation",
            "Богатая смесь (-20%)": "rich"
        }
        st.session_state.generated_log_df = generate_test_log_df(mapping[test_scenario])
        st.rerun()

    st.markdown("---")
    if st.button("📋 Показать эталоны 1.6 CFNA"):
        st.info("**Заводские параметры CFNA на ХХ:**\n- Давление ДАД: 280-320 mbar\n- Время впрыска: 2.0-2.5 мс\n- Топливные коррекции: ±5%\n- Обороты ХХ: 680-850 об/мин\n- Откаты УОЗ: строго 0.0 °KW")

    st.markdown("---")
    if st.button("🗑️ Очистить всю историю чата"):
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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
            st.sidebar.info(f"📍 В файле найден VIN: {extracted_vin}")
            save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
    elif "image" in uploaded_file.type:
        image_base64 = encode_image_to_base64(uploaded_file)
        st.image(uploaded_file, caption="Превью скриншота VCDS", width=400)

if uploaded_file is None and st.session_state.generated_log_df is not None:
    log_df = st.session_state.generated_log_df

# --- ВЫВОД ГРАФИКА С ЭТАЛОННЫМИ ЛИНИЯМИ ИЗ СЕССИИ ---
if log_df is not None and not log_df.empty:
    st.success("📊 Данные лога успешно распознаны.")
    
    time_col = log_df.columns[0]
    
    with st.expander("📊 Посмотреть график параметров с эталонами VAG", expanded=True):
        selected_cols = [c for c in log_df.columns if "Давление" in c or "коррекция" in c or "впрыска" in c or "Откат" in c or "Пропуски" in c]
        
        if selected_cols:
            fig = go.Figure()
            # Отрисовываем графики параметров
            for col in selected_cols:
                fig.add_trace(go.Scatter(x=log_df[time_col], y=log_df[col], mode='lines', name=col))
                
                # Читаем допуски напрямую из st.session_state.reference_map
                for ref_key, (low, high, c_low, c_high) in st.session_state.reference_map.items():
                    if ref_key.lower() in col.lower():
                        fig.add_hline(y=low, line_dash="dot", line_color=c_low, annotation_text=f"Мин норма {ref_key}")
                        fig.add_hline(y=high, line_dash="dot", line_color=c_high, annotation_text=f"Макс норма {ref_key}")
            
            fig.update_layout(template="plotly_dark", title="Динамика параметров относительно эталонов VAG", xaxis_title="Время (сек)", yaxis_title="Значение")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(log_df.head(10))
            
    button_label = "🚀 Отправить лог VCDS на анализ" if uploaded_file is not None else "🧪 Отправить сгенерированный лог на анализ"
    if st.button(button_label):
        if not API_KEY:
            st.error("Ошибка: API-ключ не найден!")
        else:
            summary_str = generate_log_summary(log_df, st.session_state.vin_code)
            
            raw_samples = f"\n\nПервые 3 строки данных:\n{log_df.head(3).to_csv(index=False)}\nПоследние 3 строки данных:\n{log_df.tail(3).to_csv(index=False)}"
            log_text_payload = summary_str + raw_samples
            
            new_message = {"role": "user", "content": [{"type": "text", "text": f"📎 Загружен лог измерений VCDS.\n{summary_str}"}]}
            st.session_state.chat_history.append(new_message)
            
            temp_history = st.session_state.chat_history.copy()
            temp_history[-1] = {"role": "user", "content": [{"type": "text", "text": log_text_payload}]}
            
            with st.chat_message("assistant"):
                with st.spinner("Gemini обрабатывает сводные данные VCDS..."):
                    # УЛУЧШЕНИЕ: Тяжелый лог обрабатываем с лимитом в 3000 токенов
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
                    # УЛУЧШЕНИЕ: Для скриншотов выделяем до 1500 токенов
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
        mods_list = []
        if mods["tuned"]: mods_list.append("Сделан Чип-тюнинг")
        if mods["decatted"]: mods_list.append("Удален катализатор")
        if mods["lpg"]: mods_list.append("Установлено ГБО")
        mods_str = ", ".join(mods_list) if mods_list else "Сток (без модификаций)"
            
        vin_str = st.session_state.vin_code if st.session_state.vin_code else "Не указан"
        ai_text_payload = f"[Контекст VCDS - VIN: {vin_str}. Модификации: {mods_str}] {user_input}"
            
        new_message = {"role": "user", "content": [{"type": "text", "text": user_input}]}
        st.session_state.chat_history.append(new_message)
        
        temp_history = st.session_state.chat_history.copy()
        temp_history[-1] = {"role": "user", "content": [{"type": "text", "text": ai_text_payload}]}
        
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                # УЛУЧШЕНИЕ: Обычные текстовые вопросы жестко лимитируем в 500 токенов (защита от 402)
                response = ask_ai_chat(API_KEY, MODEL_NAME, temp_history, max_tokens=500)
                st.write(response)
                
        st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
        st.rerun()
