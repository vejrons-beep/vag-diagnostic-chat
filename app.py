import streamlit as st
import pandas as pd
import plotly.express as px
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
CACHE_FILE = "chat_history_cache.json"  # Файл постоянной памяти на сервере

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С ПОСТОЯННОЙ ПАМЯТЬЮ ---

def save_history_to_disk(history, vin_code):
    try:
        data = {
            "vin_code": vin_code,
            "chat_history": history
        }
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

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДИАГНОСТИКИ ---

@st.cache_data(show_spinner=False)
def safe_parse_log(file_bytes):
    try:
        text_data = file_bytes.decode('cp1251', errors='ignore')
        lines = text_data.splitlines()
        
        # 1. Поиск VIN-кода
        vin_pattern = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b', re.IGNORECASE)
        extracted_vin = None
        for line in lines[:20]:
            match = vin_pattern.search(line)
            if match:
                extracted_vin = match.group(1).upper()
                break
        
        # 2. Умный поиск начала таблицы с данными
        data_start_idx = None
        for i, line in enumerate(lines):
            parts = line.split()
            if parts and re.match(r'^\d+[\.,]?\d*$', parts[0]):
                if len(parts) >= 2:
                    data_start_idx = i
                    break
                    
        if data_start_idx is None:
            return None, extracted_vin
            
        # 3. Собираем заголовки
        headers = []
        for offset in [2, 3, 4]:
            check_idx = data_start_idx - offset
            if check_idx >= 0:
                potential_headers = lines[check_idx].split()
                if len(potential_headers) >= 2 and any(not x.replace('.','').isdigit() for x in potential_headers):
                    headers = potential_headers
                    break
                    
        # Извлекаем только строки с данными
        numeric_lines = []
        for line in lines[data_start_idx:]:
            parts = line.split()
            if parts and re.match(r'^-?\d+[\.,]?\d*$', parts[0].replace('-','')):
                clean_parts = [p.replace(',', '.') for p in parts]
                numeric_lines.append(clean_parts)
                
        if not numeric_lines:
            return None, extracted_vin
            
        # Определяем итоговое количество колонок
        col_count = min(len(row) for row in numeric_lines)
        numeric_lines = [row[:col_count] for row in numeric_lines]
        
        # Создаем понятные названия колонок
        if len(headers) != col_count:
            headers = [f"Колонка_{i}" for i in range(col_count)]
            headers[0] = "TIME STAMP"
            if col_count == 5:
                headers[1] = "Цилиндр 1 (Откат УОЗ)"
                headers[2] = "Цилиндр 2 (Откат УОЗ)"
                headers[3] = "Цилиндр 3 (Откат УОЗ)"
                headers[4] = "Цилиндр 4 (Откат УОЗ)"
            
        df = pd.DataFrame(numeric_lines, columns=headers)
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna()
        if len(df) > 150: 
            df = df.iloc[:150]
            
        return df, extracted_vin
    except Exception as e:
        return None, None

def encode_image_to_base64(uploaded_file):
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        encoded_string = base64.b64encode(file_bytes).decode('utf-8')
        mime_type = uploaded_file.type
        return f"data:{mime_type};base64,{encoded_string}"
    return None

def ask_ai_chat(api_key, model_name, messages):
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
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Ошибка API OpenRouter: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Ошибка сети: {e}"

# --- ИНИЦИАЛИЗАЦИЯ И ПОДГРУЗКА ПАМЯТИ ---

SYSTEM_PROMPT = """Ты — профессиональный мультимодальный автодиагност концерна VAG, специализирующийся на данных программы 'Вася Диагност' / VCDS.
Твоя задача — помочь владельцу локализовать проблему с его автомобилем VAG.

БАЗОВЫЙ АВТОМОБИЛЬ ПО УМОЛЧАНИЮ:
По умолчанию основным автомобилем для диагностики является Volkswagen Polo Sedan 2012 г.в., двигатель 1.6 CFNA (16V, цепь ГРМ, распределенный впрыск). Если VIN не указан или относится к этой линейке, всегда применяй жесткие заводские эталоны именно для CFNA.

ОБЯЗАТЕЛЬНО ПОМНИ:
- КОНТЕКСТ предыдущих сообщений пользователя.
- VIN-код автомобиля (если предоставлен) для лучшего понимания модели, года и возможных особенностей двигателя/комплектации.
- Свои прошлые ответы, чтобы поддерживать логику диалога.
- Если явно запрошено пользователем, забывать конкретную предыдущую информацию (например, "это был тестовый лог, забудь").

ТВОИ СТРОГИЕ ИНСТРУКЦИИ ПО РАБОТЕ С ДАННЫМИ:

1. ПРИОРИТЕТ РАСПОЗНАВАНИЯ И ДЕТАЛИЗАЦИЯ:
- Всегда распознавай НОМЕРА измеряемых групп VCDS (например, 001, 020, 032, 115) и понимай, какие параметры в них находятся.
- Учитывай ЕДИНИЦЫ ИЗМЕРЕНИЯ (об/мин, °C, %, мс, В, А, г/с, °KW, mbar, hPa).
- При анализе скриншотов с ОШИБКАМИ (DTC):
    - Выдавай сам КОД ОШИБКИ (P-код или VAG-код).
    - Предоставляй ТЕКСТОВОЕ ОПИСАНИЕ ошибки.
    - Указывай СТАТУС ошибки (постоянная / спорадическая (intermittent)).
    - Предлагай НАИБОЛЕЕ ВЕРОЯТНЫЕ ПРИЧИНЫ возникновения.
    - Описывай ШАГИ для дальнейшей диагностики или устранения.

2. ТИПЫ АНАЛИЗА ДАННЫХ:
- СТАТИЧЕСКИЙ АНАЛИЗ (скриншоты):
    - Оценивай, находятся ли предоставленные значения в ПРЕДЕЛАХ НОРМЫ для VAG.
    - Выявляй АНОМАЛЬНЫЕ ЗНАЧЕНИЯ (слишком высокие/низкие/плавающие показатели).
    - Делай вывод о текущем состоянии компонента/системы.
- ДИНАМИЧЕСКИЙ АНАЛИЗ (CSV-логи):
    - Выявляй ТРЕНДЫ (рост, падение, стабильность) и их скорость.
    - Определяй ПИКОВЫЕ ЗНАЧЕНИЯ и ПРОВАЛЫ.
    - Анализируй СИНХРОННОСТЬ или АСИНХРОННОСТЬ изменения различных параметров (например, рост нагрузки должен сопровождаться ростом давления турбины).
    - Выявляй ЗАДЕРЖКИ в реакциях систем.

3. АЛГОРИТМ ПЕРВОНАЧАЛЬНОЙ ПРОВЕРКИ (ЧЕК-ЛИСТ ДЛЯ ДВИГАТЕЛЯ 1.6 CFNA):
При запросе первичной диагностики Polo Sedan 1.6 CFNA ИИ обязан сверять статические показатели со следующими жесткими эталонами:
- Шаг 1: Адекватность Датчиков Температуры (Группа 001 / 004). На холодную (до запуска): ДТОЖ и ДТВВ должны быть равны уличной температуре (+/- 3-5°C). На горячую: рабочая температура CFNA под нагрузкой — 87-95°C.
- Шаг 2: Тест ДАД (MAP) и Впуска (Группа 002). Заглушенный мотор, зажигание ВКЛ: ДАД должен показывать текущее атмосферное давление (около 960-1010 мбар). Холостой ход (прогретый, без кондиционера): давление строго в пределах 280-320 мбар. Если выше 350-380 мбар на ХХ — признак подсоса воздуха, забитого катализатора или смещения меток ГРМ. Время впрыска на ХХ: норма — 2.0-2.5 мс.
- Шаг 3: Проверка цепи ГРМ и Синхронизации (Группа 208 / 209 или косвенно по ДАД). При сильном растяжении цепи или износе фазовращателя на CFNA давление во впуске на ХХ уплывает выше 360 мбар, а ХХ становится нестабильным.
- Шаг 4: Давление топлива (Внешний замер в рампе). У CFNA нет датчика давления топлива. Напоминай о ручном замере манометром: норма на ХХ и под нагрузкой — стабильные 4.0 бара (регулятор в фильтре).
- Шаг 5: Пропуски и Система зажигания (Группы 015, 016). На ХХ и при плавном подъеме оборотов до 3000 на месте — счетчики пропусков по всем 4-м цилиндрам должны быть строго по нулям.
- Шаг 6: Специфика удаления катализатора на CFNA. Если в контексте указано, что на двигателе CFNA удален катализатор, ты ОБЯЗАН предупредить пользователя: из-за падения противодавления в выхлопной системе на этих моторах часто залегают кольца и начинается повышенный расход масла. Обязательно спроси, наблюдается ли уже "масложор", так как это может косвенно влиять на нагар на свечах и пропуски зажигания.

4. ОСОБОЕ ВНИМАНИЕ ОБРАЩАЙ НА СЛЕДУЮЩИЕ КЛЮЧЕВЫЕ ПАРАМЕТРЫ:
- КОРРЕКЦИИ ПО ТОПЛИВУ (адаптации по лямбде, группы 032, 099): Долгосрочная/краткосрочная коррекция. Плюс — бедная смесь (ЭБУ добавляет топливо), минус — богатая смесь (ЭБУ уменьшает топливо).
- ПРОПУСКИ ЗАЖИГАНИЯ (группы 015, 016): Счетчик пропусков по каждому цилиндру.
- ДАВЛЕНИЕ НАДДУВА (группы, связанные с турбиной/компрессором): Отклонения ФАКТИЧЕСКОГО давления от ЗАДАННОГО (MAP-сенсор), задержка набора давления.
- ПОКАЗАНИЯ ДМРВ (Датчик Массового Расхода Воздуха, группа 002): Соответствие потребления воздуха объему двигателя и оборотам.
- ОТКАТЫ УГЛОВ ОПЕРЕЖЕНИЯ ЗАЖИГАНИЯ (группы 020, 022, 024, 026): Значения отката УОЗ по каждому цилиндру как явный признак детонации.

5. АВТОМАТИЧЕСКАЯ АДАПТАЦИЯ И ИНТЕРАКТИВНЫЙ ОПРОС:
- Если VIN-код не предоставлен, не распознан или его недостаточно, вежливо задай пользователю вопросы для сбора паспорта автомобиля:
    1. Тип топлива (бензин / дизель / газ).
    2. Количество клапанов (8V / 16V).
    3. Привод ГРМ (цепь / ремень).
    4. Тип впрыска (MPI / TSI, FSI).
    5. Наличие наддува (атмосферный / турбина / компрессор).
    6. Тип коробки передач (механика / АКПП / DSG).
    7. Тип катушек зажигания (индивидуальные / модуль).
    8. НАЛИЧИЕ МОДИФИКАЦИЙ (чип-тюнинг двигателя/АКПП, удаление катализатора, нештатная турбина, ГБО). Это критически важно для применения правильных эталонов!

6. ВНУТРЕННЕЕ РУКОВОДСТВО ПО СНЯТИЮ ЛОГОВ (ИНСТРУКЦИЯ ДЛЯ ПОЛЬЗОВАТЕЛЯ):
При нехватке данных или явных проявлениях симптомов (потеря мощности, рывки), ты должен направлять пользователя и выдавать ему инструкции по снятию логов, основываясь на следующем внутреннем регламенте:
- Цель: получить несколько логов, каждый из которых охватывает определенный аспект. Не снимать всё за один раз.
- Подготовка: Подключить адаптер к OBD-II, запустить "Вася Диагност", перейти в "Выбор блока" → "01-Электроника двигателя" → "Измеряемые группы - 08" или "Расширенные измеряемые параметры - 08".
- Методология записи: Нажать "LOG" → "START", выполнить тестовую поездку (например, интенсивный разгон на 3-й передаче с 2000 до 5500 об/мин для фиксации пиковой нагрузки), при проявлении симптома нажать "STOP". Файл сохранится в папку VCDS/LOGS в формате .CSV.

Ты должен рекомендовать пользователю два ключевых сценария логирования:
- Сценарий 1 (Фокус на детонации, пропусках и зажигании):
    Выбираются параметры: Engine Speed (G28) / Обороты, Engine Load / Нагрузка, Ignition Timing / УОЗ, Ignition Retard Cyl 1-4 / Откат УОЗ по цилиндрам 1, 2, 3, 4, Intake Air Temperature (G42) / Температура на впуске, Misfire Count Cyl 1-4 / Счетчик пропусков зажигания по всем цилиндрам.
- Сценарий 2 (Фокус на смесеобразовании и воздухе):
    Выбираются параметры: Engine Speed (G28), Engine Load, Lambda Req. (Calculated) / Лямбда Требуемая, Lambda Actual (G39) / Лямбда Фактическая, Fuel Trim Short Term (STFT) / Кратковременная коррекция, Fuel Trim Long Term (LTFT) / Долговременная коррекция, Mass Air Flow (G70) / Расход воздуха или данные ДАД, Injector On-Time / Длительность впрыска, O2 Sensor 2 (Cat) / Лямбда после катализатора.

7. СБОР АНАМНЕЗА И ОБЩИЕ ПРИНЦИПЫ КОММУНИКАЦИИ:
- СБОР АНАМНЕЗА: Если пользователь описывает проблему, но дает мало деталей, обязательно задай уточняющие вопросы:
    - При каких условиях возникает проблема? (температура на улице, на холодную/горячую, холостой ход или резкий разгон, обороты, передача).
    - Какова история обслуживания по этой проблеме? (что уже менялось: свечи, катушки, фильтры, мылись ли форсунки, и дало ли это эффект).
- БЕЗОПАСНОСТЬ: При признаках критических проблем предупреждай о последствиях.
- ДИАЛОГ: Веди диагностику как опытный, дотошный мастер-приёмщик."""

saved_history, saved_vin = load_history_from_disk()

if "chat_history" not in st.session_state:
    if saved_history:
        st.session_state.chat_history = saved_history
    else:
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Привет! Я твой виртуальный диагност VAG. 🚗\n\nОпиши симптомы, загрузи CSV-лог или просто скинь СКРИНШОТ экрана 'Васи Диагноста' с ошибками."}]}
        ]

if "vin_code" not in st.session_state:
    st.session_state.vin_code = saved_vin if saved_vin else ""

# Пытаемся импортировать easyocr для распознавания VIN по фото
try:
    import easyocr
    @st.cache_resource
    def get_ocr_reader():
        return easyocr.Reader(['en'], gpu=False)
except ImportError:
    easyocr = None

# Функция для поиска VIN в тексте
def extract_vin(text):
    clean_text = text.upper().replace(" ", "").replace("-", "").replace("_", "")
    clean_text = clean_text.replace("O", "0").replace("I", "1").replace("Q", "9")
    vin_pattern = re.compile(r'([A-HJ-NPR-Z0-9]{17})')
    match = vin_pattern.search(clean_text)
    return match.group(1) if match else None

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    st.markdown("---")
    st.subheader("🚗 Идентификация автомобиля")

    uploaded_vin_img = None
    if easyocr:
        uploaded_vin_img = st.file_uploader(
            "📷 Сканировать VIN по фото (СТТС, кузов)", 
            type=["jpg", "jpeg", "png"],
            key="vin_image_uploader"
        )
        
    if "last_processed_vin_img" not in st.session_state:
        st.session_state.last_processed_vin_img = None

    if uploaded_vin_img is not None:
        if uploaded_vin_img.name != st.session_state.last_processed_vin_img:
            try:
                with st.spinner("Распознаю VIN-код с фотографии..."):
                    image = Image.open(uploaded_vin_img)
                    img_np = np.array(image)
                    reader = get_ocr_reader()
                    result = reader.readtext(img_np, detail=0)
                    
                    full_text = "".join(result)
                    found_vin = extract_vin(full_text)
                    
                    if found_vin:
                        st.session_state.vin_code = found_vin
                        st.session_state.last_processed_vin_img = uploaded_vin_img.name
                        st.rerun()
                    else:
                        st.error("❌ Не удалось четко распознать 17-значный VIN. Попробуйте другое фото или введите вручную.")
            except Exception as e:
                st.error(f"Ошибка сканирования: {e}")
    elif uploaded_vin_img is None and st.session_state.last_processed_vin_img is not None:
        st.session_state.last_processed_vin_img = None

    vin_input = st.text_input(
        "Ввести VIN-код вручную:", 
        value=st.session_state.vin_code,
        max_chars=17
    )

    if vin_input != st.session_state.vin_code:
        st.session_state.vin_code = vin_input.upper()

    # --- НОВЫЙ БЛОК: МОДИФИКАЦИИ ---
    st.markdown("---")
    st.subheader("🔧 Модификации автомобиля")
    is_tuned = st.checkbox("⚙️ Чип-тюнинг (Stage 1/2/3)")
    is_decatted = st.checkbox("💨 Удален катализатор (Декат)")
    
    # Сохраняем в сессию, чтобы передать в чат
    st.session_state.mods = {
        "tuned": is_tuned,
        "decatted": is_decatted
    }
    # -------------------------------
    
    st.markdown("---")
    
    if st.button("🗑️ Очистить всю историю чата"):
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Привет! Я твой виртуальный диагност VAG. 🚗\n\nОпиши симптомы, загрузи CSV-лог или просто скинь СКРИНШОТ экрана 'Васи Диагноста' с ошибками."}]}
        ]
        st.session_state.vin_code = ""
        st.session_state.last_processed_vin_img = None
        clear_history_on_disk()
        st.rerun()

# --- ЗДЕСЬ БОКОВАЯ ПАНЕЛЬ ЗАКОНЧИЛАСЬ. КОД НИЖЕ ВЫВОДИТСЯ ПО ЦЕНТРУ ---

# --- ИНТЕРФЕЙС ЧАТА (ОСНОВНОЙ ЭКРАН) ---
st.title("VAG Expert Chat + Vision 💬")

# 1. ПЕРВЫМ ДЕЛОМ ВЫВОДИМ ИСТОРИЮ ЧАТА (чтобы старые сообщения были вверху, а новые внизу)
for msg in st.session_state.chat_history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            for content_item in msg["content"]:
                if content_item["type"] == "text":
                    st.write(content_item["text"])
                elif content_item["type"] == "image_url":
                    st.image(content_item["image_url"]["url"], width=300)

st.markdown("---")

# 2. ТЕПЕРЬ ВЫВОДИМ ЗОНУ ЗАГРУЗКИ ДАННЫХ ВНИЗУ ЧАТА
st.subheader("📁 Загрузка данных для анализа")
uploaded_file = st.file_uploader("Загрузи лог (.csv, .txt) ИЛИ Скриншот (.png, .jpg)", type=["csv", "txt", "png", "jpg", "jpeg"])

# --- ОБРАБОТКА ФАЙЛА ---
log_df = None
image_base64 = None
file_type = None

if uploaded_file is not None:
    file_type = uploaded_file.type
    if "text" in file_type or "csv" in file_type:
        file_bytes = uploaded_file.read()
        log_df, extracted_vin = safe_parse_log(file_bytes)
        if extracted_vin and extracted_vin != st.session_state.vin_code:
            st.session_state.vin_code = extracted_vin
            st.sidebar.info(f"📍 Найден VIN: {extracted_vin}")
            save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
            
    elif "image" in file_type:
        image_base64 = encode_image_to_base64(uploaded_file)
        st.image(uploaded_file, caption="Превью загруженного скриншота", width=400)

# Кнопка для анализа CSV-лога
if log_df is not None and not log_df.empty:
    st.info("📊 CSV-лог успешно загружен в систему.")
    
    rpm_cols = [c for c in log_df.columns if any(x in c.lower() for x in ["обороты", "rpm", "speed", "об/мин"])]
    map_cols = [c for c in log_df.columns if any(x in c.lower() for x in ["давлен", "map", "pressure", "бар", "bar"])]
    
    with st.expander("📊 Посмотреть график параметров лога", expanded=True):
        if rpm_cols and map_cols:
            fig = px.line(log_df, x=rpm_cols[0], y=map_cols[0], 
                          title=f"Диагностический график: {map_cols[0]} от {rpm_cols[0]}",
                          labels={rpm_cols[0]: rpm_cols[0], map_cols[0]: map_cols[0]},
                          template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            time_col = log_df.columns[0]
            other_cols = [c for c in log_df.columns if c != time_col]
            
            if other_cols:
                fig = px.line(log_df, x=time_col, y=other_cols, 
                              title=f"Изменение параметров по времени ({time_col})",
                              labels={time_col: "Время (сек)", "value": "Значение / Откат углов"},
                              template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Недостаточно числовых колонок для построения графика.")
    
    if st.button("🚀 Отправить загруженный лог на анализ"):
        if not API_KEY:
            st.error("Ошибка: API-ключ не найден!")
        else:
            csv_str = log_df.to_csv(index=False)
            
            is_duplicate_log = False
            for old_msg in st.session_state.chat_history:
                if old_msg["role"] == "user":
                    for item in old_msg["content"]:
                        if item["type"] == "text" and csv_str in item["text"]:
                            is_duplicate_log = True
                            break
            
            if is_duplicate_log:
                st.session_state.chat_history.append({"role": "user", "content": [{"type": "text", "text": "📎 [Повторная попытка отправить тот же лог-файл]"}]})
                st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": "Ты ошибся, ты присылал мне этот лог ранее. Посмотри на анализ выше!"}]})
                save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
                st.rerun()
                
            log_text_payload = f"Пользователь загрузил CSV-лог. Вот данные:\n{csv_str}"
            if st.session_state.vin_code:
                log_text_payload = f"VIN: {st.session_state.vin_code}. " + log_text_payload
                
            new_message = {"role": "user", "content": [{"type": "text", "text": log_text_payload}]}
            st.session_state.chat_history.append(new_message)
            
            with st.chat_message("assistant"):
                with st.spinner("Gemini анализирует CSV-лог..."):
                    response = ask_ai_chat(API_KEY, MODEL_NAME, st.session_state.chat_history)
                    st.write(response)
                    
            st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
            st.rerun()

# Кнопка для анализа Скриншота
if image_base64 is not None:
    st.success("🖼️ Скриншот готов к отправке.")
    if st.button("👁️ Отправить Скриншот на анализ"):
        if not API_KEY:
            st.error("Ошибка: API-ключ не найден!")
        else:
            is_duplicate_img = False
            pure_base64 = image_base64.split(",")[1] if "," in image_base64 else image_base64
            
            for old_msg in st.session_state.chat_history:
                if old_msg["role"] == "user":
                    for item in old_msg["content"]:
                        if item["type"] == "image_url" and pure_base64 in item["image_url"]["url"]:
                            is_duplicate_img = True
                            break
            
            if is_duplicate_img:
                st.session_state.chat_history.append({"role": "user", "content": [{"type": "text", "text": "📎 [Повторная попытка отправить тот же скриншот]"}]})
                st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": "Ты ошибся, ты присылал мне этот скриншот ранее. Посмотри на анализ выше!"}]})
                save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
                st.rerun()
            
            prompt_text = "Пользователь загрузил скриншот. Распознай ошибки, группы или графики на нем и дай диагностический вердикт."
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
                with st.spinner("Gemini 'смотрит' на скриншот..."):
                    response = ask_ai_chat(API_KEY, MODEL_NAME, st.session_state.chat_history)
                    st.write(response)
                    
            st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
            st.rerun()

# 3. В САМОМ НИЗУ ОСТАЕТСЯ ОКНО ВВОДА ТЕКСТА КЛАССИЧЕСКОГО ЧАТА
if user_input := st.chat_input("Напишите симптомы или задайте вопрос..."):
    if not API_KEY:
        st.error("Ошибка: API-ключ не найден!")
    else:
        with st.chat_message("user"):
            st.write(user_input)
            
        # Считываем галочки модификаций
        mods = st.session_state.get("mods", {"tuned": False, "decatted": False})
        mods_list = []
        if mods["tuned"]: mods_list.append("Сделан Чип-тюнинг")
        if mods["decatted"]: mods_list.append("Удален катализатор")
        mods_str = ", ".join(mods_list) if mods_list else "Сток (без модификаций)"
            
        # Формируем скрытый payload для ИИ с VIN и модификациями
        ai_text_payload = user_input
        vin_str = st.session_state.vin_code if st.session_state.vin_code else "Не указан"
        
        ai_text_payload = f"[Контекст системы - VIN: {vin_str}. Тюнинг: {mods_str}] {user_input}"
            
        # В историю чата пишем чистый текст пользователя
        new_message = {"role": "user", "content": [{"type": "text", "text": user_input}]}
        st.session_state.chat_history.append(new_message)
        
        # Подменяем последнее сообщение для отправки в API
        temp_history = st.session_state.chat_history.copy()
        temp_history[-1] = {"role": "user", "content": [{"type": "text", "text": ai_text_payload}]}
        
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response = ask_ai_chat(API_KEY, MODEL_NAME, temp_history)
                st.write(response)
                
        st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
        st.rerun()
