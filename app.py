import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import io
import re

# Настройка страницы Streamlit
st.set_page_config(page_title="VAG Expert Chat", page_icon="🚗", layout="wide")

# Кэшируем функцию парсинга, чтобы не перечитывать тяжелый файл при перезапуске сессии
@st.cache_data(show_spinner=False)
def safe_parse_log(file_bytes):
    try:
        text_data = file_bytes.decode('cp1251', errors='ignore')
        lines = text_data.splitlines()
        
        # Поиск VIN (17 символов)
        vin_pattern = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b', re.IGNORECASE)
        extracted_vin = None
        
        for line in lines[:20]:
            match = vin_pattern.search(line)
            if match:
                extracted_vin = match.group(1).upper()
                break
        
        # Ищем начало таблицы параметров
        start_idx = 0
        for i, line in enumerate(lines):
            if any(x in line for x in ["Группа", "Group", "Об/мин", "RPM"]):
                start_idx = i
                break
                
        clean_csv_data = "\n".join(lines[start_idx:])
        df = pd.read_csv(io.StringIO(clean_csv_data), sep=None, engine='python')
        
        # Фильтрация колонок
        safe_columns = []
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() / len(df) > 0.8:
                df[col] = converted
                safe_columns.append(col)
                
        df_safe = df[safe_columns].dropna()
        
        if len(df_safe) > 150:
            df_safe = df_safe.iloc[:150]
            
        return df_safe, extracted_vin
    except Exception as e:
        return None, None

# Функция отправки истории диалога в OpenRouter
def ask_ai_chat(api_key, model_name, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://share.streamlit.io", 
    }
    
    data = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=45)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Ошибка API OpenRouter: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Ошибка сети: {e}"

# --- ИНИЦИАЛИЗАЦИЯ И КЭШИРОВАНИЕ СЕССИИ ---

MODEL_NAME = "google/gemini-2.5-flash"
API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

SYSTEM_PROMPT = (
    "Ты — профессиональный автодиагност концерна VAG. Твоя задача — помочь владельцу локализовать проблему с машиной.\n"
    "ОБЯЗАТЕЛЬНО помни контекст предыдущих сообщений пользователя и свои прошлые ответы!"
)

# Сохраняем состояние чата в глобальный кэш Streamlit, привязанный к пользователю
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Привет! Я твой виртуальный ассистент-диагност VAG. 🚗\n\nОпиши, что происходит с машиной или загрузи файл лога в боковой панели слева."}
    ]

if "vin_code" not in st.session_state:
    st.session_state.vin_code = ""

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    vin_input = st.text_input("VIN-код автомобиля", value=st.session_state.vin_code, max_chars=17)
    if vin_input:
        st.session_state.vin_code = vin_input.upper()
        
    st.markdown("---")
    st.subheader("📁 Загрузка логов")
    uploaded_file = st.file_uploader("Перетащи сюда файл .csv / .txt", type=["csv", "txt"])

# --- ОБРАБОТКА ФАЙЛА ---
log_df = None
if uploaded_file is not None:
    # Передаем байты вместо объекта файла, чтобы работал декоратор @st.cache_data
    file_bytes = uploaded_file.read()
    log_df, extracted_vin = safe_parse_log(file_bytes)
    if extracted_vin and extracted_vin != st.session_state.vin_code:
        st.session_state.vin_code = extracted_vin
        st.sidebar.info(f"📍 Найден VIN: {extracted_vin}")

# --- ИНТЕРФЕЙС ЧАТА ---
st.title("VAG Expert Chat 💬")

# Отображаем историю
for msg in st.session_state.chat_history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Лог и кнопка анализа
if log_df is not None and not log_df.empty:
    st.info("📊 Лог-файл успешно загружен в систему.")
    
    rpm_cols = [c for c in log_df.columns if any(x in c.lower() for x in ["обороты", "rpm", "speed"])]
    map_cols = [c for c in log_df.columns if any(x in c.lower() for x in ["давлен", "map", "pressure"])]
    
    if rpm_cols and map_cols:
        with st.expander("Посмотреть график заезда", expanded=True):
            fig = px.line(log_df, x=rpm_cols[0], y=map_cols[0], 
                          title="Давление впуска (MAP) от Оборотов",
                          labels={rpm_cols[0]: "Обороты", map_cols[0]: "Давление"},
                          template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🚀 Отправить лог на анализ ИИ"):
        if not API_KEY:
            st.error("Ошибка: API-ключ не найден в Secrets!")
        else:
            csv_str = log_df.to_csv(index=False)
            log_payload = f"Пользователь загрузил лог-файл. Вот данные:\n{csv_str}"
            if st.session_state.vin_code:
                log_payload = f"VIN: {st.session_state.vin_code}. " + log_payload
                
            st.session_state.chat_history.append({"role": "user", "content": log_payload})
            
            with st.chat_message("assistant"):
                with st.spinner("Анализирую параметры лога..."):
                    response = ask_ai_chat(API_KEY, MODEL_NAME, st.session_state.chat_history)
                    st.write(response)
                    
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

# Ввод сообщения
if user_input := st.chat_input("Напишите симптомы или задайте вопрос..."):
    if not API_KEY:
        st.error("Ошибка: API-ключ не найден in Secrets!")
    else:
        with st.chat_message("user"):
            st.write(user_input)
            
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response = ask_ai_chat(API_KEY, MODEL_NAME, st.session_state.chat_history)
                st.write(response)
                
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
