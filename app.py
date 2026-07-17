import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import io
import re

# Настройка страницы Streamlit
st.set_page_config(page_title="VAG Expert Chat", page_icon="🚗", layout="wide")

# Функция безопасного парсинга и извлечения VIN-кода
def safe_parse_log(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        text_data = file_bytes.decode('cp1251', errors='ignore')
        lines = text_data.splitlines()
        
        # Регулярка для поиска VIN (17 символов)
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
        st.error(f"Ошибка безопасности при анализе структуры файла: {e}")
        return None, None

# Функция отправки истории диалога в OpenRouter
def ask_ai_chat(api_key, model_name, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://huggingface.co/spaces", 
    }
    
    data = {
        "model": model_name,
        "messages": messages
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=45)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Ошибка API OpenRouter: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Ошибка сети: {e}"

# --- ИНИЦИАЛИЗАЦИЯ СЕССИИ ---

# Системный промпт, задающий роль ИИ
SYSTEM_PROMPT = (
    "Ты — профессиональный автодиагност концерна VAG. Твоя задача — помочь владельцу локализовать проблему с машиной в формате чата.\n\n"
    "ПРАВИЛА ОТВЕТОВ:\n"
    "1. Общайся вежливо, профессионально, но простым языком.\n"
    "2. Если пользователь указывает VIN-код, расшифруй его (модель, год, мотор) и используй эти спецификации.\n"
    "3. Если лог еще не загружен, расспроси о симптомах, предложи вероятные версии и назови точные номера групп (например, 003, 020, 031) для записи в программе 'Вася Диагност'.\n"
    "4. Если лог загружен, детально сопоставь цифры (обороты, давление MAP, откаты УОЗ, лямбда) с симптомами, найди аномалии и дай четкий пошаговый план ремонта."
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Привет! Я твой виртуальный ассистент-диагност VAG. 🚗\n\nОпиши, что происходит с машиной (симптомы, когда проявляется тупняк), или загрузи файл лога в боковой панели слева. Если знаешь VIN-код, тоже напиши его — это поможет мне точнее определить параметры твоего мотора."}
    ]

if "vin_code" not in st.session_state:
    st.session_state.vin_code = ""

# --- БОКОВАЯ ПАНЕЛЬ (НАСТРОЙКИ И ФАЙЛЫ) ---

with st.sidebar:
    st.header("⚙️ Панель управления")
    
    saved_key = st.secrets.get("OPENROUTER_API_KEY", "")
    api_key = st.text_input("OpenRouter API Key", value=saved_key, type="password")
    
    model_option = st.selectbox(
        "Модель ИИ",
        options=["deepseek/deepseek-r1:free", "deepseek/deepseek-chat", "google/gemini-2.5-flash"],
        index=0
    )
    
    vin_input = st.text_input("VIN-код автомобиля", value=st.session_state.vin_code, max_chars=17)
    if vin_input:
        st.session_state.vin_code = vin_input.upper()
        
    st.markdown("---")
    st.subheader("📁 Загрузка логов")
    uploaded_file = st.file_uploader("Перетащи сюда файл .csv / .txt", type=["csv", "txt"])

# --- ОБРАБОТКА ЗАГРУЖЕННОГО ФАЙЛА ---

log_df = None
if uploaded_file is not None:
    # Парсим лог
    log_df, extracted_vin = safe_parse_log(uploaded_file)
    if extracted_vin and extracted_vin != st.session_state.vin_code:
        st.session_state.vin_code = extracted_vin
        st.sidebar.info(f"📍 Найден VIN в файле: {extracted_vin}")

# --- ОСНОВНОЙ ИНТЕРФЕЙС ЧАТА ---

st.title("VAG Expert Chat 💬")

# Отображение истории чата (кроме скрытого системного промпта)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Если лог загружен, выводим график и кнопку прямо в ленту чата
if log_df is not None and not log_df.empty:
    st.info("📊 Лог-файл успешно загружен в систему.")
    
    # Строим график прямо внутри чата
    rpm_cols = [c for c in log_df.columns if any(x in c.lower() for x in ["обороты", "rpm", "speed"])]
    map_cols = [c for c in log_df.columns if any(x in c.lower() for x in ["давлен", "map", "pressure"])]
    
    if rpm_cols and map_cols:
        with st.expander("Посмотреть график заезда", expanded=True):
            fig = px.line(log_df, x=rpm_cols[0], y=map_cols[0], 
                          title="Давление впуска (MAP) от Оборотов",
                          labels={rpm_cols[0]: "Обороты", map_cols[0]: "Давление"},
                          template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
    # Кнопка для отправки лога на анализ
    if st.button("🚀 Отправить лог на анализ ИИ"):
        if not api_key:
            st.error("Добавь API-ключ в левой панели!")
        else:
            csv_str = log_df.to_csv(index=False)
            
            # Добавляем контекст в историю для ИИ
            system_msg = f"Пользователь загрузил лог-файл. "
            if st.session_state.vin_code:
                system_msg += f"VIN автомобиля: {st.session_state.vin_code}. "
            system_msg += f"Вот данные лога для анализа:\n{csv_str}"
            
            st.session_state.messages.append({"role": "user", "content": system_msg})
            st.session_state.chat_history.append({"role": "user", "content": "📎 [Файл лога отправлен на анализ]"})
            
            # Показываем анимацию загрузки
            with st.chat_message("assistant"):
                with st.spinner("Анализирую параметры лога..."):
                    response = ask_ai_chat(api_key, model_option, st.session_state.messages)
                    st.write(response)
                    
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

# --- ВВОД НОВОГО СООБЩЕНИЯ В ЧАТ ---

if user_input := st.chat_input("Напишите симптомы или задайте вопрос..."):
    if not api_key:
        st.error("Пожалуйста, сначала введите API-ключ в левой панели!")
    else:
        # Отображаем сообщение пользователя в чате
        with st.chat_message("user"):
            st.write(user_input)
            
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Подготавливаем сообщение для ИИ (включая VIN, если он есть)
        ai_payload = user_input
        if st.session_state.vin_code and len(st.session_state.messages) == 1:
            ai_payload = f"Мой VIN: {st.session_state.vin_code}. " + ai_payload
            
        st.session_state.messages.append({"role": "user", "content": ai_payload})
        
        # Запрос к модели
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response = ask_ai_chat(api_key, model_option, st.session_state.messages)
                st.write(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
