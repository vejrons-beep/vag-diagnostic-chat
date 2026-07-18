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

# Настройка страницы Streamlit
st.set_page_config(page_title="VAG Expert Chat + Vision", page_icon="🚗", layout="wide")

# --- КОНСТАНТЫ И НАСТРОЙКИ ---
MODEL_NAME = "google/gemini-2.5-flash"
API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
CACHE_FILE = "chat_history_cache.json"  # Файл постоянной памяти на сервере

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С ПОСТОЯННОЙ ПАМЯТЬЮ ---

def save_history_to_disk(history, vin_code):
    """Сохраняет историю чата и VIN в файл на сервере"""
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
    """Загружает историю чата и VIN из файла на сервере"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("chat_history", []), data.get("vin_code", "")
        except Exception as e:
            return [], ""
    return [], ""

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДИАГНОСТИКИ ---

@st.cache_data(show_spinner=False)
def safe_parse_log(file_bytes):
    try:
        text_data = file_bytes.decode('cp1251', errors='ignore')
        lines = text_data.splitlines()
        
        # Поиск VIN
        vin_pattern = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b', re.IGNORECASE)
        extracted_vin = None
        for line in lines[:20]:
            match = vin_pattern.search(line)
            if match:
                extracted_vin = match.group(1).upper()
                break
        
        # Ищем начало таблицы
        start_idx = 0
        for i, line in enumerate(lines):
            if any(x in line for x in ["Группа", "Group", "Об/мин", "RPM"]):
                start_idx = i
                break
                
        clean_csv_data = "\n".join(lines[start_idx:])
        df = pd.read_csv(io.StringIO(clean_csv_data), sep=None, engine='python')
        
        # Фильтрация числовых колонок
        safe_columns = []
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() / len(df) > 0.8:
                df[col] = converted
                safe_columns.append(col)
                
        df_safe = df[safe_columns].dropna()
        if len(df_safe) > 150: df_safe = df_safe.iloc[:150]
            
        return df_safe, extracted_vin
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
    
    # Чтобы не отправлять гигантские картинки в Base64 повторно при каждом запросе истории
    # и не забивать контекст, очищаем старые картинки в отправляемой истории
    cleaned_messages = []
    for m in messages:
        new_content = []
        for item in m["content"]:
            if item["type"] == "text":
                new_content.append(item)
            elif item["type"] == "image_url":
                # Заменяем старую тяжелую картинку текстом, чтобы OpenRouter не выдавал ошибку размера
                new_content.append({"type": "text", "text": "[Ранее отправленный скриншот экрана диагностики]"})
        cleaned_messages.append({"role": m["role"], "content": new_content})

    # На всякий случай возвращаем последнюю картинку активной, если это был самый свежий запрос
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

SYSTEM_PROMPT = (
    "Ты — профессиональный мультимодальный автодиагност концерна VAG. Твоя задача — помочь владельцу локализовать проблему.\n"
    "ОБЯЗАТЕЛЬНО помни контекст предыдущих сообщений пользователя и свои прошлые ответы!"
)

# Пытаемся восстановить историю с жесткого диска сервера
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

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("⚙️ Панель управления")
    vin_input = st.text_input("VIN-код автомобиля", value=st.session_state.vin_code, max_chars=17)
    if vin_input and vin_input.upper() != st.session_state.vin_code:
        st.session_state.vin_code = vin_input.upper()
        save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
        
    # Кнопка полной очистки памяти (если захочешь начать новый диалог с нуля)
    if st.button("🗑️ Очистить всю историю чата"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.session_state.clear()
        st.rerun()
        
    st.markdown("---")
    st.subheader("📁 Загрузка данных")
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
        st.sidebar.image(uploaded_file, caption="Превью скриншота", use_column_width=True)

# --- ИНТЕРФЕЙС ЧАТА ---
st.title("VAG Expert Chat + Vision 💬")

# Отображаем историю
for msg in st.session_state.chat_history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            for content_item in msg["content"]:
                if content_item["type"] == "text":
                    st.write(content_item["text"])
                elif content_item["type"] == "image_url":
                    st.image(content_item["image_url"]["url"], width=300)

# Кнопка для анализа CSV-лога
if log_df is not None and not log_df.empty:
    st.info("📊 CSV-лог успешно загружен.")
    
    rpm_cols = [c for c in log_df.columns if any(x in c.lower() for x in ["обороты", "rpm", "speed"])]
    map_cols = [c for c in log_df.columns if any(x in c.lower() for x in ["давлен", "map", "pressure"])]
    
    if rpm_cols and map_cols:
        with st.expander("Посмотреть график заезда", expanded=True):
            fig = px.line(log_df, x=rpm_cols[0], y=map_cols[0], 
                          title="Давление впуска (MAP) от Оборотов",
                          labels={rpm_cols[0]: "Обороты", map_cols[0]: "Давление"},
                          template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🚀 Отправить CSV-лог на анализ"):
        if not API_KEY:
            st.error("Ошибка: API-ключ не найден!")
        else:
            csv_str = log_df.to_csv(index=False)
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

# Ввод обычного текстового сообщения
if user_input := st.chat_input("Напишите симптомы или задайте вопрос..."):
    if not API_KEY:
        st.error("Ошибка: API-ключ не найден!")
    else:
        with st.chat_message("user"):
            st.write(user_input)
            
        new_message = {"role": "user", "content": [{"type": "text", "text": user_input}]}
        st.session_state.chat_history.append(new_message)
        
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response = ask_ai_chat(API_KEY, MODEL_NAME, st.session_state.chat_history)
                st.write(response)
                
        st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        save_history_to_disk(st.session_state.chat_history, st.session_state.vin_code)
        st.rerun()
