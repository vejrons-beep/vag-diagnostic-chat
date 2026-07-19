"""
app.py — VAG Expert Chat + Vision
Исправленная версия: защита от падений на Streamlit Cloud
"""

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

# Опциональные импорты — с защитой от отсутствия
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# Импорт наших модулей
try:
    from config import (
        MODEL_NAME, API_KEY, 
        get_system_prompt, build_reference_map, 
        TEST_SCENARIOS, VCDS_GROUPS, CFNA_FAULTS
    )
    from vcds_engine import (
        parse_vcds_csv, generate_test_log_df, 
        generate_log_summary, encode_image_to_base64
    )
    CUSTOM_MODULES_OK = True
except Exception as e:
    CUSTOM_MODULES_OK = False
    CUSTOM_MODULES_ERROR = str(e)

try:
    from audio_engine_diagnosis import analyze_engine_audio, get_image_base64, cleanup_all_temp_files
    AUDIO_MODULE_OK = True
except Exception as e:
    AUDIO_MODULE_OK = False
    AUDIO_MODULE_ERROR = str(e)

st.set_page_config(page_title="VAG Expert Chat + Vision", page_icon="🚗", layout="wide")


# ==================== API КЛЮЧ ====================
# Пробуем st.secrets, потом os.environ, потом пустую строку
API_KEY = ""
try:
    API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
except Exception:
    pass
if not API_KEY:
    API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MODEL_NAME = "google/gemini-2.5-flash"


# ==================== АУТЕНТИФИКАЦИЯ (с защитой) ====================

def check_password():
    """Проверка пин-кода с защитой от ошибок Streamlit Cloud."""
    try:
        query_params = st.query_params
        if "auth" in query_params:
            st.session_state.authenticated = True
            try:
                st.query_params.clear()
            except Exception:
                pass
            return True
    except Exception:
        pass

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
            correct_password = "1234"
            try:
                correct_password = st.secrets.get("APP_PASSWORD", "1234")
            except Exception:
                correct_password = os.environ.get("APP_PASSWORD", "1234")
            if password == correct_password:
                st.session_state.authenticated = True
                try:
                    st.query_params["auth"] = "1"
                except Exception:
                    pass
                st.rerun()
            else:
                st.error("Неверный пин-код")
    return False

if not check_password():
    st.stop()


# ==================== GOOGLE SHEETS (с fallback) ====================

def _get_gsheet():
    if not GSPREAD_AVAILABLE:
        return None
    try:
        b64_str = st.secrets.get("GSPREAD_SERVICE_ACCOUNT_BASE64", "")
        if not b64_str:
            return None
        creds_json = base64.b64decode(b64_str).decode()
        creds_dict = json.loads(creds_json)
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(
            "https://docs.google.com/spreadsheets/d/1ALFMvWYfjJsT_OeLWQRDXIzuvoA-8Xvn9CB5tm8qH1w/edit#gid=0"
        )
        return sheet.sheet1
    except Exception:
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
        trimmed = history[-50:] if len(history) > 50 else history
        sheet.update("B1", [[json.dumps(trimmed, ensure_ascii=False)]])
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


# ==================== API ИИ ====================

def ask_ai_chat(api_key, model_name, messages, max_tokens=2500):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://share.streamlit.io",
    }

    cleaned_messages = []
    for i, m in enumerate(messages):
        new_content = []
        for item in m["content"]:
            if item["type"] == "text":
                new_content.append(item)
            elif item["type"] == "image_url":
                if i == len(messages) - 1:
                    new_content.append(item)
                else:
                    new_content.append({"type": "text", "text": "[Ранее отправленный скриншот/спектрограмма]"})
        cleaned_messages.append({"role": m["role"], "content": new_content})

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


# ==================== ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЙ ====================

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
        if saved_history and saved_history[0].get("role") == "system":
            st.session_state.chat_history = saved_history
        else:
            st.session_state.chat_history = [
                {"role": "system", "content": [{"type": "text", "text": get_system_prompt(
                    st.session_state.diagnostic_mode, st.session_state.is_base_trim,
                    mods=st.session_state.mods
                )}]}
            ] + saved_history
    else:
        st.session_state.chat_history = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt(
                st.session_state.diagnostic_mode, st.session_state.is_base_trim,
                mods=st.session_state.mods
            )}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Привет! Я твой виртуальный диагност VAG VCDS. 🚗\n\nЗагрузи CSV-лог (снятый на бензине), скриншот VCDS, или опиши симптомы."}]}
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


# ==================== БОКОВАЯ ПАНЕЛЬ ====================

with st.sidebar:
    st.header("⚙️ Конфигурация автомобиля")
    st.write("Настрой параметры для точной работы ИИ.")
    st.markdown("---")

    st.subheader("📦 Комплектация")
    prev_base = st.session_state.is_base_trim
    st.session_state.is_base_trim = st.checkbox(
        "Базовая комплектация (CFNA BASE)",
        value=st.session_state.is_base_trim,
        help="МКПП, без кондиционера, без ABS. ИИ сузит допуски MAP и проигнорирует отсутствие блоков по CAN."
    )

    st.markdown("---")
    st.subheader("🛠️ Модификации")

    decatted = st.checkbox(
        "Катализатор удален (Евро-2)",
        value=st.session_state.mods.get("decatted", False),
        help="ИИ проигнорирует вторую лямбду (Группа 041) и не будет советовать замену ката."
    )
    lpg = st.checkbox(
        "Установлено ГБО (Газ)",
        value=st.session_state.mods.get("lpg", False),
        help="ИИ даст советы по ГБО в чате и аудио. Логи всё равно анализируются как бензин."
    )
    tuned = st.checkbox(
        "Чип-тюнинг (Stage 1/Custom)",
        value=st.session_state.mods.get("tuned", False),
        help="ИИ сделает скидку на ранние УОЗ и повышенное время впрыска."
    )

    prev_mods = dict(st.session_state.mods)
    st.session_state.mods = {"tuned": tuned, "decatted": decatted, "lpg": lpg}

    settings_changed = (prev_base != st.session_state.is_base_trim or 
                        prev_mods != st.session_state.mods)

    if settings_changed and len(st.session_state.chat_history) > 0:
        st.session_state.reference_map = build_reference_map(
            st.session_state.is_base_trim, st.session_state.mods
        )
        new_system = get_system_prompt(
            st.session_state.diagnostic_mode,
            st.session_state.is_base_trim,
            mods=st.session_state.mods
        )
        st.session_state.chat_history[0] = {
            "role": "system",
            "content": [{"type": "text", "text": new_system}]
        }
        save_chat_history(st.session_state.chat_history)
        st.toast("⚙️ Настройки обновлены")

    save_profile(st.session_state.vin_code, st.session_state.is_base_trim, st.session_state.mods)

    st.markdown("---")
    st.subheader("🔍 Режим диагностики")
    prev_mode = st.session_state.diagnostic_mode
    st.session_state.diagnostic_mode = st.radio(
        "Выберите контур:",
        ["Механика (Группы 001-063)", "Электрика и CAN (Группы 125-135)"],
        index=0 if st.session_state.diagnostic_mode.startswith("Механика") else 1
    )

    if prev_mode != st.session_state.diagnostic_mode and len(st.session_state.chat_history) > 0:
        new_system = get_system_prompt(
            st.session_state.diagnostic_mode,
            st.session_state.is_base_trim,
            mods=st.session_state.mods
        )
        st.session_state.chat_history[0] = {
            "role": "system",
            "content": [{"type": "text", "text": new_system}]
        }
        save_chat_history(st.session_state.chat_history)
        st.toast("🔍 Режим изменён")

    if st.session_state.get("vin_code"):
        st.markdown("---")
        st.info(f"🆔 **VIN:**\n`{st.session_state.vin_code}`")

    st.markdown("---")
    st.subheader("🧪 Симулятор")

    current_scenarios = TEST_SCENARIOS.get(st.session_state.diagnostic_mode, {})
    test_scenario = st.selectbox("Сценарий:", list(current_scenarios.keys()))
    scenario_key = current_scenarios[test_scenario]

    if st.button("⚡ Сгенерировать лог"):
        st.session_state.generated_log_df = generate_test_log_df(
            scenario_key,
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
        if AUDIO_MODULE_OK:
            cleanup_all_temp_files()
        st.rerun()

    if st.button("🚪 Выйти"):
        try:
            if "auth" in st.query_params:
                st.query_params.clear()
        except Exception:
            pass
        if "chat_history" in st.session_state:
            del st.session_state.chat_history
        st.session_state.authenticated = False
        st.rerun()


# ==================== ОСНОВНОЙ ЭКРАН ====================

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


# ==================== ЗАГРУЗКА ДАННЫХ ====================

st.subheader("📁 Загрузка данных")

st.info("📌 **Важно:** Логи VCDS снимайте на **БЕНЗИНЕ** для точной диагностики. ГБО-анализ — только в текстовом чате.")

uploaded_file = st.file_uploader(
    "Лог VCDS (.csv/.txt) или скриншот (.png/.jpg)",
    type=["csv", "txt", "png", "jpg", "jpeg"],
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


# ==================== ГРАФИКИ ====================

if log_df is not None and not log_df.empty:
    st.success("📊 Данные лога распознаны (режим: бензин)")
    time_col = log_df.columns[0]

    with st.expander("📊 График параметров", expanded=True):
        if st.session_state.diagnostic_mode.startswith("Электрика"):
            selected_cols = [c for c in log_df.columns if any(x in c for x in ["АКПП", "АБС", "Приборка", "SRS"])]
            if selected_cols:
                fig = px.line(log_df, x=time_col, y=selected_cols,
                              title="Статус CAN-связи (1=ОК, 0=обрыв, -1=блок отсутствует)",
                              template="plotly_dark")
                fig.update_yaxes(range=[-1.2, 1.2], tickvals=[-1, 0, 1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            selected_cols = [c for c in log_df.columns if any(x in c for x in [
                "Давление", "коррекция", "Пропуски", "Откат", "G187", "G79",
                "Угол дросселя", "Напряжение ДД", "Температура ОЖ", "Температура впуска",
                "Фазовое положение", "Сопротивление Зонда", "Конверсия катализатора",
                "Напряжение Зонда", "Датчик дросселя", "Педаль газа"
            ])]

            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Scatter(x=log_df[time_col], y=log_df[col], mode='lines', name=col))

                    for ref_key, values in st.session_state.reference_map.items():
                        if len(values) == 4 and ref_key.lower() in col.lower():
                            low, high, c_low, c_high = values
                            fig.add_hline(y=low, line_dash="dot", line_color=c_low,
                                          annotation_text=f"Мин {ref_key}")
                            fig.add_hline(y=high, line_dash="dot", line_color=c_high,
                                          annotation_text=f"Макс {ref_key}")

                fig.update_layout(
                    template="plotly_dark", 
                    title="Параметры с заводскими допусками VAG (бензин)",
                    xaxis_title="Время (сек)", 
                    yaxis_title="Значение"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(log_df.head(10))

    if st.button("🧠 Запустить экспертный анализ лога"):
        if not API_KEY:
            st.error("API-ключ не найден! Добавь OPENROUTER_API_KEY в Secrets (Settings → Secrets).")
        else:
            log_summary_text = generate_log_summary(log_df, st.session_state.vin_code, st.session_state.is_base_trim)

            system_instruction = get_system_prompt(
                mode=st.session_state.diagnostic_mode,
                is_base_trim=st.session_state.is_base_trim,
                mods=st.session_state.mods
            )

            user_msg = {
                "role": "user", 
                "content": [{"type": "text", "text": f"Привет! Я загрузил CSV-лог диагностики (снятый на БЕНЗИНЕ). Вот статистический срез данных:\n\n{log_summary_text}\n\nПроанализируй эти параметры, найди скрытые аномалии, поставь диагноз и распиши пошаговый план ремонта."}]
            }

            st.session_state.chat_history.append(user_msg)

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
                *st.session_state.chat_history[1:],
            ]

            with st.spinner("VAG Expert анализирует лог (бензин)..."):
                ai_response = ask_ai_chat(API_KEY, MODEL_NAME, messages, max_tokens=3000)

            st.session_state.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": ai_response}]})
            save_chat_history(st.session_state.chat_history)
            st.rerun()


# ==================== СКРИНШОТ VCDS ====================

if image_base64 is not None:
    st.success("🖼️ Скриншот готов")
    if st.button("👁️ Отправить скриншот в Gemini"):
        if not API_KEY:
            st.error("API-ключ не найден! Добавь OPENROUTER_API_KEY в Secrets.")
        else:
            prompt_text = "Пользователь загрузил скриншот окна VCDS. Распознай коды ошибок или группы измерений и выдай структурированный диагностический вердикт."
            if st.session_state.vin_code:
                prompt_text = f"VIN: {st.session_state.vin_code}. " + prompt_text

            msg = {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]
            }
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


# ==================== АУДИО/ВИДЕО ДИАГНОСТИКА ====================

st.markdown("---")
st.subheader("🎙️ Аудио/Видео диагностика мотора")

if not AUDIO_MODULE_OK:
    st.warning(f"⚠️ Аудио-модуль не загружен: {AUDIO_MODULE_ERROR}")
    st.info("Установите: pip install librosa soundfile scipy matplotlib")
else:
    st.caption("📹 Запиши 5–10 секунд работы мотора. Для точности загрузи также CSV-лог VCDS (на бензине) — RPM подтянутся автоматически.")

    audio_file = st.file_uploader(
        "Загрузи видео или аудио записи работы мотора",
        type=["mp4", "avi", "mov", "mkv", "wav", "mp3", "m4a", "ogg"],
        key="audio_diagnosis_uploader"
    )

    if audio_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.audio(audio_file)
        with col2:
            audio_temp = st.radio(
                "Температура мотора:",
                ["Холодный", "Прогретый", "Горячий"],
                index=1,
                key="engine_temp_audio"
            )
            temp_map = {"Холодный": "cold", "Прогретый": "warm", "Горячий": "hot"}

            rpm_default = 840.0
            if log_df is not None and not log_df.empty:
                rpm_col = [c for c in log_df.columns if "Обороты" in c or "RPM" in c.upper()]
                if rpm_col:
                    rpm_default = float(log_df[rpm_col[0]].mean())

            rpm_input = st.number_input(
                "Обороты двигателя (об/мин):",
                value=rpm_default,
                min_value=0.0, max_value=8000.0, step=10.0,
                key="audio_rpm_input",
                help="Автоподстановка из CSV-лога. Для CFNA на ХХ ≈ 840 об/мин."
            )

        if st.button("🔊 Запустить акустический анализ", key="run_audio_analysis"):
            with st.spinner("Анализирую звук... 10-20 секунд"):
                try:
                    result = analyze_engine_audio(
                        audio_file,
                        rpm=rpm_input,
                        engine_temp=temp_map[audio_temp],
                        has_lpg=st.session_state.mods.get("lpg", False)
                    )

                    if not result["success"]:
                        st.error(f"❌ Ошибка: {result['error']}")
                    else:
                        st.success("✅ Акустический анализ завершён")

                        if result.get("spectrogram_path") and os.path.exists(result["spectrogram_path"]):
                            st.image(result["spectrogram_path"], caption="Спектрограмма", use_container_width=True)

                        if result.get("spectrum_plot_path") and os.path.exists(result["spectrum_plot_path"]):
                            st.image(result["spectrum_plot_path"], caption="Доминирующие частоты", use_container_width=True)

                        st.subheader("📊 Локальный анализ")

                        with st.expander("Сырые акустические признаки"):
                            if result.get("raw_features"):
                                for k, v in result["raw_features"].items():
                                    if isinstance(v, float):
                                        st.write(f"**{k}:** {v:.4f}")
                                    else:
                                        st.write(f"**{k}:** {v}")

                        if result.get("sound_scores"):
                            st.write("**Обнаруженные паттерны:**")
                            for s in result["sound_scores"][:5]:
                                urgency_emoji = {
                                    "Низкая": "🟢", "Средняя": "🟡",
                                    "Высокая": "🔴", "Критическая": "🆘"
                                }
                                urgency_key = s["urgency"].split(" — ")[0] if " — " in s["urgency"] else s["urgency"]
                                emoji = urgency_emoji.get(urgency_key, "⚪")

                                with st.container():
                                    st.write(f"{emoji} **{s['name']}** — уверенность {s['confidence']*100:.0f}%")
                                    st.caption(f"Диапазон: {s['freq_range']} | Срочность: {s['urgency']}")
                                    if s.get("vcds_groups"):
                                        st.caption(f"Проверить VCDS группы: {', '.join(s['vcds_groups'])}")
                                    st.caption(f"Рекомендация: {s['typical_fix']}")
                        else:
                            st.info("Локальный анализ не выявил паттернов. Звук в пределах нормы.")

                        st.subheader("🤖 Экспертный анализ Gemini")

                        if st.button("👁️ Отправить спектрограмму в Gemini", key="send_to_gemini_audio"):
                            if not API_KEY:
                                st.error("API-ключ не найден!")
                            else:
                                img_b64 = get_image_base64(result["spectrogram_path"])

                                messages = [
                                    {"role": "system", "content": [{"type": "text", "text": "Ты — эксперт по акустической диагностике двигателей VAG. Анализируй спектрограммы и частотные данные."}]},
                                    {"role": "user", "content": [
                                        {"type": "text", "text": result["prompt_for_gemini"]},
                                        {"type": "image_url", "image_url": {"url": img_b64}}
                                    ]}
                                ]

                                with st.spinner("Gemini анализирует спектрограмму..."):
                                    gemini_response = ask_ai_chat(API_KEY, MODEL_NAME, messages, max_tokens=2500)

                                st.markdown(gemini_response)

                                st.session_state.chat_history.append({
                                    "role": "user",
                                    "content": [{"type": "text", "text": f"[Аудио-диагностика] {audio_file.name}"}]
                                })
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": gemini_response}]
                                })
                                save_chat_history(st.session_state.chat_history)

                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")


# ==================== ЧАТ-ВВОД ====================

if user_input := st.chat_input("Напиши симптомы или задай вопрос..."):
    if not API_KEY:
        st.error("API-ключ не найден! Добавь OPENROUTER_API_KEY в Secrets (Settings → Secrets).")
    else:
        with st.chat_message("user"):
            st.write(user_input)

        vin_match = re.search(r'\b([A-HJ-NPR-Z0-9]{17})\b', user_input, re.IGNORECASE)
        if vin_match and vin_match.group(1).upper() != st.session_state.vin_code:
            st.session_state.vin_code = vin_match.group(1).upper()
            save_profile(st.session_state.vin_code, st.session_state.is_base_trim, st.session_state.mods)
            st.sidebar.info(f"📍 VIN обновлён: {st.session_state.vin_code}")

        mods = st.session_state.mods
        mods_list = []
        if mods["tuned"]: mods_list.append("Чип-тюнинг")
        if mods["decatted"]: mods_list.append("Кат удалён")
        if mods["lpg"]: mods_list.append("ГБО")
        mods_str = ", ".join(mods_list) if mods_list else "Сток"
        vin_str = st.session_state.vin_code if st.session_state.vin_code else "Не указан"
        base_str = "BASE" if st.session_state.is_base_trim else "Comfortline/Highline"

        ai_text = f"[Контекст: VIN={vin_str}, Компл={base_str}, Моды={mods_str}] {user_input}"

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
