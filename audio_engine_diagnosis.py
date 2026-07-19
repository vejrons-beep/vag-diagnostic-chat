"""
Модуль аудио-диагностики двигателя CFNA 1.6 MPI
Поддерживает: .mp4, .avi, .mov (видео), .wav, .mp3, .m4a, .ogg (аудио)
Интеграция: Streamlit + Gemini 2.5 Flash через OpenRouter
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tempfile
import subprocess
import re
from typing import Dict, List, Tuple, Optional, Union
import json

# --- КОНФИГУРАЦИЯ ---
SAMPLE_RATE = 22050          # Целевая частота дискретизации
N_FFT = 4096                 # Размер окна FFT (разрешение по частоте)
HOP_LENGTH = 512             # Шаг окна
DURATION_LIMIT = 30.0        # Макс. длительность анализа (сек)
MIN_FREQ = 50                # Мин. частота для анализа (убираем выхлоп/ветер)
MAX_FREQ = 8000              # Макс. частота (выше — только свист роликов)

# Частотные диапазоны характерных звуков CFNA
CFNA_SOUND_PROFILES = {
    "chain_rattle_cold": {
        "name": "Стук цепи ГРМ (холодная)",
        "freq_range": (500, 1500),
        "harmonic_of_rpm": True,
        "rpm_multiplier": 1.0,
        "confidence_threshold": 0.6,
        "description": "Металлический ритмичный стук, усиливается при запуске, затихает через 2-3 минуты",
        "vcds_groups": ["093", "004"],
        "typical_fix": "Замена успокоителя цепи (06H109467N) или натяжителя (06H109507M)",
        "urgency": "Средняя — 10-15 тыс. км"
    },
    "valve_tick": {
        "name": "Тик гидрокомпенсаторов",
        "freq_range": (2000, 4000),
        "harmonic_of_rpm": True,
        "rpm_multiplier": 0.5,
        "confidence_threshold": 0.55,
        "description": "Высокий металлический тик, регулярный, часто на холодную или при низком давлении масла",
        "vcds_groups": ["007", "022"],
        "typical_fix": "Замена масла 5W-40 VW 502.00, проверка давления масла, при неудаче — гидрокомпенсаторы (036109651)",
        "urgency": "Низкая — если проходит прогревом"
    },
    "alternator_bearing": {
        "name": "Подшипник генератора",
        "freq_range": (1000, 3500),
        "harmonic_of_rpm": True,
        "rpm_multiplier": 2.5,
        "confidence_threshold": 0.5,
        "description": "Нарастающий гул/вой с оборотами, не зависит от температуры мотора",
        "vcds_groups": ["004"],
        "typical_fix": "Замена подшипников генератора (6203-2RS + 6202-2RS) или генератора целиком",
        "urgency": "Средняя — может заклинить, оборвать ремень"
    },
    "tensioner_roller": {
        "name": "Обводной/натяжной ролик ремня ГРМ",
        "freq_range": (800, 3000),
        "harmonic_of_rpm": True,
        "rpm_multiplier": 1.0,
        "confidence_threshold": 0.5,
        "description": "Свист или гул, усиливается с оборотами, пропадает при снятии ремня",
        "vcds_groups": [],
        "typical_fix": "Замена ролика натяжителя (03C145299C) или обводного (03C145276B)",
        "urgency": "Высокая — обрыв ремня = загиб клапанов"
    },
    "water_pump": {
        "name": "Помпа системы охлаждения",
        "freq_range": (600, 2000),
        "harmonic_of_rpm": True,
        "rpm_multiplier": 1.2,
        "confidence_threshold": 0.5,
        "description": "Ровный гул, иногда с металлическим скрежем, может пульсировать",
        "vcds_groups": ["007"],
        "typical_fix": "Замена помпы (03C121004J) с прокладкой",
        "urgency": "Средняя — течь ОЖ, перегрев"
    },
    "piston_knock": {
        "name": "Стук поршневых пальцев",
        "freq_range": (1000, 3000),
        "harmonic_of_rpm": True,
        "rpm_multiplier": 0.5,
        "confidence_threshold": 0.45,
        "description": "Глухой стук под нагрузкой, особенно при резком газе с низких оборотов",
        "vcds_groups": ["015", "016"],
        "typical_fix": "Капитальный ремонт — замена поршней с пальцами (036107065N)",
        "urgency": "Высокая — разрушение поршня"
    },
    "con_rattle": {
        "name": "Стук шатунных вкладышей",
        "freq_range": (50, 500),
        "harmonic_of_rpm": True,
        "rpm_multiplier": 0.5,
        "confidence_threshold": 0.4,
        "description": "Низкий тяжёлый стук, металлический, на холодную или при масляном голодании",
        "vcds_groups": ["007"],
        "typical_fix": "Срочная диагностика давления масла, замена вкладышей (036105591A)",
        "urgency": "Критическая — вращательный удар, заклинивание"
    },
    "lpg_injector_click": {
        "name": "Щелчки газовых форсунок ГБО",
        "freq_range": (50, 200),
        "harmonic_of_rpm": False,
        "rpm_multiplier": None,
        "confidence_threshold": 0.3,
        "description": "Регулярные щелчки низкой частоты, характерны только при работе на газе",
        "vcds_groups": [],
        "typical_fix": "Норма работы ГБО, неисправность если щелчки нерегулярные или сопровождаются пропусками",
        "urgency": "Низкая — норма"
    }
}


# --- ИЗВЛЕЧЕНИЕ АУДИО ---

def extract_audio_from_video(video_path: str, output_audio_path: Optional[str] = None) -> str:
    """
    Извлекает аудио из видео через ffmpeg.
    Возвращает путь к .wav файлу.
    """
    if output_audio_path is None:
        output_audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        output_audio_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")
        return output_audio_path
    except FileNotFoundError:
        raise RuntimeError("ffmpeg не установлен. Установите: sudo apt-get install ffmpeg")


def load_audio(file_path: str, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Загружает аудио файл. Поддерживает любой формат, который понимает librosa.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=duration or DURATION_LIMIT)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки аудио: {e}")


# --- ПРЕДОБРАБОТКА ---

def preprocess_audio(y: np.ndarray, sr: int, 
                     filter_low: float = MIN_FREQ, 
                     filter_high: float = MAX_FREQ) -> np.ndarray:
    """
    Фильтрация: high-pass + low-pass + нормализация.
    Убирает выхлоп, ветер, ультразвук.
    """
    # High-pass: убираем низкие частоты (выхлоп, ветер, гул ГБО)
    y_hp = librosa.effects.preemphasis(y, coef=0.97)

    # STFT-based bandpass
    D = librosa.stft(y_hp, n_fft=N_FFT, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # Обнуляем частоты вне диапазона
    mask = (freqs >= filter_low) & (freqs <= filter_high)
    D_filtered = D * mask[:, np.newaxis]

    # Обратное преобразование
    y_filtered = librosa.istft(D_filtered, hop_length=HOP_LENGTH, length=len(y_hp))

    # Нормализация
    if np.max(np.abs(y_filtered)) > 0:
        y_filtered = y_filtered / np.max(np.abs(y_filtered))

    return y_filtered


def detect_stable_segments(y: np.ndarray, sr: int, 
                           frame_length: int = 2048,
                           min_duration_sec: float = 2.0) -> List[Tuple[int, int]]:
    """
    Находит стабильные участки записи (без резких всплесков, запуска, глушения).
    Возвращает список (start_sample, end_sample).
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=HOP_LENGTH)[0]

    # Скользящее среднее для сглаживания
    window = int(sr / HOP_LENGTH * 0.5)  # 0.5 сек
    rms_smooth = np.convolve(rms, np.ones(window)/window, mode='same')

    # Находим участки с RMS в пределах [0.3*mean, 2.0*mean]
    mean_rms = rms_smooth.mean()
    mask = (rms_smooth > mean_rms * 0.3) & (rms_smooth < mean_rms * 2.5)

    # Группируем последовательные True
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_segment:
            in_segment = True
            start = i
        elif not val and in_segment:
            in_segment = False
            end = i
            duration = (end - start) * HOP_LENGTH / sr
            if duration >= min_duration_sec:
                segments.append((start * HOP_LENGTH, end * HOP_LENGTH))

    # Если не нашли стабильных — берём середину записи
    if not segments:
        mid = len(y) // 2
        half = int(min_duration_sec * sr / 2)
        segments.append((max(0, mid - half), min(len(y), mid + half)))

    return segments


# --- АНАЛИЗ СПЕКТРА ---

def compute_spectrogram(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляет спектрограмму в dB.
    Возвращает: S_db, freqs, times
    """
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    return S_db, freqs, times


def find_dominant_frequencies(S_db: np.ndarray, freqs: np.ndarray, 
                               top_n: int = 10) -> List[Tuple[float, float]]:
    """
    Находит доминирующие частоты (усреднённый спектр по времени).
    Возвращает [(freq_hz, magnitude_db), ...]
    """
    # Усредняем по времени
    mean_spectrum = S_db.mean(axis=1)

    # Находим пики
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(mean_spectrum, height=-60, distance=20)

    # Сортируем по амплитуде
    peak_data = [(freqs[p], mean_spectrum[p]) for p in peaks]
    peak_data.sort(key=lambda x: x[1], reverse=True)

    return peak_data[:top_n]


def analyze_harmonics(dominant_freqs: List[Tuple[float, float]], 
                      rpm: Optional[float] = None) -> Dict:
    """
    Анализирует гармоники доминирующих частот относительно оборотов мотора.
    """
    if rpm is None:
        return {"engine_freq": None, "harmonics": []}

    engine_freq = rpm / 60.0  # Гц
    harmonics = []

    for freq, mag in dominant_freqs:
        if engine_freq > 0:
            ratio = freq / engine_freq
            nearest_harmonic = round(ratio)
            deviation = abs(ratio - nearest_harmonic)

            if deviation < 0.15:  # ±15% от гармоники
                harmonics.append({
                    "freq": freq,
                    "harmonic_n": nearest_harmonic,
                    "ratio": ratio,
                    "magnitude_db": mag,
                    "matches_engine": True
                })

    return {
        "engine_freq": engine_freq,
        "harmonics": harmonics
    }


def score_sound_profiles(S_db: np.ndarray, freqs: np.ndarray, 
                         harmonic_analysis: Dict,
                         engine_temp: str = "warm") -> List[Dict]:
    """
    Сопоставляет спектр с профилями звуков CFNA.
    Возвращает отсортированный список с confidence score.
    """
    scores = []
    mean_spectrum = S_db.mean(axis=1)

    for key, profile in CFNA_SOUND_PROFILES.items():
        # Пропускаем «холодные» звуки если мотор прогрет
        if "cold" in key and engine_temp == "warm":
            continue

        # Вычисляем энергию в диапазоне профиля
        f_low, f_high = profile["freq_range"]
        idx_low = np.argmin(np.abs(freqs - f_low))
        idx_high = np.argmin(np.abs(freqs - f_high))

        band_energy = np.mean(mean_spectrum[idx_low:idx_high])
        band_peak = np.max(mean_spectrum[idx_low:idx_high])

        # Нормализуем score (0-1)
        score = (band_peak + 60) / 60  # -60dB = 0, 0dB = 1
        score = np.clip(score, 0, 1)

        # Проверяем гармоничность
        if profile["harmonic_of_rpm"] and harmonic_analysis.get("engine_freq"):
            expected_freq = harmonic_analysis["engine_freq"] * profile["rpm_multiplier"]
            # Ищем ближайшую доминирующую частоту к ожидаемой
            closest = min(harmonic_analysis["harmonics"], 
                         key=lambda h: abs(h["freq"] - expected_freq), 
                         default=None)
            if closest and abs(closest["freq"] - expected_freq) / expected_freq < 0.2:
                score *= 1.3  # Бонус за совпадение с гармоникой оборотов

        if score >= profile["confidence_threshold"]:
            scores.append({
                "key": key,
                "name": profile["name"],
                "confidence": min(score, 1.0),
                "freq_range": f"{f_low}-{f_high} Гц",
                "description": profile["description"],
                "vcds_groups": profile["vcds_groups"],
                "typical_fix": profile["typical_fix"],
                "urgency": profile["urgency"]
            })

    scores.sort(key=lambda x: x["confidence"], reverse=True)
    return scores


# --- ВИЗУАЛИЗАЦИЯ ---

def create_spectrogram_image(S_db: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                           title: str = "Спектрограмма звука мотора CFNA",
                           figsize: Tuple[int, int] = (12, 6)) -> str:
    """
    Создаёт PNG спектрограммы. Возвращает путь к файлу.
    """
    fig, ax = plt.subplots(figsize=figsize)

    img = librosa.display.specshow(S_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                    x_axis='time', y_axis='hz', ax=ax,
                                    cmap='magma')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Время (сек)", fontsize=12)
    ax.set_ylabel("Частота (Гц)", fontsize=12)
    ax.set_ylim(MIN_FREQ, MAX_FREQ)

    plt.colorbar(img, ax=ax, format='%+2.0f dB', label='Амплитуда (dB)')
    plt.tight_layout()

    # Сохраняем во временный файл
    tmp_path = tempfile.mktemp(suffix=".png")
    plt.savefig(tmp_path, dpi=150, bbox_inches='tight')
    plt.close()

    return tmp_path


def create_spectrum_plot(dominant_freqs: List[Tuple[float, float]], 
                         harmonic_analysis: Dict,
                         figsize: Tuple[int, int] = (10, 5)) -> str:
    """
    Создаёт график доминирующих частот с отметкой гармоник оборотов.
    """
    fig, ax = plt.subplots(figsize=figsize)

    freqs = [f for f, _ in dominant_freqs]
    mags = [m for _, m in dominant_freqs]

    bars = ax.bar(range(len(freqs)), mags, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f:.0f}" for f in freqs], rotation=45)
    ax.set_xlabel("Частота (Гц)", fontsize=12)
    ax.set_ylabel("Амплитуда (dB)", fontsize=12)
    ax.set_title("Доминирующие частоты", fontsize=14, fontweight='bold')
    ax.axhline(y=-40, color='red', linestyle='--', alpha=0.5, label='Порог значимости')

    # Отмечаем гармоники оборотов
    if harmonic_analysis.get("engine_freq"):
        ef = harmonic_analysis["engine_freq"]
        for n in range(1, 11):
            ax.axvline(x=n * ef * len(freqs) / max(freqs) if freqs else 0, 
                      color='green', linestyle=':', alpha=0.3)

    ax.legend()
    plt.tight_layout()

    tmp_path = tempfile.mktemp(suffix=".png")
    plt.savefig(tmp_path, dpi=150, bbox_inches='tight')
    plt.close()

    return tmp_path


# --- ПОЛНЫЙ КОНВЕЙЕР ---

def analyze_engine_audio(uploaded_file, 
                         rpm: Optional[float] = None,
                         engine_temp: str = "warm",
                         has_lpg: bool = False) -> Dict:
    """
    Полный конвейер анализа аудио/видео файла.

    Args:
        uploaded_file: Streamlit UploadedFile или путь к файлу
        rpm: Обороты двигателя (из лога VCDS)
        engine_temp: "cold" | "warm" | "hot"
        has_lpg: Установлено ли ГБО

    Returns:
        Dict с результатами анализа
    """
    result = {
        "success": False,
        "error": None,
        "spectrogram_path": None,
        "spectrum_plot_path": None,
        "dominant_frequencies": [],
        "harmonic_analysis": {},
        "sound_scores": [],
        "prompt_for_gemini": "",
        "raw_features": {}
    }

    try:
        # 1. Определяем тип файла
        if hasattr(uploaded_file, 'name'):
            file_name = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
        else:
            file_name = os.path.basename(uploaded_file)
            with open(uploaded_file, 'rb') as f:
                file_bytes = f.read()

        # 2. Сохраняем во временный файл
        suffix = os.path.splitext(file_name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # 3. Извлекаем аудио если видео
        audio_path = tmp_path
        if suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            audio_path = extract_audio_from_video(tmp_path)

        # 4. Загружаем аудио
        y, sr = load_audio(audio_path, duration=DURATION_LIMIT)

        # 5. Предобработка
        y_filtered = preprocess_audio(y, sr)

        # 6. Находим стабильные сегменты
        segments = detect_stable_segments(y_filtered, sr)

        if not segments:
            result["error"] = "Не удалось найти стабильные участки записи"
            return result

        # Берём самый длинный стабильный сегмент
        best_segment = max(segments, key=lambda s: s[1] - s[0])
        y_stable = y_filtered[best_segment[0]:best_segment[1]]

        # 7. Спектрограмма
        S_db, freqs, times = compute_spectrogram(y_stable, sr)
        result["spectrogram_path"] = create_spectrogram_image(S_db, freqs, times)

        # 8. Доминирующие частоты
        dominant = find_dominant_frequencies(S_db, freqs, top_n=15)
        result["dominant_frequencies"] = [(f"{f:.0f}", f"{m:.1f}") for f, m in dominant]

        # 9. Гармонический анализ
        harmonic = analyze_harmonics(dominant, rpm)
        result["harmonic_analysis"] = harmonic

        # 10. Спектр-плот
        result["spectrum_plot_path"] = create_spectrum_plot(dominant, harmonic)

        # 11. Скоринг профилей
        scores = score_sound_profiles(S_db, freqs, harmonic, engine_temp)
        result["sound_scores"] = scores

        # 12. Формируем промпт для Gemini
        result["prompt_for_gemini"] = _build_gemini_prompt(
            dominant, harmonic, scores, rpm, engine_temp, has_lpg
        )

        # 13. Сырые признаки
        result["raw_features"] = {
            "rms_mean": float(np.sqrt(np.mean(y_stable**2))),
            "rms_std": float(np.std(y_stable)),
            "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(y_stable).mean()),
            "spectral_centroid": float(librosa.feature.spectral_centroid(y=y_stable, sr=sr).mean()),
            "spectral_rolloff": float(librosa.feature.spectral_rolloff(y=y_stable, sr=sr).mean()),
            "duration_analyzed": len(y_stable) / sr
        }

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
    finally:
        # Чистим временные файлы
        for p in [tmp_path, audio_path]:
            if p and os.path.exists(p) and p != uploaded_file:
                try:
                    os.unlink(p)
                except:
                    pass

    return result


def _build_gemini_prompt(dominant_freqs: List[Tuple[float, float]], 
                         harmonic_analysis: Dict,
                         scores: List[Dict],
                         rpm: Optional[float],
                         engine_temp: str,
                         has_lpg: bool) -> str:
    """
    Строит текстовый промпт для Gemini на основе анализа.
    """
    lines = [
        "Ты — эксперт по акустической диагностике двигателей VAG.",
        "Анализируй спектрограмму и данные частотного анализа.",
        "",
        "=== КОНТЕКСТ ЗАПИСИ ===",
        f"Двигатель: CFNA 1.6 MPI (105 л.с.), Magneti Marelli 7GV",
        f"Температура: {engine_temp}",
    ]

    if rpm:
        lines.append(f"Обороты: {rpm:.0f} об/мин (частота вращения {rpm/60:.1f} Гц)")
    else:
        lines.append("Обороты: неизвестны (не предоставлен лог VCDS)")

    if has_lpg:
        lines.append("ГБО: установлено (возможны щелчки форсунок 50-200 Гц — игнорировать)")

    lines.extend([
        "",
        "=== ДОМИНИРУЮЩИЕ ЧАСТОТЫ ===",
    ])

    for i, (freq, mag) in enumerate(dominant_freqs[:10], 1):
        lines.append(f"{i}. {freq:.0f} Гц @ {mag:.1f} dB")

    if harmonic_analysis.get("harmonics"):
        lines.extend([
            "",
            "=== ГАРМОНИКИ ОБОРОТОВ ===",
        ])
        for h in harmonic_analysis["harmonics"][:5]:
            lines.append(f"- {h['freq']:.0f} Гц = {h['harmonic_n']}-я гармоника (отклонение {h['ratio']:.2f}x)")

    if scores:
        lines.extend([
            "",
            "=== ВЕРОЯТНОСТНЫЙ АНАЛИЗ (локальный) ===",
        ])
        for s in scores[:5]:
            lines.append(f"- {s['name']}: {s['confidence']*100:.0f}% | {s['freq_range']} | {s['urgency']}")

    lines.extend([
        "",
        "=== ИНСТРУКЦИЯ ===",
        "1. Проанализируй спектрограмму (изображение выше).",
        "2. Учитывай, что в записи могут быть помехи:",
        "   - Помпа: ровный гул 600-2000 Гц, гармоника оборотов x1.2",
        "   - Генератор: нарастающий гул, гармоника x2.5",
        "   - Ролики ремня ГРМ: свист 800-3000 Гц",
        "   - ГБО форсунки: щелчки 50-200 Гц (если has_lpg=True)",
        "3. Ищи ИМПУЛЬСНЫЕ пики на гармониках оборотов — это моторный стук.",
        "4. Постоянный гармонический гул — скорее навесное оборудование.",
        "5. Дай вероятностный вердикт (0-100%) для каждой возможной неисправности.",
        "6. Укажи номера групп VCDS для проверки.",
        "7. Оцени срочность ремонта.",
    ])

    return "\n".join(lines)


# --- ИНТЕГРАЦИЯ СО STREAMLIT ---

def render_audio_diagnosis_ui(api_key: str, model_name: str, 
                              ask_ai_func,
                              current_rpm: Optional[float] = None,
                              current_temp: str = "warm",
                              has_lpg: bool = False):
    """
    Рендерит UI для аудио-диагностики в Streamlit.
    Встраивается в основное приложение.

    Args:
        api_key: OPENROUTER_API_KEY
        model_name: например "google/gemini-2.5-flash"
        ask_ai_func: ваша функция ask_ai_chat
        current_rpm: обороты из лога VCDS (если загружен)
        current_temp: "cold" | "warm" | "hot"
        has_lpg: из st.session_state.mods["lpg"]
    """
    st.markdown("---")
    st.subheader("🎙️ Аудио/Видео диагностика мотора")

    audio_file = st.file_uploader(
        "Загрузите видео или аудио записи работы мотора",
        type=["mp4", "avi", "mov", "mkv", "wav", "mp3", "m4a", "ogg"],
        key="audio_diagnosis_uploader"
    )

    if audio_file is None:
        return None

    col1, col2 = st.columns(2)
    with col1:
        st.audio(audio_file)
    with col2:
        temp = st.radio(
            "Температура мотора при записи:",
            ["cold", "warm", "hot"],
            index=["cold", "warm", "hot"].index(current_temp),
            key="engine_temp_audio"
        )
        rpm_input = st.number_input(
            "Обороты (из лога VCDS):",
            value=float(current_rpm) if current_rpm else 840.0,
            min_value=0.0,
            max_value=8000.0,
            step=10.0,
            key="audio_rpm_input"
        )

    if st.button("🔊 Запустить акустический анализ", key="run_audio_analysis"):
        with st.spinner("Анализирую звук... Это может занять 10-20 секунд"):
            result = analyze_engine_audio(
                audio_file,
                rpm=rpm_input,
                engine_temp=temp,
                has_lpg=has_lpg
            )

        if not result["success"]:
            st.error(f"❌ Ошибка анализа: {result['error']}")
            return result

        st.success("✅ Анализ завершён")

        # Показываем спектрограмму
        if result["spectrogram_path"] and os.path.exists(result["spectrogram_path"]):
            st.image(result["spectrogram_path"], caption="Спектрограмма звука мотора", use_container_width=True)

        # Показываем спектр-плот
        if result["spectrum_plot_path"] and os.path.exists(result["spectrum_plot_path"]):
            st.image(result["spectrum_plot_path"], caption="Доминирующие частоты", use_container_width=True)

        # Локальные результаты
        st.subheader("📊 Локальный анализ (без ИИ)")

        with st.expander("Сырые признаки"):
            st.json(result["raw_features"])

        if result["sound_scores"]:
            st.write("**Обнаруженные паттерны:**")
            for s in result["sound_scores"][:5]:
                urgency_color = {"Низкая": "🟢", "Средняя": "🟡", "Высокая": "🔴", "Критическая": "🆘"}
                color = urgency_color.get(s["urgency"].split(" — ")[0], "⚪")
                st.write(f"{color} **{s['name']}** — {s['confidence']*100:.0f}%")
                st.caption(f"Диапазон: {s['freq_range']} | Срочность: {s['urgency']}")
                if s["vcds_groups"]:
                    st.caption(f"Проверить группы VCDS: {', '.join(s['vcds_groups'])}")
        else:
            st.info("Локальный анализ не выявил характерных паттернов. Звук в пределах нормы или требует экспертной оценки.")

        # Отправляем в Gemini
        st.subheader("🤖 Экспертный анализ Gemini")

        if st.button("👁️ Отправить спектрограмму в Gemini", key="send_to_gemini_audio"):
            if not api_key:
                st.error("API-ключ не найден!")
                return result

            # Кодируем спектрограмму в base64
            with open(result["spectrogram_path"], "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode()
            img_data = f"data:image/png;base64,{img_b64}"

            # Формируем сообщение
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "Ты — эксперт по акустической диагностике двигателей VAG. Анализируй спектрограммы и частотные данные."}]},
                {"role": "user", "content": [
                    {"type": "text", "text": result["prompt_for_gemini"]},
                    {"type": "image_url", "image_url": {"url": img_data}}
                ]}
            ]

            with st.spinner("Gemini анализирует спектрограмму..."):
                gemini_response = ask_ai_func(api_key, model_name, messages, max_tokens=2500)

            st.markdown(gemini_response)

            # Сохраняем в историю чата
            if "chat_history" in st.session_state:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"[Аудио-диагностика] {audio_file.name}"}]
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": gemini_response}]
                })

        return result

    return None


# --- УТИЛИТЫ ---

def cleanup_temp_files():
    """Очищает временные PNG файлы спектрограмм."""
    import glob
    for f in glob.glob(os.path.join(tempfile.gettempdir(), "*.png")):
        try:
            os.unlink(f)
        except:
            pass


if __name__ == "__main__":
    # Тестовый запуск
    print("Модуль audio_engine_diagnosis.py загружен успешно")
    print(f"Доступные профили звуков: {list(CFNA_SOUND_PROFILES.keys())}")
