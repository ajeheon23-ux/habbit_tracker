# app.py
import os
import re
from datetime import date, timedelta

import requests
import streamlit as st
import pandas as pd

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")
st.title("📊 AI 습관 트래커")

# ---------------------------
# Sidebar: API Keys
# ---------------------------
with st.sidebar:
    st.header("🔑 API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    owm_api_key = st.text_input("OpenWeatherMap API Key", type="password", value=os.getenv("OPENWEATHERMAP_API_KEY", ""))

    st.caption("팁: 로컬에서는 환경변수(OPENAI_API_KEY, OPENWEATHERMAP_API_KEY)로도 설정할 수 있어요.")

# ---------------------------
# Constants / Helpers
# ---------------------------
HABITS = [
    ("wake", "🌅", "기상 미션"),
    ("water", "💧", "물 마시기"),
    ("study", "📚", "공부/독서"),
    ("workout", "🏋️", "운동하기"),
    ("sleep", "😴", "수면"),
]

CITIES = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Ulsan", "Suwon", "Seongnam", "Jeju",
]

COACH_STYLES = {
    "스파르타 코치": {
        "system": (
            "너는 엄격하지만 공정한 '스파르타 코치'다. "
            "사용자의 변명을 허용하지 않고, 행동을 촉구하며, 짧고 단호하게 말한다. "
            "비난은 하지 말고, 개선 행동을 명확히 제시한다."
        )
    },
    "따뜻한 멘토": {
        "system": (
            "너는 따뜻하고 공감적인 '멘토'다. "
            "사용자의 감정과 상황을 존중하고, 작은 성취를 인정하며, "
            "현실적인 다음 कदम을 부드럽게 제안한다."
        )
    },
    "게임 마스터": {
        "system": (
            "너는 RPG 세계관의 '게임 마스터'다. "
            "사용자의 하루를 퀘스트/스탯/레벨업 관점에서 재미있게 해석한다. "
            "과장된 폭력/위협 없이 유쾌하게 동기부여하고, 내일 퀘스트를 제시한다."
        )
    },
}


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def calc_achievement(habit_dict: dict) -> tuple[int, float]:
    """Returns (checked_count, achievement_percent)."""
    checked = sum(1 for k, _, _ in HABITS if habit_dict.get(k))
    pct = round((checked / len(HABITS)) * 100, 1)
    return checked, pct


def init_sample_data():
    """Create 6 days of demo data."""
    # Fixed-ish demo pattern (deterministic) to avoid randomness surprises
    base = date.today() - timedelta(days=6)
    demo = []
    patterns = [
        {"wake": True, "water": True, "study": False, "workout": True, "sleep": True, "mood": 7},
        {"wake": True, "water": False, "study": True, "workout": False, "sleep": True, "mood": 6},
        {"wake": True, "water": True, "study": True, "workout": False, "sleep": False, "mood": 5},
        {"wake": False, "water": True, "study": True, "workout": True, "sleep": True, "mood": 8},
        {"wake": True, "water": True, "study": True, "workout": True, "sleep": False, "mood": 7},
        {"wake": True, "water": False, "study": False, "workout": True, "sleep": True, "mood": 6},
    ]
    for i in range(6):
        d = base + timedelta(days=i)
        row = {"date": d.isoformat(), "city": "Seoul"}
        row.update(patterns[i])
        demo.append(row)
    return demo


def ensure_state():
    if "history" not in st.session_state:
        st.session_state.history = init_sample_data()
    if "last_report" not in st.session_state:
        st.session_state.last_report = ""
    if "last_weather" not in st.session_state:
        st.session_state.last_weather = None
    if "last_dog" not in st.session_state:
        st.session_state.last_dog = None


def upsert_today_record(record: dict):
    """Insert or replace today's record in session_state.history."""
    today = date.today().isoformat()
    hist = st.session_state.history
    idx = next((i for i, r in enumerate(hist) if r.get("date") == today), None)
    if idx is None:
        hist.append(record)
    else:
        hist[idx] = record
    # keep only last 30 for sanity
    hist_sorted = sorted(hist, key=lambda r: r.get("date", ""))
    st.session_state.history = hist_sorted[-30:]


# ---------------------------
# API Integrations
# ---------------------------
def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap에서 날씨 가져오기 (한국어, 섭씨).
    실패 시 None 반환, timeout=10
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "kr",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        weather = {
            "city": city,
            "temp": data.get("main", {}).get("temp"),
            "feels_like": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "desc": (data.get("weather") or [{}])[0].get("description"),
            "icon": (data.get("weather") or [{}])[0].get("icon"),
        }
        if weather["temp"] is None or weather["desc"] is None:
            return None
        return weather
    except Exception:
        return None


def extract_dog_breed_from_url(url: str) -> str:
    """
    Dog CEO URL 예:
    https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
    -> hound afghan
    """
    try:
        m = re.search(r"/breeds/([^/]+)/", url)
        if not m:
            return "알 수 없음"
        raw = m.group(1).replace("-", " ").strip()
        return raw if raw else "알 수 없음"
    except Exception:
        return "알 수 없음"


def get_dog_image():
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기
    실패 시 None 반환, timeout=10
    """
    try:
        r = requests.get("https://dog.ceo/api/breeds/image/random", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        url = data.get("message")
        if not url:
            return None
        breed = extract_dog_breed_from_url(url)
        return {"url": url, "breed": breed}
    except Exception:
        return None


# ---------------------------
# AI Report
# ---------------------------
def generate_report(
    openai_api_key: str,
    coach_style: str,
    habits_today: dict,
    mood: int,
    weather: dict | None,
    dog: dict | None,
):
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달
    출력 형식:
    - 컨디션 등급(S~D)
    - 습관 분석
    - 날씨 코멘트
    - 내일 미션
    - 오늘의 한마디
    모델: gpt-5-mini
    """
    if not openai_api_key:
        return None, "OpenAI API Key가 필요해요."

    checked, pct = calc_achievement(habits_today)
    habit_lines = []
    for k, emo, label in HABITS:
        habit_lines.append(f"- {emo} {label}: {'✅' if habits_today.get(k) else '❌'}")

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')} / {weather.get('desc')} / "
            f"{weather.get('temp')}°C (체감 {weather.get('feels_like')}°C), 습도 {weather.get('humidity')}%"
        )

    dog_breed = (dog or {}).get("breed", "알 수 없음")

    system_prompt = COACH_STYLES.get(coach_style, COACH_STYLES["따뜻한 멘토"])["system"]

    user_prompt = f"""
[오늘 체크인]
달성률: {pct}% ({checked}/{len(HABITS)})
기분(1~10): {mood}

[습관]
{chr(10).join(habit_lines)}

[날씨]
{weather_text}

[랜덤 강아지 품종]
{dog_breed}

요구사항:
1) 반드시 한국어로 답해줘.
2) 아래 형식을 정확히 지켜줘. (제목 포함)
3) '컨디션 등급'은 S/A/B/C/D 중 하나로만 출력해줘.

형식:
컨디션 등급: <S|A|B|C|D>

습관 분석:
- (핵심 3~5줄, 구체적으로)

날씨 코멘트:
- (1~2줄)

내일 미션:
- (최대 3개, 체크리스트처럼)

오늘의 한마디:
- (짧고 임팩트 있게 1줄)
""".strip()

    try:
        from openai import OpenAI  # openai>=1.0.0

        client = OpenAI(api_key=openai_api_key)

        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        return text, None
    except Exception as e:
        return None, f"OpenAI 호출 실패: {e}"


# ---------------------------
# State init
# ---------------------------
ensure_state()

# ---------------------------
# Habit Check-in UI
# ---------------------------
st.subheader("✅ 오늘 체크인")

col_left, col_right = st.columns(2, gap="large")

# 2열 배치 체크박스(5개) - 왼쪽 3개, 오른쪽 2개
today_habits = {}
with col_left:
    for k, emo, label in HABITS[:3]:
        today_habits[k] = st.checkbox(f"{emo} {label}", value=False, key=f"habit_{k}")
with col_right:
    for k, emo, label in HABITS[3:]:
        today_habits[k] = st.checkbox(f"{emo} {label}", value=False, key=f"habit_{k}")

mood = st.slider("🙂 기분 (1~10)", min_value=1, max_value=10, value=6, step=1)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    city = st.selectbox("🌍 도시 선택", CITIES, index=0)
with c2:
    coach_style = st.radio("🧠 코치 스타일", list(COACH_STYLES.keys()), horizontal=True)

checked_cnt, achievement_pct = calc_achievement(today_habits)

# ---------------------------
# Metrics
# ---------------------------
st.subheader("📈 오늘 요약")
m1, m2, m3 = st.columns(3, gap="large")
with m1:
    st.metric("달성률", f"{achievement_pct}%")
with m2:
    st.metric("달성 습관", f"{checked_cnt}/{len(HABITS)}")
with m3:
    st.metric("기분", f"{mood}/10")

# Save to session_state
today_record = {"date": date.today().isoformat(), "city": city, "mood": mood}
today_record.update({k: bool(today_habits.get(k)) for k, _, _ in HABITS})
upsert_today_record(today_record)

# ---------------------------
# 7-day Bar Chart (6 demo + today)
# ---------------------------
st.subheader("🗓️ 최근 7일 달성률")

hist_df = pd.DataFrame(st.session_state.history)
# Ensure last 7 days exist; if gaps, we still show last 7 records in history
hist_df = hist_df.sort_values("date").tail(7).copy()

def row_achievement_pct(row):
    habit_dict = {k: bool(row.get(k)) for k, _, _ in HABITS}
    _, pct = calc_achievement(habit_dict)
    return pct

if not hist_df.empty:
    hist_df["achievement_pct"] = hist_df.apply(row_achievement_pct, axis=1)
    chart_df = hist_df.set_index("date")[["achievement_pct"]]
    st.bar_chart(chart_df)
else:
    st.info("표시할 기록이 없어요.")

# ---------------------------
# Results: Weather + Dog + AI Report
# ---------------------------
st.subheader("🧾 AI 코치 리포트")

btn = st.button("컨디션 리포트 생성", type="primary")

if btn:
    with st.spinner("날씨/강아지/AI 리포트 생성 중..."):
        weather = get_weather(city, owm_api_key)
        dog = get_dog_image()

        report, err = generate_report(
            openai_api_key=openai_api_key,
            coach_style=coach_style,
            habits_today=today_habits,
            mood=mood,
            weather=weather,
            dog=dog,
        )

        st.session_state.last_weather = weather
        st.session_state.last_dog = dog
        st.session_state.last_report = report if report else ""

        if err:
            st.error(err)

# Show cards + report (if available)
weather = st.session_state.last_weather
dog = st.session_state.last_dog
report = st.session_state.last_report

card1, card2 = st.columns(2, gap="large")

with card1:
    st.markdown("#### ☁️ 오늘의 날씨")
    if weather:
        st.write(f"**도시**: {weather.get('city')}")
        st.write(f"**상태**: {weather.get('desc')}")
        st.write(f"**기온**: {weather.get('temp')}°C (체감 {weather.get('feels_like')}°C)")
        st.write(f"**습도**: {weather.get('humidity')}%")
    else:
        st.info("날씨 정보를 가져오지 못했어요. (API Key/도시/네트워크 확인)")

with card2:
    st.markdown("#### 🐶 오늘의 강아지")
    if dog and dog.get("url"):
        st.write(f"**품종(추정)**: {dog.get('breed')}")
        st.image(dog.get("url"), use_container_width=True)
    else:
        st.info("강아지 이미지를 가져오지 못했어요. (네트워크 확인)")

st.markdown("#### 🧠 AI 리포트")
if report:
    st.markdown(report)
else:
    st.caption("버튼을 눌러 리포트를 생성해보세요.")

# Shareable text
st.markdown("#### 📌 공유용 텍스트")
share_text = f"""[AI 습관 트래커 - 오늘 기록]
- 날짜: {date.today().isoformat()}
- 도시: {city}
- 달성률: {achievement_pct}% ({checked_cnt}/{len(HABITS)})
- 기분: {mood}/10

[습관]
{chr(10).join([f"- {emo} {label}: {'✅' if today_habits.get(k) else '❌'}" for k, emo, label in HABITS])}

[AI 리포트]
{report if report else "(아직 생성 전)"}
"""
st.code(share_text, language="text")

# ---------------------------
# API 안내 (Expander)
# ---------------------------
with st.expander("📎 API 안내 / 문제 해결", expanded=False):
    st.markdown(
        """
**1) OpenAI API Key**
- OpenAI 키가 있어야 '컨디션 리포트'가 생성돼요.
- 사이드바에 입력하거나 환경변수 `OPENAI_API_KEY`로 설정하세요.

**2) OpenWeatherMap API Key**
- 날씨 카드는 OpenWeatherMap 키가 있어야 동작해요.
- 사이드바에 입력하거나 환경변수 `OPENWEATHERMAP_API_KEY`로 설정하세요.

**3) Dog CEO**
- 강아지 이미지는 무료 공개 API라 키가 필요 없어요.
- 네트워크가 불안정하면 실패할 수 있어요(실패 시 None 처리).

**4) 실행**
```bash
streamlit run app.py
