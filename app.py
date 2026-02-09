import calendar
import os
import re
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")
st.title("📊 AI 습관 트래커 (Advanced)")

HABITS = [
    ("wake", "🌅", "기상 미션"),
    ("water", "💧", "물 마시기"),
    ("study", "📚", "공부/독서"),
    ("workout", "🏋️", "운동하기"),
    ("sleep", "😴", "수면"),
]

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Seongnam",
    "Jeju",
]

COACH_STYLES = {
    "스파르타 코치": {
        "system": (
            "너는 엄격하지만 공정한 '스파르타 코치'다. "
            "변명은 줄이고 실행 가능한 행동을 짧고 단호하게 제시한다."
        )
    },
    "따뜻한 멘토": {
        "system": (
            "너는 따뜻하고 공감적인 멘토다. "
            "작은 성취를 인정하며 현실적인 다음 행동을 부드럽게 제안한다."
        )
    },
    "게임 마스터": {
        "system": (
            "너는 RPG 세계관의 게임 마스터다. "
            "하루를 퀘스트와 스탯 관점으로 유쾌하게 해석하고 내일 미션을 제시한다."
        )
    },
}


# ---------------------------
# Helpers
# ---------------------------
def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def iso(d: date) -> str:
    return d.isoformat()


def calc_achievement(habit_dict: dict) -> tuple[int, float]:
    checked = sum(1 for key, _, _ in HABITS if bool(habit_dict.get(key)))
    pct = round((checked / len(HABITS)) * 100, 1)
    return checked, pct


def get_record_by_date(records: list[dict], target_iso: str) -> dict | None:
    return next((r for r in records if r.get("date") == target_iso), None)


def normalize_record(record: dict) -> dict:
    out = {
        "date": record.get("date", iso(date.today())),
        "city": record.get("city", "Seoul"),
        "mood": safe_int(record.get("mood"), 6),
    }
    for key, _, _ in HABITS:
        out[key] = bool(record.get(key))
    return out


def init_sample_data() -> list[dict]:
    base = date.today() - timedelta(days=27)
    demo = []
    for i in range(28):
        d = base + timedelta(days=i)
        weekday = d.weekday()
        rec = {
            "date": iso(d),
            "city": "Seoul",
            "wake": weekday <= 4,
            "water": weekday != 6,
            "study": weekday in [0, 1, 2, 3, 5],
            "workout": weekday in [1, 3, 5],
            "sleep": weekday not in [4],
            "mood": [6, 7, 7, 6, 5, 8, 7][weekday],
        }
        demo.append(rec)
    return demo


def ensure_state():
    if "history" not in st.session_state:
        st.session_state.history = init_sample_data()
    if "last_context" not in st.session_state:
        st.session_state.last_context = None
    if "last_report" not in st.session_state:
        st.session_state.last_report = ""


def upsert_record(record: dict):
    target = record.get("date")
    history = st.session_state.history
    idx = next((i for i, r in enumerate(history) if r.get("date") == target), None)
    if idx is None:
        history.append(record)
    else:
        history[idx] = record
    history = sorted(history, key=lambda r: r.get("date", ""))
    st.session_state.history = history[-365:]


# ---------------------------
# API Layer
# ---------------------------
@st.cache_data(ttl=600)
def get_weather(city: str, api_key: str):
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
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200:
            return None
        data = res.json()
        return {
            "city": city,
            "temp": data.get("main", {}).get("temp"),
            "feels_like": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "desc": (data.get("weather") or [{}])[0].get("description"),
            "icon": (data.get("weather") or [{}])[0].get("icon"),
        }
    except Exception:
        return None


@st.cache_data(ttl=600)
def get_dog_image():
    try:
        res = requests.get("https://dog.ceo/api/breeds/image/random", timeout=10)
        if res.status_code != 200:
            return None
        payload = res.json()
        if payload.get("status") != "success":
            return None
        url = payload.get("message")
        if not url:
            return None
        m = re.search(r"/breeds/([^/]+)/", url)
        breed = (m.group(1).replace("-", " ").strip() if m else "알 수 없음") or "알 수 없음"
        return {"url": url, "breed": breed}
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_quote():
    """ZenQuotes 오늘의 명언."""
    try:
        res = requests.get("https://zenquotes.io/api/today", timeout=10)
        if res.status_code != 200:
            return None
        payload = res.json()
        if not payload or not isinstance(payload, list):
            return None
        item = payload[0]
        return {
            "quote": item.get("q", ""),
            "author": item.get("a", ""),
        }
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_advice():
    """Advice Slip 랜덤 조언."""
    try:
        res = requests.get("https://api.adviceslip.com/advice", timeout=10)
        if res.status_code != 200:
            return None
        payload = res.json()
        advice = (payload.get("slip") or {}).get("advice")
        if not advice:
            return None
        return {"advice": advice}
    except Exception:
        return None


def fetch_context(city: str, owm_key: str) -> dict:
    return {
        "weather": get_weather(city, owm_key),
        "dog": get_dog_image(),
        "quote": get_quote(),
        "advice": get_advice(),
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------
# AI Report
# ---------------------------
def generate_report(openai_api_key: str, coach_style: str, record: dict, context: dict):
    if not openai_api_key:
        return None, "OpenAI API Key가 필요합니다."

    try:
        from openai import OpenAI

        checked, pct = calc_achievement(record)
        habit_lines = [
            f"- {emoji} {label}: {'✅' if record.get(key) else '❌'}"
            for key, emoji, label in HABITS
        ]

        weather = (context or {}).get("weather")
        weather_text = "없음"
        if weather:
            weather_text = (
                f"{weather.get('city')} {weather.get('desc')}, "
                f"{weather.get('temp')}도 (체감 {weather.get('feels_like')}도), 습도 {weather.get('humidity')}%"
            )

        dog = (context or {}).get("dog") or {}
        quote = (context or {}).get("quote") or {}
        advice = (context or {}).get("advice") or {}

        system_prompt = COACH_STYLES.get(coach_style, COACH_STYLES["따뜻한 멘토"])["system"]

        user_prompt = f"""
[체크인 날짜]
{record.get('date')}

[요약]
달성률: {pct}% ({checked}/{len(HABITS)})
기분: {record.get('mood')}/10

[습관 상세]
{chr(10).join(habit_lines)}

[외부 API 컨텍스트]
- 날씨: {weather_text}
- 강아지 품종: {dog.get('breed', '없음')}
- 명언: {quote.get('quote', '없음')} / {quote.get('author', '')}
- 조언: {advice.get('advice', '없음')}

요구사항:
1) 한국어로만 작성
2) 아래 형식을 정확히 지킬 것
3) 컨디션 등급은 S/A/B/C/D 중 하나

형식:
컨디션 등급: <S|A|B|C|D>

핵심 분석:
- 3~5줄

내일 액션:
- 최대 3개 체크리스트

코치 한마디:
- 한 줄
""".strip()

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, None
    except Exception as e:
        return None, f"OpenAI 호출 실패: {e}"


# ---------------------------
# Calendar / Stats
# ---------------------------
def build_history_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).copy()
    for key, _, _ in HABITS:
        if key not in df.columns:
            df[key] = False
    df["checked"] = df.apply(lambda row: sum(bool(row.get(k)) for k, _, _ in HABITS), axis=1)
    df["achievement_pct"] = (df["checked"] / len(HABITS) * 100).round(1)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def pct_to_color(pct: float) -> str:
    if pct >= 80:
        return "#166534"
    if pct >= 60:
        return "#15803d"
    if pct >= 40:
        return "#65a30d"
    if pct >= 20:
        return "#ca8a04"
    return "#b91c1c"


def render_month_calendar(year: int, month: int, records_map: dict):
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdayscalendar(year, month)
    weekdays = ["일", "월", "화", "수", "목", "금", "토"]

    html = [
        """
<style>
.calendar-wrap table {width: 100%; border-collapse: collapse; table-layout: fixed;}
.calendar-wrap th {padding: 8px; border: 1px solid #e5e7eb; background: #f8fafc;}
.calendar-wrap td {height: 86px; vertical-align: top; border: 1px solid #e5e7eb; padding: 6px;}
.calendar-day {font-weight: 700; margin-bottom: 4px;}
.calendar-pill {display: inline-block; padding: 2px 6px; border-radius: 999px; color: white; font-size: 12px;}
.calendar-mood {font-size: 12px; color: #334155; margin-top: 4px;}
</style>
<div class="calendar-wrap">
<table>
<thead><tr>
"""
    ]
    for wd in weekdays:
        html.append(f"<th>{wd}</th>")
    html.append("</tr></thead><tbody>")

    for week in weeks:
        html.append("<tr>")
        for d in week:
            if d == 0:
                html.append("<td></td>")
                continue
            key = f"{year:04d}-{month:02d}-{d:02d}"
            rec = records_map.get(key)
            if rec:
                _, pct = calc_achievement(rec)
                mood = safe_int(rec.get("mood"), 0)
                color = pct_to_color(pct)
                html.append(
                    f"<td><div class='calendar-day'>{d}</div>"
                    f"<span class='calendar-pill' style='background:{color}'>{pct}%</span>"
                    f"<div class='calendar-mood'>기분 {mood}/10</div></td>"
                )
            else:
                html.append(f"<td><div class='calendar-day'>{d}</div></td>")
        html.append("</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ---------------------------
# App start
# ---------------------------
ensure_state()

with st.sidebar:
    st.header("🔑 API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    owm_api_key = st.text_input(
        "OpenWeatherMap API Key",
        type="password",
        value=os.getenv("OPENWEATHERMAP_API_KEY", ""),
    )
    st.caption("환경변수 OPENAI_API_KEY / OPENWEATHERMAP_API_KEY 도 사용 가능합니다.")

st.subheader("✅ 날짜별 체크인")

selected_date = st.date_input("기록할 날짜", value=date.today(), max_value=date.today())
selected_iso = iso(selected_date)

existing = normalize_record(get_record_by_date(st.session_state.history, selected_iso) or {"date": selected_iso})

col_l, col_r = st.columns(2, gap="large")
today_habits = {}
with col_l:
    for key, emoji, label in HABITS[:3]:
        today_habits[key] = st.checkbox(
            f"{emoji} {label}",
            value=bool(existing.get(key)),
            key=f"habit_{selected_iso}_{key}",
        )
with col_r:
    for key, emoji, label in HABITS[3:]:
        today_habits[key] = st.checkbox(
            f"{emoji} {label}",
            value=bool(existing.get(key)),
            key=f"habit_{selected_iso}_{key}",
        )

mood = st.slider(
    "🙂 기분 (1~10)",
    min_value=1,
    max_value=10,
    value=safe_int(existing.get("mood"), 6),
    step=1,
    key=f"mood_{selected_iso}",
)

c1, c2 = st.columns(2, gap="large")
with c1:
    default_city_idx = CITIES.index(existing.get("city")) if existing.get("city") in CITIES else 0
    city = st.selectbox("🌍 도시", CITIES, index=default_city_idx, key=f"city_{selected_iso}")
with c2:
    coach_style = st.radio("🧠 코치 스타일", list(COACH_STYLES.keys()), horizontal=True)

save_col, _ = st.columns([1, 2])
with save_col:
    if st.button("기록 저장 / 수정", type="primary"):
        record = {"date": selected_iso, "city": city, "mood": mood}
        record.update({k: bool(today_habits.get(k)) for k, _, _ in HABITS})
        upsert_record(record)
        st.success(f"{selected_iso} 기록을 저장했습니다.")

checked_cnt, achievement_pct = calc_achievement(today_habits)
m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achievement_pct}%")
m2.metric("달성 습관", f"{checked_cnt}/{len(HABITS)}")
m3.metric("기분", f"{mood}/10")

st.subheader("📅 달력 기반 습관 트래킹")
records_map = {r["date"]: normalize_record(r) for r in st.session_state.history}

left, right = st.columns([1, 1])
with left:
    cal_year = st.selectbox("연도", list(range(date.today().year - 2, date.today().year + 1)), index=2)
with right:
    cal_month = st.selectbox("월", list(range(1, 13)), index=date.today().month - 1)

render_month_calendar(cal_year, cal_month, records_map)

st.subheader("📈 주간/월간 통계")
df = build_history_df(st.session_state.history)
if not df.empty:
    st.line_chart(df.set_index("date")[["achievement_pct", "mood"]])

    today_ts = pd.Timestamp(date.today())
    week_from = today_ts - pd.Timedelta(days=6)
    month_from = today_ts - pd.Timedelta(days=29)
    week_avg = round(df[df["date"] >= week_from]["achievement_pct"].mean(), 1)
    month_avg = round(df[df["date"] >= month_from]["achievement_pct"].mean(), 1)
    best_day = df.loc[df["achievement_pct"].idxmax()]

    s1, s2, s3 = st.columns(3)
    s1.metric("최근 7일 평균", f"{week_avg}%")
    s2.metric("최근 30일 평균", f"{month_avg}%")
    s3.metric("최고 달성일", f"{best_day['date'].date()} ({best_day['achievement_pct']}%)")
else:
    st.info("통계를 표시할 기록이 없습니다.")

st.subheader("🌐 API 허브")
api_btn = st.button("외부 API 데이터 새로고침")
if api_btn:
    with st.spinner("API 데이터를 가져오는 중..."):
        st.session_state.last_context = fetch_context(city, owm_api_key)

context = st.session_state.last_context or {}
weather = context.get("weather")
dog = context.get("dog")
quote = context.get("quote")
advice = context.get("advice")

k1, k2 = st.columns(2)
with k1:
    st.markdown("#### ☁️ 날씨")
    if weather:
        st.write(f"{weather.get('city')} / {weather.get('desc')}")
        st.write(f"{weather.get('temp')}°C (체감 {weather.get('feels_like')}°C), 습도 {weather.get('humidity')}%")
    else:
        st.caption("날씨 데이터 없음 (키/네트워크 확인)")

    st.markdown("#### 💬 명언")
    if quote and quote.get("quote"):
        st.write(f"\"{quote.get('quote')}\"")
        if quote.get("author"):
            st.caption(f"- {quote.get('author')}")
    else:
        st.caption("명언 데이터 없음")

with k2:
    st.markdown("#### 🐶 랜덤 강아지")
    if dog and dog.get("url"):
        st.caption(f"품종(추정): {dog.get('breed')}")
        st.image(dog.get("url"), use_container_width=True)
    else:
        st.caption("강아지 데이터 없음")

    st.markdown("#### 🧠 한 줄 조언")
    if advice and advice.get("advice"):
        st.write(advice.get("advice"))
    else:
        st.caption("조언 데이터 없음")

if context.get("fetched_at"):
    st.caption(f"API 갱신 시각: {context.get('fetched_at')}")

st.subheader("🧾 AI 코치 리포트")
if st.button("선택 날짜 리포트 생성", type="primary"):
    with st.spinner("리포트 생성 중..."):
        if not st.session_state.last_context:
            st.session_state.last_context = fetch_context(city, owm_api_key)

        report_record = {
            "date": selected_iso,
            "city": city,
            "mood": mood,
            **{k: bool(today_habits.get(k)) for k, _, _ in HABITS},
        }
        report, err = generate_report(
            openai_api_key=openai_api_key,
            coach_style=coach_style,
            record=report_record,
            context=st.session_state.last_context,
        )
        if err:
            st.error(err)
        else:
            st.session_state.last_report = report

if st.session_state.last_report:
    st.markdown(st.session_state.last_report)
else:
    st.caption("리포트를 생성하면 여기에 표시됩니다.")

st.subheader("📌 공유 텍스트")
share_text = f"""[AI 습관 트래커]
- 날짜: {selected_iso}
- 도시: {city}
- 달성률: {achievement_pct}% ({checked_cnt}/{len(HABITS)})
- 기분: {mood}/10

[습관]
{chr(10).join([f"- {emoji} {label}: {'✅' if today_habits.get(key) else '❌'}" for key, emoji, label in HABITS])}

[AI 리포트]
{st.session_state.last_report if st.session_state.last_report else '(미생성)'}
"""
st.code(share_text, language="text")

with st.expander("📎 실행 / API 안내"):
    st.markdown(
        """
- 실행: `streamlit run app.py`
- 권장 설치: `pip install -r requirements.txt`
- OpenAI 키: `OPENAI_API_KEY`
- OpenWeatherMap 키: `OPENWEATHERMAP_API_KEY`
- 추가 공개 API: Dog CEO, ZenQuotes, Advice Slip
"""
    )
