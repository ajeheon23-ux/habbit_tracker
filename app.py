import calendar
import os
import sqlite3
from datetime import date, datetime

import pandas as pd
import streamlit as st

DB_PATH = "habit_ai.db"
INTERVIEW_FIELDS = [
    ("last_food", "직전 먹은 음식이 무엇인가요?"),
    ("sleep_hours", "오늘 수면시간은 몇 시간이었나요? (예: 6.5)"),
    ("recent_workout_day", "최근 운동한 날짜/요일은 언제였나요?"),
    ("recent_workout_part", "최근 운동한 부위는 어디였나요?"),
]

st.set_page_config(page_title="Habit AI Coach", page_icon="🏋️", layout="wide")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_logs (
            log_date TEXT PRIMARY KEY,
            last_food TEXT,
            sleep_hours REAL,
            recent_workout_day TEXT,
            recent_workout_part TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            log_date TEXT PRIMARY KEY,
            meal_plan TEXT,
            workout_plan TEXT,
            coach_note TEXT,
            model_name TEXT,
            generated_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_log(log_date: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT log_date, last_food, sleep_hours, recent_workout_day, recent_workout_part
        FROM daily_logs
        WHERE log_date = ?
        """,
        (log_date,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "log_date": row[0],
        "last_food": row[1] or "",
        "sleep_hours": row[2] if row[2] is not None else None,
        "recent_workout_day": row[3] or "",
        "recent_workout_part": row[4] or "",
    }


def upsert_log(record: dict) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO daily_logs (
            log_date, last_food, sleep_hours, recent_workout_day, recent_workout_part, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(log_date) DO UPDATE SET
            last_food=excluded.last_food,
            sleep_hours=excluded.sleep_hours,
            recent_workout_day=excluded.recent_workout_day,
            recent_workout_part=excluded.recent_workout_part,
            updated_at=excluded.updated_at
        """,
        (
            record["log_date"],
            record.get("last_food", ""),
            record.get("sleep_hours", None),
            record.get("recent_workout_day", ""),
            record.get("recent_workout_part", ""),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()


def get_recommendation(log_date: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT meal_plan, workout_plan, coach_note, model_name, generated_at
        FROM recommendations
        WHERE log_date = ?
        """,
        (log_date,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "meal_plan": row[0],
        "workout_plan": row[1],
        "coach_note": row[2],
        "model_name": row[3],
        "generated_at": row[4],
    }


def upsert_recommendation(log_date: str, meal_plan: str, workout_plan: str, coach_note: str, model_name: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO recommendations (log_date, meal_plan, workout_plan, coach_note, model_name, generated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(log_date) DO UPDATE SET
            meal_plan=excluded.meal_plan,
            workout_plan=excluded.workout_plan,
            coach_note=excluded.coach_note,
            model_name=excluded.model_name,
            generated_at=excluded.generated_at
        """,
        (log_date, meal_plan, workout_plan, coach_note, model_name, now),
    )
    conn.commit()
    conn.close()


def get_recent_logs(limit: int = 14) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT log_date, last_food, sleep_hours, recent_workout_day, recent_workout_part
        FROM daily_logs
        ORDER BY log_date DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append(
            {
                "log_date": r[0],
                "last_food": r[1] or "",
                "sleep_hours": r[2] if r[2] is not None else None,
                "recent_workout_day": r[3] or "",
                "recent_workout_part": r[4] or "",
            }
        )
    return out


def get_month_map(year: int, month: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    start = f"{year:04d}-{month:02d}-01"
    end = f"{year:04d}-{month:02d}-31"
    cur.execute(
        """
        SELECT log_date, last_food, sleep_hours, recent_workout_part
        FROM daily_logs
        WHERE log_date BETWEEN ? AND ?
        ORDER BY log_date ASC
        """,
        (start, end),
    )
    rows = cur.fetchall()
    conn.close()

    result = {}
    for row in rows:
        result[row[0]] = {
            "last_food": row[1] or "",
            "sleep_hours": row[2],
            "recent_workout_part": row[3] or "",
        }
    return result


def parse_ai_sections(text: str) -> tuple[str, str, str]:
    meal, workout, note = "", "", ""
    lines = [ln.strip() for ln in text.splitlines()]
    mode = None
    for ln in lines:
        if ln.startswith("식사 코칭"):
            mode = "meal"
            continue
        if ln.startswith("운동 코칭"):
            mode = "workout"
            continue
        if ln.startswith("한 줄 코치"):
            mode = "note"
            continue
        if not ln:
            continue
        if mode == "meal":
            meal += (ln + "\n")
        elif mode == "workout":
            workout += (ln + "\n")
        elif mode == "note":
            note += (ln + "\n")

    return meal.strip(), workout.strip(), note.strip()


def generate_recommendation(openai_api_key: str, model_name: str, today_record: dict, history: list[dict]):
    if not openai_api_key:
        return None, "OpenAI API Key를 입력하세요."

    try:
        from openai import OpenAI

        history_text = "\n".join(
            [
                f"- {h['log_date']} | 음식:{h['last_food']} | 수면:{h['sleep_hours']}h | 최근운동:{h['recent_workout_day']} ({h['recent_workout_part']})"
                for h in history
            ]
        )
        if not history_text:
            history_text = "기록 없음"

        prompt = f"""
사용자는 오늘 아래 상태다.
- 날짜: {today_record['log_date']}
- 직전 음식: {today_record['last_food']}
- 수면시간: {today_record['sleep_hours']}시간
- 최근 운동일: {today_record['recent_workout_day']}
- 최근 운동 부위: {today_record['recent_workout_part']}

최근 기록:
{history_text}

역할:
- 생활 패턴 기반 코치
- 오늘의 식사 방향과 근력운동 방향을 제시

제약:
- 한국어
- 과도한 의료 조언 금지
- 초보자도 실행 가능한 구체적 분량 제공

출력 형식(정확히):
식사 코칭:
- 3~5줄

운동 코칭:
- 준비운동 1줄
- 본운동 3~5개 (세트x반복 또는 시간 포함)
- 마무리 1줄

한 줄 코치:
- 1줄
""".strip()

        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "너는 식사/근력운동 코치다. 과학적으로 무리하지 않는 행동 계획을 제시한다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        )

        text = (resp.choices[0].message.content or "").strip()
        meal, workout, note = parse_ai_sections(text)
        if not meal:
            meal = text
        return {"raw": text, "meal": meal, "workout": workout, "note": note}, None
    except Exception as e:
        return None, f"OpenAI 호출 실패: {e}"


def apply_modern_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(1200px 500px at 5% -10%, #e0f2fe 0%, transparent 60%),
                        radial-gradient(1200px 500px at 100% 0%, #fef9c3 0%, transparent 55%),
                        #f8fafc;
        }
        .hero-card {
            border: 1px solid #e2e8f0;
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(4px);
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }
        .hero-title {font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-bottom: 4px;}
        .hero-sub {color: #334155; font-size: 0.95rem;}
        .calendar-wrap table {width:100%; border-collapse: collapse; table-layout: fixed;}
        .calendar-wrap th {background:#f1f5f9; border:1px solid #dbeafe; padding:8px;}
        .calendar-wrap td {height:80px; border:1px solid #dbeafe; vertical-align: top; padding:6px; background:#ffffffd9;}
        .day {font-weight:700; color:#0f172a; font-size:13px;}
        .chip {margin-top:4px; display:inline-block; padding:2px 6px; border-radius:999px; background:#0ea5e9; color:white; font-size:11px;}
        .chip2 {margin-top:4px; display:inline-block; padding:2px 6px; border-radius:999px; background:#22c55e; color:white; font-size:11px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_month_calendar(year: int, month: int, month_map: dict) -> None:
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdayscalendar(year, month)
    weekdays = ["일", "월", "화", "수", "목", "금", "토"]

    html = ["<div class='calendar-wrap'><table><thead><tr>"]
    for w in weekdays:
        html.append(f"<th>{w}</th>")
    html.append("</tr></thead><tbody>")

    for week in weeks:
        html.append("<tr>")
        for d in week:
            if d == 0:
                html.append("<td></td>")
                continue
            key = f"{year:04d}-{month:02d}-{d:02d}"
            row = month_map.get(key)
            if row:
                sleep = row.get("sleep_hours")
                part = row.get("recent_workout_part", "")
                html.append(
                    f"<td><div class='day'>{d}</div>"
                    f"<div class='chip'>수면 {sleep if sleep is not None else '-'}h</div>"
                    f"<div class='chip2'>{part if part else '운동기록'}</div></td>"
                )
            else:
                html.append(f"<td><div class='day'>{d}</div></td>")
        html.append("</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def init_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "오늘 코칭용 체크인을 시작합니다. 먼저 직전 먹은 음식을 알려주세요.",
            }
        ]
    if "interview_index" not in st.session_state:
        st.session_state.interview_index = 0
    if "draft_record" not in st.session_state:
        st.session_state.draft_record = {
            "last_food": "",
            "sleep_hours": None,
            "recent_workout_day": "",
            "recent_workout_part": "",
        }


def reset_interview_with_date(selected_iso: str) -> None:
    existing = get_log(selected_iso)
    st.session_state.draft_record = {
        "last_food": existing.get("last_food", "") if existing else "",
        "sleep_hours": existing.get("sleep_hours", None) if existing else None,
        "recent_workout_day": existing.get("recent_workout_day", "") if existing else "",
        "recent_workout_part": existing.get("recent_workout_part", "") if existing else "",
    }
    st.session_state.interview_index = 0
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "오늘 코칭용 체크인을 시작합니다. 먼저 직전 먹은 음식을 알려주세요.",
        }
    ]


def handle_user_chat_input(user_text: str) -> None:
    idx = st.session_state.interview_index
    st.session_state.chat_messages.append({"role": "user", "content": user_text})

    if idx < len(INTERVIEW_FIELDS):
        field, _ = INTERVIEW_FIELDS[idx]
        if field == "sleep_hours":
            try:
                st.session_state.draft_record[field] = float(user_text.strip())
            except Exception:
                st.session_state.draft_record[field] = None
        else:
            st.session_state.draft_record[field] = user_text.strip()

        st.session_state.interview_index += 1

    next_idx = st.session_state.interview_index
    if next_idx < len(INTERVIEW_FIELDS):
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": INTERVIEW_FIELDS[next_idx][1]}
        )
    else:
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": "입력이 완료되었습니다. '기록 저장' 버튼을 눌러 캘린더에 저장하고 코칭을 생성하세요.",
            }
        )


init_db()
init_state()
apply_modern_style()

st.markdown(
    """
<div class='hero-card'>
  <div class='hero-title'>Food + Strength Coach AI</div>
  <div class='hero-sub'>생활 패턴을 기록하고, 오늘의 식사 방향성과 근력운동 계획을 자동 코칭합니다.</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("설정")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
    )
    model_name = st.text_input("모델", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    st.caption("권장: OPENAI_API_KEY 환경변수 사용")

selected_date = st.date_input("기록 날짜", value=date.today(), max_value=date.today())
selected_iso = selected_date.isoformat()

if st.button("대화 입력 초기화"):
    reset_interview_with_date(selected_iso)
    st.rerun()

col_chat, col_result = st.columns([1.1, 1], gap="large")

with col_chat:
    st.subheader("💬 체크인 대화창")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("메시지를 입력하세요")
    if user_input:
        handle_user_chat_input(user_input)
        st.rerun()

    st.markdown("#### 현재 입력 값")
    draft = st.session_state.draft_record
    st.write(f"- 직전 음식: {draft.get('last_food') or '-'}")
    st.write(f"- 수면시간: {draft.get('sleep_hours') if draft.get('sleep_hours') is not None else '-'}")
    st.write(f"- 최근 운동일: {draft.get('recent_workout_day') or '-'}")
    st.write(f"- 최근 운동 부위: {draft.get('recent_workout_part') or '-'}")

    if st.button("기록 저장", type="primary"):
        payload = {
            "log_date": selected_iso,
            "last_food": draft.get("last_food", "").strip(),
            "sleep_hours": draft.get("sleep_hours", None),
            "recent_workout_day": draft.get("recent_workout_day", "").strip(),
            "recent_workout_part": draft.get("recent_workout_part", "").strip(),
        }
        upsert_log(payload)
        st.success(f"{selected_iso} 기록 저장 완료")

with col_result:
    st.subheader("🧠 오늘의 AI 코칭")
    current_log = get_log(selected_iso)
    if not current_log:
        st.info("먼저 대화 입력 후 '기록 저장'을 눌러주세요.")
    else:
        if st.button("식사 + 근력운동 코칭 생성", type="primary"):
            recent = get_recent_logs(14)
            result, err = generate_recommendation(
                openai_api_key=openai_api_key,
                model_name=model_name,
                today_record=current_log,
                history=recent,
            )
            if err:
                st.error(err)
            else:
                upsert_recommendation(
                    log_date=selected_iso,
                    meal_plan=result["meal"] or "",
                    workout_plan=result["workout"] or "",
                    coach_note=result["note"] or "",
                    model_name=model_name,
                )
                st.success("AI 코칭 생성 완료")

        rec = get_recommendation(selected_iso)
        if rec:
            st.markdown("**식사 코칭**")
            st.write(rec.get("meal_plan") or "-")
            st.markdown("**운동 코칭**")
            st.write(rec.get("workout_plan") or "-")
            st.markdown("**한 줄 코치**")
            st.write(rec.get("coach_note") or "-")
            st.caption(f"모델: {rec.get('model_name')} / 생성시각: {rec.get('generated_at')}")

st.subheader("📅 캘린더 기록")
cal_col1, cal_col2 = st.columns(2)
with cal_col1:
    cal_year = st.selectbox("연도", options=list(range(date.today().year - 1, date.today().year + 2)), index=1)
with cal_col2:
    cal_month = st.selectbox("월", options=list(range(1, 13)), index=date.today().month - 1)

month_map = get_month_map(cal_year, cal_month)
render_month_calendar(cal_year, cal_month, month_map)

st.subheader("📊 최근 기록")
recent_logs = get_recent_logs(30)
if recent_logs:
    df = pd.DataFrame(recent_logs)
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
    df = df.sort_values("log_date")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.line_chart(df.set_index("log_date")[["sleep_hours"]])
else:
    st.caption("아직 저장된 기록이 없습니다.")

with st.expander("실행 안내"):
    st.markdown(
        """
- 실행: `streamlit run app.py`
- 설치: `pip install -r requirements.txt`
- 저장 방식: 앱 로컬 SQLite(`habit_ai.db`)
"""
    )
