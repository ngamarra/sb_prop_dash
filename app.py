from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

DEFAULT_ANSWER_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1oLm5JYaYeLVT6ipytp2a4OeU5zTQxmyKmwYDdK2JDVo/htmlview"
    "?usp=sharing&pru=AAABnGPKwnA*sUb9cijs8-NaqcgavayiwA"
)
QUESTION_ID_PATTERN = re.compile(r"^Q\d{1,3}$")
YEAR_CONFIGS = {
    "2026": {
        "answer_url": DEFAULT_ANSWER_URL,
        "local_answer_path": "/Users/ngamarra/Downloads/Answers 2026.xlsx",
        "participants_folder": "participants/2026",
    }
}

BOARD_DESC_ALIASES_RAW = {
    "charlie puth": "Q01",
    "heads or tails": "Q02",
    "winner of coin toss": "Q03",
    "1st quarter": "Q04",
    "2nd quarter": "Q05",
    "3rd quarter": "Q06",
    "4th quarter": "Q07",
    "game total": "Q08",
    "touchdowns": "Q09",
    "field goals": "Q10",
    "interceptions": "Q11",
    "fumbles lost": "Q12",
    "sacks": "Q13",
    "punts": "Q14",
    "first downs": "Q15",
    "sam darnold completions": "Q16",
    "kenneth walker iii rushing yards": "Q17",
    "aj barner recieving yards": "Q24",
    "touchdown first": "Q36",
    "field goal first": "Q37",
    "turnover first": "Q38",
    "timeout first": "Q39",
    "punt first": "Q40",
    "sack first": "Q41",
    "first play from scrimmage is a rushing attempt": "Q42",
    "a player gains 10 or more yards on one play": "Q43",
    "team crosses midfield (passes 50 yard line)": "Q44",
    'first song "la mudanza"': "Q45",
    "special guest appearance by cardi b": "Q46",
    "sam darnold completes first 2 pass attempts": "Q47",
    "will the game be tied after 0-0": "Q48",
    "either team scores in first 6.5 minutes of the game": "Q49",
    "3+ total players to have a pass attempt": "Q50",
    "will there be a missed field goal": "Q51",
    "new england commits first accepted penalty": "Q52",
    "special teams or defensive touchdown": "Q53",
    "will seattle have the lead at halftime": "Q54",
    "new england scores first after halftime": "Q55",
    "safety": "Q56",
    "4th down conversion for a touchdown": "Q57",
    "will there be a successful 2 point conversion": "Q58",
    "will either team kick a field goal in 3rd quarter": "Q59",
    "points scored with under 2 minutes left in game": "Q60",
    "will there be a missed extra point kick": "Q61",
    "will there be overtime": "Q62",
    "super bowl 60 champion": "Q63",
    "covers point spread 1.5": "Q64",
    "position of player named mvp": "Q65",
    "color of gatorade shower on head coach": "Q66",
    "final score": "Q67",
}


def clean_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_question_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    match = re.fullmatch(r"Q?\s*0*(\d+)", text)
    if not match:
        return text
    number = int(match.group(1))
    return f"Q{number:02d}"


def is_question_id(value: object) -> bool:
    return bool(QUESTION_ID_PATTERN.fullmatch(str(value)))


def normalize_pick(value: object) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    collapsed = re.sub(r"\s+", " ", raw).upper()
    compact = re.sub(r"[^A-Z0-9/+.-]", "", collapsed)
    aliases = {
        "OVER": "O",
        "UNDER": "U",
        "HEADS": "H",
        "TAILS": "T",
        "TRUE": "Y",
        "FALSE": "N",
        "YES": "Y",
        "NO": "N",
        "SEATTLE": "SEA",
        "SEAHAWKS": "SEA",
        "NEWENGLAND": "NE",
        "PATRIOTS": "NE",
    }
    if compact in {"-", "--", "—", "N/A", "NA", "TBD", "PENDING"}:
        return ""
    return aliases.get(compact, collapsed)


def normalize_description(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    replacements = {
        "recieving": "receiving",
        "1st": "first",
        "2nd": "second",
        "3rd": "third",
        "4th": "fourth",
        " + ": " plus ",
        "&": " and ",
        "?": "",
        '"': "",
        "'": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


BOARD_DESC_ALIASES = {
    normalize_description(k): v for k, v in BOARD_DESC_ALIASES_RAW.items()
}


def pick_column(columns: Iterable[object], preferred: list[str]) -> object | None:
    original = list(columns)
    cleaned = {col: clean_name(col) for col in original}

    for target in preferred:
        exact = [col for col, name in cleaned.items() if name == target]
        if exact:
            return exact[0]
    for target in preferred:
        partial = [col for col, name in cleaned.items() if target in name]
        if partial:
            return partial[0]
    return None


def question_like_ratio(series: pd.Series) -> float:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return 0.0
    matches = values.str.contains(r"^(Q?\s*\d+)$", case=False, regex=True)
    return float(matches.mean())


def detect_question_column(df: pd.DataFrame) -> object | None:
    by_name = pick_column(df.columns, ["questionid", "qid", "propid", "prop"])
    if by_name:
        return by_name

    candidates: list[tuple[float, object]] = []
    for col in df.columns:
        ratio = question_like_ratio(df[col])
        if ratio >= 0.3:
            candidates.append((ratio, col))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def detect_answer_column(df: pd.DataFrame, excluded: set[object]) -> object | None:
    by_name = pick_column(
        [c for c in df.columns if c not in excluded],
        [
            "answer",
            "actual",
            "result",
            "correct",
            "outcome",
            "winningpick",
            "winner",
        ],
    )
    if by_name:
        return by_name

    remaining = [c for c in df.columns if c not in excluded]
    if not remaining:
        return None

    scored: list[tuple[float, int, object]] = []
    for col in remaining:
        values = df[col].dropna().astype(str).str.strip()
        if values.empty:
            continue
        short_ratio = float(values.str.len().le(16).mean())
        non_null = int(values.shape[0])
        scored.append((short_ratio, non_null, col))
    if not scored:
        return remaining[0]

    scored.sort(reverse=True)
    return scored[0][2]


def parse_choice_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    tokens = [t.strip() for t in re.split(r"[|,/]", text) if t.strip()]
    normalized = {normalize_pick(token) for token in tokens}
    return {token for token in normalized if token}


def build_master_lookup(master_df: pd.DataFrame) -> tuple[dict[str, str], list[dict[str, object]]]:
    desc_to_qid: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for row in master_df.itertuples(index=False):
        qid = str(row.question_id)
        desc_norm = normalize_description(getattr(row, "description", ""))
        if desc_norm:
            desc_to_qid[desc_norm] = qid
        rows.append(
            {
                "question_id": qid,
                "desc_norm": desc_norm,
                "allowed": parse_choice_tokens(getattr(row, "choices", "")),
            }
        )
    return desc_to_qid, rows


def match_question_id_from_description(
    text: object, desc_to_qid: dict[str, str], master_rows: list[dict[str, object]]
) -> str | None:
    desc_norm = normalize_description(text)
    if not desc_norm:
        return None
    if desc_norm in BOARD_DESC_ALIASES:
        return BOARD_DESC_ALIASES[desc_norm]
    if desc_norm in desc_to_qid:
        return desc_to_qid[desc_norm]

    contains_hits = [
        row
        for row in master_rows
        if row["desc_norm"]
        and (
            desc_norm in str(row["desc_norm"])
            or str(row["desc_norm"]) in desc_norm
        )
    ]
    if contains_hits:
        contains_hits.sort(key=lambda r: len(str(r["desc_norm"])), reverse=True)
        return str(contains_hits[0]["question_id"])

    best_qid = None
    best_score = 0.0
    for row in master_rows:
        target = str(row["desc_norm"])
        if not target:
            continue
        ratio = SequenceMatcher(None, desc_norm, target).ratio()
        if ratio > best_score:
            best_score = ratio
            best_qid = str(row["question_id"])
    if best_qid and best_score >= 0.86:
        return best_qid
    return None


def extract_answer_from_row(cells: list[str], allowed: set[str], skip_index: int | None = None) -> str:
    values = []
    for idx, cell in enumerate(cells):
        if skip_index is not None and idx == skip_index:
            continue
        values.append(normalize_pick(cell))

    for candidate in values:
        if candidate and candidate in allowed:
            return candidate
    for candidate in values:
        if candidate in {"O", "U", "H", "T", "SEA", "NE", "Y", "N"}:
            if not allowed or candidate in allowed:
                return candidate
    return ""


def to_answer_key_from_board(df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame | None:
    frame = df.copy()
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        return None

    desc_to_qid, master_rows = build_master_lookup(master_df)
    if not master_rows:
        return None
    by_qid = {str(row["question_id"]): row for row in master_rows}

    answers: dict[str, str] = {}
    for _, row in frame.iterrows():
        cells = [str(v).strip() if not pd.isna(v) else "" for v in row.tolist()]
        if not any(cells):
            continue

        qid: str | None = None
        desc_idx: int | None = None
        for idx, cell in enumerate(cells):
            maybe_qid = match_question_id_from_description(cell, desc_to_qid, master_rows)
            if maybe_qid:
                qid = maybe_qid
                desc_idx = idx
                break
        if not qid:
            continue

        allowed = set(by_qid.get(qid, {}).get("allowed", set()))
        parsed = extract_answer_from_row(cells, allowed, skip_index=desc_idx)
        if qid not in answers or parsed:
            answers[qid] = parsed

    if not answers:
        return None
    out = pd.DataFrame({"question_id": list(answers.keys()), "answer": list(answers.values())})
    out["question_id"] = out["question_id"].map(normalize_question_id)
    out["answer"] = out["answer"].map(normalize_pick)
    return out[out["question_id"].map(is_question_id)].drop_duplicates(
        "question_id", keep="last"
    ).reset_index(drop=True)


def to_answer_key(df: pd.DataFrame) -> pd.DataFrame | None:
    frame = df.copy()
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        return None

    q_col = detect_question_column(frame)
    a_col = detect_answer_column(frame, {q_col} if q_col else set())

    if q_col and a_col:
        out = frame[[q_col, a_col]].copy()
        out.columns = ["question_id", "answer"]
    else:
        qid_like_cols = [c for c in frame.columns if is_question_id(normalize_question_id(c))]
        if len(qid_like_cols) < 5:
            return None
        first_row = frame[qid_like_cols].dropna(how="all").head(1)
        if first_row.empty:
            return None
        row = first_row.iloc[0]
        out = pd.DataFrame(
            {
                "question_id": [normalize_question_id(c) for c in qid_like_cols],
                "answer": [row[c] for c in qid_like_cols],
            }
        )

    out["question_id"] = out["question_id"].map(normalize_question_id)
    out["answer"] = out["answer"].map(normalize_pick)
    out = out[out["question_id"].map(is_question_id)].drop_duplicates("question_id", keep="last")
    return out.reset_index(drop=True)


def parse_answer_key_frame(frame: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame | None:
    direct = to_answer_key(frame)
    if direct is not None and not direct.empty:
        return direct
    board = to_answer_key_from_board(frame, master_df)
    if board is not None and not board.empty:
        return board
    return None


def build_google_candidates(url: str) -> list[str]:
    candidates = [url]
    parsed = urlparse(url)
    sheet_match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", parsed.path)
    if not sheet_match:
        return candidates

    sheet_id = sheet_match.group(1)
    query = parse_qs(parsed.query)
    gid = query.get("gid", [None])[0]

    csv_export = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    csv_gviz = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    if gid:
        csv_export += f"&gid={gid}"
        csv_gviz += f"&gid={gid}"
    candidates.extend([csv_export, csv_gviz])
    return candidates


def load_answer_key(
    answer_url: str,
    master_df: pd.DataFrame,
    local_answer_path: str = "",
    uploaded_answer=None,
) -> tuple[pd.DataFrame | None, list[str], str]:
    if uploaded_answer is not None:
        name = uploaded_answer.name.lower()
        try:
            if name.endswith(".xlsx"):
                excel = pd.read_excel(uploaded_answer, sheet_name=None)
                for sheet_name, frame in excel.items():
                    parsed = parse_answer_key_frame(frame, master_df)
                    if parsed is not None and not parsed.empty:
                        return parsed, [], f"Uploaded file ({uploaded_answer.name} / {sheet_name})"
                return None, [f"{uploaded_answer.name}: no answer key detected in any sheet."], ""
            frame = pd.read_csv(uploaded_answer)
            parsed = parse_answer_key_frame(frame, master_df)
            if parsed is not None and not parsed.empty:
                return parsed, [], f"Uploaded file ({uploaded_answer.name})"
            return None, [f"{uploaded_answer.name}: no answer key detected."], ""
        except Exception as exc:
            return None, [f"{uploaded_answer.name}: failed to load ({exc})."], ""

    if local_answer_path.strip():
        path = Path(local_answer_path).expanduser()
        if not path.exists():
            return None, [f"Answer file not found: {path}"], ""
        try:
            if path.suffix.lower() == ".xlsx":
                excel = pd.read_excel(path, sheet_name=None)
                for sheet_name, frame in excel.items():
                    parsed = parse_answer_key_frame(frame, master_df)
                    if parsed is not None and not parsed.empty:
                        return parsed, [], f"Local file ({path} / {sheet_name})"
                return None, [f"{path}: no answer key detected in any sheet."], ""
            frame = pd.read_csv(path)
            parsed = parse_answer_key_frame(frame, master_df)
            if parsed is not None and not parsed.empty:
                return parsed, [], f"Local file ({path})"
            return None, [f"{path}: no answer key detected."], ""
        except Exception as exc:
            return None, [f"{path}: failed to load ({exc})."], ""

    errors: list[str] = []
    for candidate in build_google_candidates(answer_url):
        try:
            csv_df = pd.read_csv(candidate)
            parsed = parse_answer_key_frame(csv_df, master_df)
            if parsed is not None and not parsed.empty:
                return parsed, errors, candidate
            errors.append(f"{candidate}: CSV loaded but no answer key detected.")
        except Exception as exc:
            errors.append(f"{candidate}: CSV load failed ({exc}).")

        try:
            html_tables = pd.read_html(candidate)
            for idx, table in enumerate(html_tables):
                parsed = parse_answer_key_frame(table, master_df)
                if parsed is not None and not parsed.empty:
                    return parsed, errors, f"{candidate} (table {idx + 1})"
            errors.append(f"{candidate}: HTML tables loaded but no answer key detected.")
        except Exception as exc:
            errors.append(f"{candidate}: HTML parse failed ({exc}).")

    return None, errors, ""


def prep_master(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    q_col = detect_question_column(frame)
    if not q_col:
        raise ValueError("Could not find a question ID column in the master file.")

    desc_col = pick_column(frame.columns, ["description", "questiontext", "prompt", "question"])
    choices_col = pick_column(frame.columns, ["choices", "options", "allowed", "selection"])

    prepared = pd.DataFrame({"question_id": frame[q_col].map(normalize_question_id)})
    prepared["description"] = frame[desc_col].astype(str) if desc_col else ""
    prepared["choices"] = frame[choices_col].astype(str) if choices_col else ""

    prepared = prepared[prepared["question_id"].map(is_question_id)].drop_duplicates("question_id")
    if prepared.empty:
        raise ValueError("Master file did not produce any valid question IDs.")
    return prepared.reset_index(drop=True)


def parse_participant_frame(df: pd.DataFrame, fallback_name: str) -> pd.DataFrame:
    frame = df.copy()
    q_col = detect_question_column(frame)
    if not q_col:
        raise ValueError("No question ID column found.")

    pick_col = pick_column(
        [c for c in frame.columns if c != q_col],
        ["pick", "selection", "choice", "prediction", "response", "answer"],
    )
    if not pick_col:
        non_q = [c for c in frame.columns if c != q_col]
        if len(non_q) == 1:
            pick_col = non_q[0]
        else:
            raise ValueError("No pick/selection column found.")

    name_col = pick_column(frame.columns, ["participant", "person", "name", "player", "user"])
    participant_name = fallback_name
    if name_col:
        first = frame[name_col].dropna().astype(str).str.strip()
        if not first.empty:
            participant_name = first.iloc[0]

    out = pd.DataFrame(
        {
            "participant": participant_name,
            "question_id": frame[q_col].map(normalize_question_id),
            "pick": frame[pick_col].map(normalize_pick),
        }
    )
    out = out[out["question_id"].map(is_question_id)].drop_duplicates(
        ["participant", "question_id"], keep="last"
    )
    return out.reset_index(drop=True)


def score_board(
    master_df: pd.DataFrame, answers_df: pd.DataFrame, picks_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    picks_df = picks_df.drop_duplicates(["participant", "question_id"], keep="last")
    questions = master_df[["question_id", "description", "choices"]].copy()
    questions = questions.merge(answers_df[["question_id", "answer"]], on="question_id", how="left")
    questions["answer"] = questions["answer"].fillna("")

    participants = sorted(picks_df["participant"].dropna().astype(str).unique().tolist())
    if not participants:
        return pd.DataFrame(), pd.DataFrame()

    base = pd.MultiIndex.from_product(
        [participants, questions["question_id"].tolist()], names=["participant", "question_id"]
    ).to_frame(index=False)

    scored = base.merge(questions, on="question_id", how="left").merge(
        picks_df[["participant", "question_id", "pick"]],
        on=["participant", "question_id"],
        how="left",
    )
    scored["pick"] = scored["pick"].fillna("")
    scored["is_answered"] = scored["answer"] != ""
    scored["has_pick"] = scored["pick"] != ""

    scored["status"] = "INCORRECT"
    scored.loc[~scored["has_pick"], "status"] = "NO_PICK"
    scored.loc[scored["has_pick"] & ~scored["is_answered"], "status"] = "PENDING"
    scored.loc[
        scored["has_pick"] & scored["is_answered"] & (scored["pick"] == scored["answer"]), "status"
    ] = "CORRECT"

    scored["points"] = (scored["status"] == "CORRECT").astype(int)

    summary = (
        scored.groupby("participant", as_index=False)
        .agg(
            correct=("status", lambda s: int((s == "CORRECT").sum())),
            incorrect=("status", lambda s: int((s == "INCORRECT").sum())),
            pending=("status", lambda s: int((s == "PENDING").sum())),
            no_pick=("status", lambda s: int((s == "NO_PICK").sum())),
            points=("points", "sum"),
            questions=("question_id", "count"),
            answered_questions=("is_answered", "sum"),
            picked_questions=("has_pick", "sum"),
        )
        .sort_values(["points", "correct"], ascending=[False, False])
        .reset_index(drop=True)
    )

    summary["accuracy"] = (
        (summary["correct"] / summary["answered_questions"].replace({0: pd.NA})) * 100
    ).round(1)
    summary["completion"] = (
        (summary["picked_questions"] / summary["questions"].replace({0: pd.NA})) * 100
    ).round(1)
    summary["accuracy"] = summary["accuracy"].fillna(0.0)
    summary["completion"] = summary["completion"].fillna(0.0)
    summary.index = summary.index + 1
    summary.index.name = "rank"

    return summary, scored


def load_participants_from_folder(folder: str) -> tuple[pd.DataFrame, list[str]]:
    path = Path(folder).expanduser()
    if not path.exists():
        return pd.DataFrame(), [f"Folder not found: {path}"]

    rows: list[pd.DataFrame] = []
    issues: list[str] = []
    for csv_file in sorted(path.glob("*.csv")):
        try:
            parsed = parse_participant_frame(pd.read_csv(csv_file), csv_file.stem)
            rows.append(parsed)
        except Exception as exc:
            issues.append(f"{csv_file.name}: {exc}")

    if not rows:
        return pd.DataFrame(), issues
    return pd.concat(rows, ignore_index=True), issues


def load_participants_from_uploads(files: list) -> tuple[pd.DataFrame, list[str]]:
    rows: list[pd.DataFrame] = []
    issues: list[str] = []

    for uploaded in files:
        try:
            parsed = parse_participant_frame(pd.read_csv(uploaded), Path(uploaded.name).stem)
            rows.append(parsed)
        except Exception as exc:
            issues.append(f"{uploaded.name}: {exc}")

    if not rows:
        return pd.DataFrame(), issues
    return pd.concat(rows, ignore_index=True), issues


def make_template_csv(master_df: pd.DataFrame) -> bytes:
    template = master_df[["question_id", "description", "choices"]].copy()
    template.insert(1, "pick", "")
    return template.to_csv(index=False).encode("utf-8")


def render_year_tab(year: str, master_df: pd.DataFrame, auto_refresh_enabled: bool, auto_refresh_seconds: int) -> None:
    defaults = YEAR_CONFIGS.get(year, {})
    st.subheader(f"{year} Answers & Picks")

    answer_source_mode = st.radio(
        "Answer source",
        ["Google Sheet URL", "Local answer file"],
        horizontal=True,
        key=f"{year}_answer_mode",
    )
    answer_url = st.text_input(
        "Answer sheet URL",
        value=defaults.get("answer_url", DEFAULT_ANSWER_URL),
        key=f"{year}_answer_url",
    )
    local_answer_path = st.text_input(
        "Local answer file path",
        value=defaults.get("local_answer_path", ""),
        key=f"{year}_answer_path",
    )
    uploaded_answer = st.file_uploader(
        "Upload answer file (CSV/XLSX)",
        type=["csv", "xlsx"],
        key=f"{year}_answer_upload",
    )

    source_mode = st.radio(
        "Participant source",
        ["Folder", "Uploads"],
        horizontal=True,
        key=f"{year}_source_mode",
    )
    participant_folder = st.text_input(
        "Participant folder",
        value=defaults.get("participants_folder", f"participants/{year}"),
        key=f"{year}_participants_folder",
    )
    uploaded_participants = st.file_uploader(
        "Participant CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key=f"{year}_participant_upload",
    )

    selected_path = ""
    selected_upload = None
    if answer_source_mode == "Local answer file":
        selected_upload = uploaded_answer
        if selected_upload is None:
            selected_path = local_answer_path

    answers_df, answer_errors, answer_source = load_answer_key(
        answer_url=answer_url,
        master_df=master_df,
        local_answer_path=selected_path,
        uploaded_answer=selected_upload,
    )
    if answers_df is None:
        st.error("Could not load an answer key from the provided source.")
        if answer_errors:
            with st.expander("Answer-loader errors"):
                for err in answer_errors:
                    st.write(f"- {err}")
        return

    if source_mode == "Folder":
        picks_df, participant_issues = load_participants_from_folder(participant_folder)
    else:
        picks_df, participant_issues = load_participants_from_uploads(uploaded_participants)

    last_updated_epoch = time.time()
    seconds_until_next = (
        auto_refresh_seconds - (int(last_updated_epoch) % auto_refresh_seconds)
        if auto_refresh_enabled
        else None
    )

    left, right = st.columns([2, 1])
    left.caption(
        f"Last updated: `{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_updated_epoch))}`"
    )
    if auto_refresh_enabled and seconds_until_next is not None:
        right.caption(f"Next refresh in: `{seconds_until_next}s`")
    else:
        right.caption("Next refresh in: `manual`")

    total_questions = len(master_df)
    answered_now = int((answers_df["answer"] != "").sum())
    st.metric("Answered Props", f"{answered_now}/{total_questions}")
    st.caption(f"Answer source used: `{answer_source}`")

    if participant_issues:
        with st.expander("Participant file issues"):
            for issue in participant_issues:
                st.write(f"- {issue}")

    if picks_df.empty:
        st.info("No participant picks loaded yet. Add CSVs in the selected source.")
        return

    summary_df, scored_df = score_board(master_df, answers_df, picks_df)
    if summary_df.empty:
        st.info("No scoreable participant data found.")
        return

    st.subheader("Leaderboard")
    st.dataframe(summary_df, use_container_width=True)
    st.download_button(
        "Download leaderboard CSV",
        data=summary_df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=f"leaderboard_{year}.csv",
        mime="text/csv",
        key=f"{year}_leaderboard_download",
    )

    st.subheader("Visualizations")
    chart_cols = st.columns(2)

    bar_df = summary_df.reset_index()[["participant", "points"]].copy()
    bar_fig = px.bar(
        bar_df,
        x="participant",
        y="points",
        title="Points by Participant",
        color="participant",
    )
    chart_cols[0].plotly_chart(bar_fig, use_container_width=True)

    status_counts = (
        scored_df.groupby(["participant", "status"])
        .size()
        .reset_index(name="count")
        .sort_values("participant")
    )
    status_fig = px.bar(
        status_counts,
        x="participant",
        y="count",
        color="status",
        title="Status Breakdown",
        barmode="stack",
    )
    chart_cols[1].plotly_chart(status_fig, use_container_width=True)

    scatter_df = summary_df.reset_index()[["participant", "accuracy", "completion", "points"]].copy()
    scatter_fig = px.scatter(
        scatter_df,
        x="completion",
        y="accuracy",
        size="points",
        color="participant",
        title="Accuracy vs Completion",
        labels={"completion": "Completion (%)", "accuracy": "Accuracy (%)"},
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Participant Detail")
    selected = st.selectbox("Participant", summary_df["participant"].tolist(), key=f"{year}_participant_select")
    detail = scored_df[scored_df["participant"] == selected].copy()
    only_open = st.checkbox("Only show non-correct rows", value=True, key=f"{year}_only_open")
    if only_open:
        detail = detail[detail["status"] != "CORRECT"]

    detail = detail[["question_id", "description", "choices", "pick", "answer", "status"]]
    detail["_q_num"] = detail["question_id"].str.extract(r"Q(\d+)", expand=False).astype(int)
    detail = detail.sort_values("_q_num").drop(columns="_q_num")
    st.dataframe(detail, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Prop Bet Live Scorer", layout="wide")
    st.title("Prop Bet Live Scorer")
    st.caption("Live scoreboard for participant prop sheets.")

    st.sidebar.header("Data Sources")
    uploaded_master = st.sidebar.file_uploader("Master questions CSV", type=["csv"])
    master_path = st.sidebar.text_input("Or local master CSV path", value="data/props_master.csv")

    st.sidebar.divider()
    auto_refresh_enabled = st.sidebar.checkbox("Auto-refresh", value=True)
    auto_refresh_seconds = int(
        st.sidebar.number_input(
            "Auto-refresh interval (seconds)",
            min_value=5,
            max_value=300,
            value=20,
            step=5,
        )
    )
    if auto_refresh_enabled:
        tick = st_autorefresh(interval=auto_refresh_seconds * 1000, key="live_refresh")
        st.sidebar.caption(f"Live refresh active (tick {tick})")
    refresh = st.sidebar.button("Refresh now")
    if refresh:
        st.rerun()

    try:
        master_raw = pd.read_csv(uploaded_master) if uploaded_master else pd.read_csv(master_path)
        master_df = prep_master(master_raw)
    except Exception as exc:
        st.error(f"Failed to load master CSV: {exc}")
        st.stop()

    st.sidebar.download_button(
        "Download participant template",
        data=make_template_csv(master_df),
        file_name="participant_template.csv",
        mime="text/csv",
    )

    years = list(YEAR_CONFIGS.keys()) or ["2026"]
    tabs = st.tabs(years)
    for tab, year in zip(tabs, years):
        with tab:
            render_year_tab(year, master_df, auto_refresh_enabled, auto_refresh_seconds)


if __name__ == "__main__":
    main()
