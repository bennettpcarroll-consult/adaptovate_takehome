"""Streamlit front end for the verifiable interview evidence pipeline.

Run locally:  pip install -r requirements.txt && streamlit run app.py
"""

import csv
import io
import os

import streamlit as st

import core

st.set_page_config(page_title="Interview Evidence Pipeline", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CACHE_PATH = os.path.join(HERE, "last_run.json")
PRESETS = [
    "What do grantees want from the Foundation?",
    "What did the change vendor conclude about the organization's structure?",
]


def secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file
        pass
    return os.environ.get(name, default)


# --- Password gate (only when APP_PASSWORD is configured) -------------------
password = secret("APP_PASSWORD")
if password and not st.session_state.get("authed"):
    st.title("Interview Evidence Pipeline")
    if st.text_input("Password", type="password") == password:
        st.session_state.authed = True
        st.rerun()
    st.stop()

model = secret("GEMINI_MODEL", core.DEFAULT_MODEL)
api_key = secret("GEMINI_API_KEY")

# --- Sidebar: inputs ---------------------------------------------------------
with st.sidebar:
    st.header("1 · Interview files")
    uploads = st.file_uploader("Upload interview .txt files", type=["txt"], accept_multiple_files=True)
    sample_names = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".txt")) if os.path.isdir(DATA_DIR) else []
    use_sample = bool(sample_names) and st.checkbox(f"Use sample files in ./data ({len(sample_names)})", value=not uploads)

    if not api_key:
        api_key = st.text_input("Gemini API key", type="password", help="Or set GEMINI_API_KEY in secrets / env")

    st.header("2 · Run")
    run = st.button("Extract themes", type="primary", width='stretch')
    load_cache = os.path.exists(CACHE_PATH) and st.button("Load cached run", width='stretch')
    st.caption(f"Model: {model}")

files: dict[str, str] = {}
if uploads:
    files.update({u.name: u.getvalue().decode("utf-8") for u in uploads})
if use_sample:
    for name in sample_names:
        with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
            files[name] = f.read()

st.title("Interview Evidence Pipeline")
st.caption("Every excerpt shown is checked character-for-character against the source files. Themes are the model's judgment; quotes are proven.")

if not files:
    st.info("Upload interview .txt files (or use the sample files) to begin.")
    st.stop()

corpus = core.ingest(files)
n_direct = sum(e.type == "direct_speech" for e in corpus.catalog.values())
cols = st.columns(4)
cols[0].metric("Files", len(corpus.raw))
cols[1].metric("Excerpts indexed", len(corpus.catalog))
cols[2].metric("Direct speech", n_direct)
cols[3].metric("Interviewer notes", len(corpus.catalog) - n_direct)
for w in corpus.warnings:
    st.warning(w)

# Changing the files invalidates any earlier result.
fingerprint = tuple(sorted((k, hash(v)) for k, v in files.items()))
if st.session_state.get("fingerprint") != fingerprint:
    st.session_state.fingerprint = fingerprint
    st.session_state.pop("themes", None)

if run:
    if not api_key:
        st.error("Add a Gemini API key first.")
    else:
        with st.spinner("Mapping themes…"):
            try:
                themes = core.extract_themes(corpus, api_key, model)
                core.save_run(CACHE_PATH, themes, model)
                st.session_state.themes = themes
            except Exception as e:  # surface API errors plainly
                st.error(f"Theme extraction failed: {e}")
if load_cache:
    st.session_state.themes = core.load_run(CACHE_PATH)
    st.toast("Loaded cached themes; verification re-ran live against the current files.")

themes = st.session_state.get("themes")
verified = core.verify_themes(themes, corpus) if themes else []
all_checks = [c for _, checks in verified for c in checks]

tab_matrix, tab_report, tab_ask = st.tabs(["Evidence matrix", "Verification report", "Ask the interviews"])

# --- Evidence matrix -----------------------------------------------------------
with tab_matrix:
    if not themes:
        st.info("Run **Extract themes** (or load the cached run) to build the matrix.")
    else:
        order = sorted(verified, key=lambda tv: -len({c.speaker for c in tv[1] if c.ok}))
        for theme, checks in order:
            ok = [(ev, c) for ev, c in zip(theme.evidence, checks) if c.ok]
            speakers = sorted({c.speaker for _, c in ok})
            failed = len(checks) - len(ok)
            label = f"**{theme.name}** · {len(ok)} excerpts · {len(speakers)} speakers"
            if failed:
                label += f" · {failed} failed verification"
            with st.expander(label):
                st.write(theme.one_line_summary)
                st.dataframe(
                    [
                        {
                            "Speaker": c.speaker,
                            "Excerpt": c.quote_text,
                            "Type": "Direct speech" if c.type == "direct_speech" else "Interviewer note",
                            "Stance": ev.stance,
                            "Source": f"{c.file}:{c.line}",
                        }
                        for ev, c in ok
                    ],
                    width='stretch',
                    hide_index=True,
                    column_config={"Excerpt": st.column_config.TextColumn(width="large")},
                )

        rows = core.matrix_rows(verified, corpus)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else ["theme"])
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Download matrix (CSV)", buf.getvalue(), "evidence_matrix.csv", "text/csv")

# --- Verification report -----------------------------------------------------
with tab_report:
    if not themes:
        st.info("Nothing to verify yet.")
    else:
        summary = core.summarise(all_checks)
        (st.success if summary["failures"] == 0 else st.warning)(core.banner_text(summary))

        st.subheader("By status")
        st.dataframe([{"Status": k, "Count": v} for k, v in summary["by_status"].items()], hide_index=True)

        st.subheader("By theme")
        st.dataframe(
            [
                {"Theme": t.name, "Extracted": len(cs), "Verified": sum(c.ok for c in cs), "Failures": sum(not c.ok for c in cs)}
                for t, cs in verified
            ],
            hide_index=True,
            width='stretch',
        )

        failures = [c for c in all_checks if not c.ok]
        if failures:
            st.subheader("Failures")
            for c in failures:
                src = corpus.catalog.get(c.quote_id)
                st.markdown(f"**{c.status}** · `{c.quote_id}`")
                st.text(f"Model returned: {c.quote_text}\nSource line:    {src.text if src else '(no such id)'}")
                if c.found_in:
                    st.caption(f"Text actually found at: {', '.join(c.found_in)}")

        st.subheader("Tamper self-test")
        st.caption("Proves the verifier can fail: alter one character of a verified excerpt, and cite a verified excerpt against the wrong line.")
        if st.button("Run tamper test"):
            sample = next((c for c in all_checks if c.ok), None)
            if sample is None:
                st.error("No verified excerpt to tamper with.")
            else:
                altered = core.tamper(sample.quote_text)
                t1 = core.verify(sample.quote_id, altered, corpus)
                other = next(i for i in corpus.catalog if i != sample.quote_id and sample.quote_text not in corpus.catalog[i].text)
                t2 = core.verify(other, sample.quote_text, corpus)
                st.text(f"Original ({sample.quote_id}): {sample.quote_text}\nAltered:            {altered}\n→ {t1.status}")
                st.text(f"Same text cited as {other} → {t2.status}")
                (st.success if not t1.ok and not t2.ok else st.error)(
                    "Verifier rejected both tampered citations." if not t1.ok and not t2.ok else "Verifier accepted a tampered citation."
                )

# --- Ask the interviews --------------------------------------------------------
with tab_ask:
    preset_cols = st.columns(len(PRESETS))
    for col, q in zip(preset_cols, PRESETS):
        if col.button(q, width='stretch'):
            st.session_state.question = q
    question = st.text_input("Ask a question about the interviews", key="question")
    if question:
        if not api_key:
            st.error("Add a Gemini API key first.")
        else:
            cache = st.session_state.setdefault("answers", {})
            key = (fingerprint, question)
            answer = cache.get(key)
            if answer is None:
                with st.spinner("Searching the interviews…"):
                    try:
                        answer = cache[key] = core.resolve_answer(core.ask(question, corpus, api_key, model), corpus)
                    except Exception as e:
                        st.error(f"Question failed: {e}")
            if answer:
                if not answer.answered:
                    st.error(answer.text)
                    if answer.note:
                        st.caption(answer.note)
                else:
                    st.markdown(answer.text)
                    for c in answer.kept:
                        kind = "direct speech" if c.type == "direct_speech" else "interviewer note"
                        st.markdown(f"> {c.quote_text}\n\n— {c.speaker} · {kind} · `{c.file}:{c.line}` ✅")
                if answer.dropped:
                    with st.expander(f"{len(answer.dropped)} citation(s) dropped by the verifier"):
                        for c in answer.dropped:
                            st.text(f"{c.status} · {c.quote_id}: {c.quote_text}")
