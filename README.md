# Interview Evidence Pipeline

Turns a folder of interview-note `.txt` files into a themed evidence matrix, proves every excerpt word-for-word against the source files, and answers questions only when the notes contain the answer.

**Run it:** `pip install -r requirements.txt && streamlit run app.py`, then upload the interview files (or put them in `./data`), add a Gemini API key, and click **Extract themes**.

## How it works

| Step | Who does it | What happens |
| --- | --- | --- |
| M0 Ingest | Python | Each bullet becomes an excerpt with an ID tied to its real line number (`I2-L17` = file 2, line 17), the speaker from the header line (or the file name), its section, and a type: `direct_speech` if it contains quoted words, otherwise `interviewer_note`. |
| M1 Themes | Gemini, one call | Gets the numbered excerpts and returns every major theme. For each piece of evidence it returns the ID **and** the exact text it is quoting. |
| M2 Verifier | Python | Checks the text the model returned against the raw file **and** the cited line. Statuses: `VERIFIED`, `WRONG_LINE`, `NORMALISED_ONLY` (matches only after quote/whitespace normalisation; never counted as verified), `NOT_FOUND`, `BAD_ID`, `TOO_SHORT` (a fragment under 3 words cut from a longer note; a short note quoted whole is fine). Only `VERIFIED` excerpts reach the matrix. |
| M3 Q&A | Gemini + Python | Single-shot question. If the model says the notes don't answer it, or none of its citations verify, the app prints **"Not in the interviews."** |

Design choices: no vector DB (the notes fit in one prompt), no chat history, no inference beyond the text, temperature 0. The verifier never trusts the model: it is tested with a one-character tamper and a wrong-line citation (`tests/test_core.py`, and the **Run tamper test** button in the app).

## Files

- `core.py`: ingest, verifier, prompts and LLM calls (no Streamlit).
- `app.py`: Streamlit UI with three tabs: Evidence matrix (CSV export), Verification report, Ask the interviews.
- `tests/test_core.py`: run with `pytest`. Needs no API key or case data.

## Configuration

Set these in `.streamlit/secrets.toml` (see the `.example`) or as environment variables: `GEMINI_API_KEY`, optional `GEMINI_MODEL` (default `gemini-3.8-flash`), optional `APP_PASSWORD` to gate a hosted link.

After each run the themes are cached to `last_run.json`. **Load cached run** reuses them without an API call, and verification always re-runs live against the current files.

## Data

Interview files are confidential and are **not** in this repo. `data/` and `last_run.json` are git-ignored.
