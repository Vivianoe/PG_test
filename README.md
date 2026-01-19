# Reading & Class Profile Workflows Backend (with Perusall Integration)

This backend prototype provides:

1. **Reading Scaffold Workflow**  
2. **Class Profile Workflow**  
3. **FastAPI API for interactive workflow execution**  
4. **Perusall integration for posting annotations**

Both workflows use **Gemini** via `langchain-google-genai` and run multi-step orchestration using **LangGraph**.

---

# Project Structure

```
backend/
  .env
  .gitignore
  requirements.txt
  README.md

  main.py                     # FastAPI server (ALL API endpoints)
  workflow.py                 # Reading-scaffold workflow
  profile.py                  # Class profile workflow
  scaffold_reviewer.py        # HITL utilities
  perusall.py                 # Optional standalone Perusall posting script

  prompts/
    material_prompt.py
    focus_prompt.py
    scaffold_prompt.py
    class_profile_prompt.py

  pdf/                        # PDF → JSONL utilities
  reading01_chunks.jsonl      # Example reading chunks
  approved_scaffolds.json     # Example output
```

---

# Environment Setup

Create a `.env` file inside `backend/` with:

```
# Google Gemini API key
GOOGLE_API_KEY=your_google_api_key

# Perusall API authentication
PERUSALL_INSTITUTION=your_x_institution_header
PERUSALL_API_TOKEN=your_x_api_token

# Perusall resource identifiers
PERUSALL_COURSE_ID=your_course_id
PERUSALL_ASSIGNMENT_ID=your_assignment_id
PERUSALL_DOCUMENT_ID=your_document_id
PERUSALL_USER_ID=user_id_you_are_posting_as
```

---

# 1. Class Profile Workflow (`profile.py`)

Generates a 4-paragraph instructional profile from class metadata.

Input example:

```json
{
  "class_id": "class_001",
  "discipline_info": "...",
  "course_info": "...",
  "class_info": "..."
}
```

Run:

```
python profile.py
```

---

# 2. Reading Scaffold Workflow (`workflow.py`)

Processes reading chunks → generates scaffolds → HITL review → export approved.

Run:

```
python workflow.py
```

---

# 3. FastAPI Backend (`main.py`)

Start server:

```
uvicorn main:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

# 4. Perusall Annotation Posting API

FastAPI endpoint:

```
POST /api/perusall/annotations
```

Example request:

```json
{
  "annotations": [
    {
      "rangeType": "text",
      "rangePage": 2,
      "rangeStart": 1159,
      "rangeEnd": 1349,
      "fragment": "As cofounders...",
      "positionStartX": 0.114,
      "positionStartY": 2.488,
      "positionEndX": 0.499,
      "positionEndY": 2.548
    }
  ]
}
```

---

# 5. Standalone Script (`perusall.py`)

Run directly:

```
python perusall.py
```

---

# Local Development Summary

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

---

