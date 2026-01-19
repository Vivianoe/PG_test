import json
import os
import time
import re
from pathlib import Path
from typing import Any, List, Literal, Dict, Optional

from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Prompt imports (from prompts/ folder)
from prompts.material_prompt import get_material_prompt
from prompts.focus_prompt import get_focus_prompt
from prompts.scaffold_prompt import get_scaffold_prompt

# ======================================================
# 0. ENVIRONMENT
# ======================================================

load_dotenv()


# ======================================================
# 1. DATA STRUCTURES (HITL)
# ======================================================

class HistoryEntry(TypedDict, total=False):
    ts: float
    action: Literal["init", "approve", "reject", "manual_edit", "llm_refine"]
    prompt: str
    old_text: str
    new_text: str


class ReviewedScaffold(TypedDict, total=False):
    id: str
    fragment: str
    text: str
    status: Literal["pending", "approved", "rejected"]
    history: List[HistoryEntry]


class WorkflowState(TypedDict, total=False):
    # Inputs
    reading_chunks: Any        # JSON: { "chunks": [ {...}, ... ] }
    class_profile: Any         # JSON: { "class_id", "profile", "design_consideration" }
    reading_info: Any          # JSON: { "assignment_id", "session_description", ... }

    # Intermediate / Outputs
    material_report_text: str                  # free text
    focus_report_json: str                     # JSON string: { "focus_areas": [...] }
    scaffold_json: str                         # JSON string: { "annotation_scaffolds": [...] }

    # HITL review objects (after scaffold)
    annotation_scaffolds_review: List[ReviewedScaffold]

    # Model config (optional)
    model: str
    temperature: float
    max_output_tokens: int


# ======================================================
# 2. UTILS
# ======================================================

def clean_json_output(raw: str) -> str:
    """
    Remove markdown fences like ```json ... ``` or ``` ... ``` and strip whitespace.
    """
    if raw is None:
        return ""
    raw = raw.strip()
    # Remove starting ```json or ``` plus following newline
    raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
    # Remove trailing ```
    raw = re.sub(r"\n```$", "", raw)
    return raw.strip()


def safe_json_loads(raw: str, context: str = "") -> Any:
    """
    Safely parse JSON with helpful error messages.
    """
    cleaned = clean_json_output(raw)
    if not cleaned:
        raise ValueError(f"{context}: JSON string is empty after cleaning.")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        debug_msg = (
            f"\n===== JSONDecodeError in {context} =====\n"
            f"Error: {e}\n"
            f"Cleaned value was:\n{repr(cleaned)}\n"
            f"===== END JSON ERROR ({context}) =====\n"
        )
        print(debug_msg)
        raise ValueError(f"Failed to parse JSON in {context}: {e}") from e


# ======================================================
# 3. LLM CREATOR & INVOCATION HELPER
# ======================================================

def make_llm(state: WorkflowState) -> ChatGoogleGenerativeAI:
    """
    Creates a Gemini 2.5 Flash LLM using values from state,
    with GOOGLE_API_KEY loaded from environment variables.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Did you set it in .env?")

    model_name = state.get("model", "gemini-2.5-flash")
    temperature = state.get("temperature", 0.3)
    max_output_tokens = state.get("max_output_tokens")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
    )


def run_chain(
    llm: ChatGoogleGenerativeAI,
    prompt: ChatPromptTemplate,
    variables: Dict[str, Any],
    context: str,
) -> str:
    """
    Helper to run a prompt|llm|parser chain with unified error handling.
    """
    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke(variables)
    except Exception as e:
        print(f"\n===== ERROR in {context} node =====")
        print(f"Variables: {json.dumps(variables, ensure_ascii=False)[:1000]}...")
        print(f"Error: {e}")
        print("===== END ERROR =====\n")
        raise RuntimeError(f"LLM invocation failed in {context}: {e}") from e

    if not isinstance(result, str):
        raise TypeError(f"{context}: expected string result from chain, got {type(result)}")

    return result


# ======================================================
# 4. NODE 1 — MATERIAL ANALYSIS
# ======================================================

def node_material(state: WorkflowState) -> dict:
    """
    Node 1:
    Input:  reading_chunks, class_profile
    Output: material_report_text (teacher-facing analysis)
    """
    if "class_profile" not in state or "reading_chunks" not in state:
        raise KeyError("node_material requires 'class_profile' and 'reading_chunks' in state.")

    llm = make_llm(state)
    prompt: ChatPromptTemplate = get_material_prompt()

    result = run_chain(
        llm=llm,
        prompt=prompt,
        variables={
            "class_profile": state["class_profile"],
            "reading_chunks": state["reading_chunks"],
        },
        context="node_material",
    )

    return {"material_report_text": result}


# ======================================================
# 5. NODE 2 — FOCUS IDENTIFICATION
# ======================================================

def node_focus(state: WorkflowState) -> dict:
    """
    Node 2:
    Input:  reading_chunks, class_profile, material_report_text, reading_info
    Output: focus_report_json (JSON string describing focus_areas)
    """
    required_keys = ["class_profile", "reading_info", "reading_chunks", "material_report_text"]
    missing = [k for k in required_keys if k not in state]
    if missing:
        raise KeyError(f"node_focus missing required keys in state: {missing}")

    llm = make_llm(state)
    prompt: ChatPromptTemplate = get_focus_prompt()

    result = run_chain(
        llm=llm,
        prompt=prompt,
        variables={
            "class_profile": state["class_profile"],
            "reading_info": state["reading_info"],
            "reading_chunks": state["reading_chunks"],
            "material_report_text": state["material_report_text"],
        },
        context="node_focus",
    )

    return {"focus_report_json": result}


# ======================================================
# 6. NODE 3 — ANNOTATION SCAFFOLD GENERATION
# ======================================================

def node_scaffold(state: WorkflowState) -> dict:
    """
    Node 3:
    Input:  reading_chunks, class_profile, focus_report_json, reading_info
    Output: scaffold_json with format:
        {
          "annotation_scaffolds": [
            {
              "fragment": "...",  # exact text from reading
              "text": "..."       # generated question or prompt
            },
            ...
          ]
        }
    """
    required_keys = ["class_profile", "reading_info", "reading_chunks", "focus_report_json"]
    missing = [k for k in required_keys if k not in state]
    if missing:
        raise KeyError(f"node_scaffold missing required keys in state: {missing}")

    llm = make_llm(state)
    prompt: ChatPromptTemplate = get_scaffold_prompt()

    result = run_chain(
        llm=llm,
        prompt=prompt,
        variables={
            "class_profile": state["class_profile"],
            "reading_info": state["reading_info"],
            "reading_chunks": state["reading_chunks"],
            "focus_report_json": state["focus_report_json"],
        },
        context="node_scaffold",
    )

    return {"scaffold_json": result}


# ======================================================
# 7. NODE 4 — INIT SCAFFOLD REVIEW OBJECTS
# ======================================================

def node_init_scaffold_review(state: WorkflowState) -> dict:
    """
    Take scaffold_json from node_scaffold and wrap each item with:
    - id
    - status = "pending"
    - history = [{ ts, action: "init" }]
    So it's ready for human-in-the-loop review.

    fragment remains unchanged; text may later be edited by humans / LLM.
    """
    raw = state.get("scaffold_json", "")
    scaffold = safe_json_loads(raw, context="node_init_scaffold_review")

    annos = scaffold.get("annotation_scaffolds", [])
    if not isinstance(annos, list):
        raise TypeError(
            "node_init_scaffold_review: 'annotation_scaffolds' must be a list "
            f"but got {type(annos)}"
        )

    reviewed: List[ReviewedScaffold] = []
    now = time.time()

    for idx, item in enumerate(annos):
        if not isinstance(item, dict):
            raise TypeError(
                f"node_init_scaffold_review: each annotation_scaffold must be a dict, got {type(item)}"
            )

        if "fragment" not in item or "text" not in item:
            raise KeyError(
                "Each annotation_scaffold must contain 'fragment' and 'text' keys."
            )

        reviewed.append({
            "id": f"scaf{idx + 1:03d}",
            "fragment": item["fragment"],
            "text": item["text"],
            "status": "pending",
            "history": [
                {
                    "ts": now,
                    "action": "init",
                }
            ],
        })

    return {"annotation_scaffolds_review": reviewed}


# ======================================================
# 8. BUILD THE WORKFLOW GRAPH
# ======================================================

def build_workflow():
    graph = StateGraph(WorkflowState)

    graph.add_node("material", node_material)
    graph.add_node("focus", node_focus)
    graph.add_node("scaffold", node_scaffold)
    graph.add_node("init_scaffold_review", node_init_scaffold_review)

    graph.set_entry_point("material")
    graph.add_edge("material", "focus")
    graph.add_edge("focus", "scaffold")
    graph.add_edge("scaffold", "init_scaffold_review")
    graph.add_edge("init_scaffold_review", END)

    return graph.compile()


# ======================================================
# 9. UTIL: LOAD READING CHUNKS FROM JSONL
# ======================================================

def load_reading_chunks_from_jsonl(path: str) -> dict:
    """
    Load reading01_chunks.jsonl into the required JSON structure:
    {
       "chunks": [ {...}, {...} ]
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reading chunks file not found: {path}")

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON line in {path}: {e}") from e

    return {"chunks": chunks}


# ======================================================
# 10. HITL HELPER FUNCTIONS (approve / reject / edit / llm_refine)
# ======================================================

def _ensure_history(scaffold: ReviewedScaffold) -> None:
    if "history" not in scaffold or scaffold["history"] is None:
        scaffold["history"] = []  # type: ignore[assignment]


def approve_scaffold(scaffold: ReviewedScaffold) -> ReviewedScaffold:
    _ensure_history(scaffold)
    scaffold["status"] = "approved"
    scaffold["history"].append({
        "ts": time.time(),
        "action": "approve",
    })
    return scaffold


def reject_scaffold(scaffold: ReviewedScaffold) -> ReviewedScaffold:
    _ensure_history(scaffold)
    scaffold["status"] = "rejected"
    scaffold["history"].append({
        "ts": time.time(),
        "action": "reject",
    })
    return scaffold


def manual_edit_scaffold(scaffold: ReviewedScaffold, new_text: str) -> ReviewedScaffold:
    """
    Manually edit the scaffold's text (fragment remains unchanged), and record in history.
    Typically: manual_edit -> approve in a separate step.
    """
    if "text" not in scaffold:
        raise KeyError("manual_edit_scaffold: scaffold has no 'text' field.")

    _ensure_history(scaffold)
    old_text = scaffold["text"]
    scaffold["text"] = new_text
    scaffold["history"].append({
        "ts": time.time(),
        "action": "manual_edit",
        "old_text": old_text,
        "new_text": new_text,
    })
    return scaffold


def llm_refine_scaffold(
    scaffold: ReviewedScaffold,
    user_prompt: str,
    llm: ChatGoogleGenerativeAI,
) -> ReviewedScaffold:
    """
    Use the LLM to refine the scaffold text based on teacher instructions.
    fragment remains unchanged.
    """
    if "text" not in scaffold or "fragment" not in scaffold:
        raise KeyError("llm_refine_scaffold: scaffold must have 'fragment' and 'text'.")

    fragment = scaffold["fragment"]
    old_text = scaffold["text"]

    refine_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You refine existing annotation scaffolds for students.\n"
            "Keep the fragment unchanged. Only rewrite the scaffold text.\n"
            "Return ONLY the new scaffold text string, no explanation."
        ),
        (
            "human",
            "Fragment:\n{fragment}\n\n"
            "Current scaffold:\n{old_text}\n\n"
            "Refinement instruction from teacher:\n{user_prompt}\n\n"
            "Rewrite the scaffold according to the instruction."
        ),
    ])

    new_text = run_chain(
        llm=llm,
        prompt=refine_prompt,
        variables={
            "fragment": fragment,
            "old_text": old_text,
            "user_prompt": user_prompt,
        },
        context="llm_refine_scaffold",
    )

    _ensure_history(scaffold)
    scaffold["text"] = new_text
    scaffold["history"].append({
        "ts": time.time(),
        "action": "llm_refine",
        "prompt": user_prompt,
        "old_text": old_text,
        "new_text": new_text,
    })
    return scaffold


# ======================================================
# 11. EXPORT ONLY APPROVED SCAFFOLDS
# ======================================================

def export_approved_scaffolds(review_list: List[ReviewedScaffold]) -> dict:
    """
    Keep only status == 'approved' scaffolds and export final JSON structure
    for student/teaching systems.

    Output structure:
    {
      "annotation_scaffolds": [
        { "id": "...", "fragment": "...", "text": "..." },
        ...
      ]
    }
    """
    approved_items = [
        {
            "id": item["id"],
            "fragment": item["fragment"],
            "text": item["text"],
        }
        for item in review_list
        if item.get("status") == "approved"
    ]

    return {"annotation_scaffolds": approved_items}


# ======================================================
# 12. READING CONFIGURATIONS
# ======================================================

READING_CONFIGS = {
    "reading01": {
        "chunks_file": "reading01_chunks.jsonl",
        "class_profile": {
            "class_id": "class_001",
            "profile": "undergraduate CS class with mixed prior experience.",
            "design_consideration": "CS learners need scaffolded support for technical reading."
        },
        "reading_info": {
            "assignment_id": "reading01",
            "session_description": "Session 3 of unit on version control and tools.",
            "assignment_description": "Students read about differences between distributed and centralized version control.",
            "assignment_objective": "Students can explain key concepts and compare workflows."
        }
    },
    "reading02": {
        "chunks_file": "reading02_chunks.jsonl",
        "class_profile": {
            "class_id": "class_001",
            "profile": "undergraduate biology class with mixed prior knowledge of ecology.",
            "design_consideration": "Students with limited background in ecosystems may need reading scaffolds and real-world examples to understand energy flow."
        },
        "reading_info": {
            "assignment_id": "reading02",
            "session_description": "Session 3 of the unit on ecosystems and energy flow.",
            "assignment_description": "Students read about energy transfer, food chains, and food webs in different ecosystems.",
            "assignment_objective": "Students can describe the flow of energy in an ecosystem, identify producers, consumers, and decomposers, and explain the importance of energy transfer efficiency."
        }
    },
    "Environment - Ecosystems and Energy": {
        "chunks_file": "Environment - Ecosystems and Energy_chunks.jsonl",
        "class_profile": {
            "class_id": "class_001",
            "profile": "undergraduate biology class with mixed prior knowledge of ecology.",
            "design_consideration": "Students with limited background in ecosystems may need reading scaffolds and real-world examples to understand energy flow."
        },
        "reading_info": {
            "assignment_id": "reading_ecosystems",
            "session_description": "Session on ecosystems and energy flow.",
            "assignment_description": "Students read about energy transfer, food chains, and food webs in different ecosystems.",
            "assignment_objective": "Students can describe the flow of energy in an ecosystem, identify producers, consumers, and decomposers, and explain the importance of energy transfer efficiency."
        }
    },
    "Introduction to Algorithms - Introduction": {
        "chunks_file": "Introduction to Algorithms - Introduction_chunks.jsonl",
        "class_profile": {
            "class_id": "class_001",
            "profile": "undergraduate CS class with mixed prior experience.",
            "design_consideration": "CS learners need scaffolded support for technical reading."
        },
        "reading_info": {
            "assignment_id": "reading_algorithms",
            "session_description": "Session on introduction to algorithms.",
            "assignment_description": "Students read about fundamental algorithm concepts and analysis.",
            "assignment_objective": "Students can explain basic algorithm concepts and understand algorithmic thinking."
        }
    }
}


# ======================================================
# 13. BATCH PROCESSING FUNCTION
# ======================================================

def process_single_reading(
    reading_id: str,
    chunks_file: str,
    class_profile: dict,
    reading_info: dict,
    output_dir: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
    save_outputs: bool = True,
) -> dict:
    """
    Process a single reading and generate scaffolds.
    
    Args:
        reading_id: Identifier for this reading
        chunks_file: Path to the JSONL chunks file
        class_profile: Class profile dictionary
        reading_info: Reading info dictionary
        output_dir: Directory to save outputs (default: current directory)
        model: LLM model name
        temperature: LLM temperature
        max_output_tokens: Max output tokens
        save_outputs: Whether to save outputs to files
    
    Returns:
        Final workflow state dictionary
    """
    print(f"\n{'='*60}")
    print(f"Processing: {reading_id}")
    print(f"{'='*60}\n")
    
    # Load reading chunks
    if not os.path.exists(chunks_file):
        # Try in pdf/ directory
        pdf_chunks_file = os.path.join("pdf", chunks_file)
        if os.path.exists(pdf_chunks_file):
            chunks_file = pdf_chunks_file
        else:
            raise FileNotFoundError(f"Reading chunks file not found: {chunks_file}")
    
    reading_chunks = load_reading_chunks_from_jsonl(chunks_file)
    
    # Create initial state
    initial_state: WorkflowState = {
        "reading_chunks": reading_chunks,
        "class_profile": class_profile,
        "reading_info": reading_info,
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    
    # Run workflow
    app = build_workflow()
    final = app.invoke(initial_state)
    
    # Save outputs if requested
    if save_outputs:
        output_dir = output_dir or "."
        os.makedirs(output_dir, exist_ok=True)
        
        # Save material report
        material_file = os.path.join(output_dir, f"{reading_id}_material_report.txt")
        with open(material_file, "w", encoding="utf-8") as f:
            f.write(final["material_report_text"])
        print(f"  -> Saved material report to {material_file}")
        
        # Save focus report
        focus_file = os.path.join(output_dir, f"{reading_id}_focus_report.json")
        with open(focus_file, "w", encoding="utf-8") as f:
            f.write(final["focus_report_json"])
        print(f"  -> Saved focus report to {focus_file}")
        
        # Save scaffold JSON
        scaffold_file = os.path.join(output_dir, f"{reading_id}_scaffolds.json")
        scaffold_data = safe_json_loads(final["scaffold_json"], context="save_scaffold")
        with open(scaffold_file, "w", encoding="utf-8") as f:
            json.dump(scaffold_data, f, ensure_ascii=False, indent=2)
        print(f"  -> Saved scaffolds to {scaffold_file}")
        
        # Save review objects
        review_file = os.path.join(output_dir, f"{reading_id}_scaffolds_review.json")
        with open(review_file, "w", encoding="utf-8") as f:
            json.dump(final["annotation_scaffolds_review"], f, ensure_ascii=False, indent=2)
        print(f"  -> Saved review objects to {review_file}")
    
    return final


def run_batch_demo(reading_ids: Optional[List[str]] = None, output_dir: str = "outputs"):
    """
    Batch process multiple readings and generate scaffolds for each.
    
    Args:
        reading_ids: List of reading IDs to process. If None, processes all configured readings.
        output_dir: Directory to save outputs
    """
    if reading_ids is None:
        reading_ids = list(READING_CONFIGS.keys())
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING: {len(reading_ids)} reading(s)")
    print(f"{'='*60}\n")
    
    results = {}
    
    for reading_id in reading_ids:
        if reading_id not in READING_CONFIGS:
            print(f"Warning: Reading '{reading_id}' not found in configurations. Skipping.")
            continue
        
        config = READING_CONFIGS[reading_id]
        
        try:
            final = process_single_reading(
                reading_id=reading_id,
                chunks_file=config["chunks_file"],
                class_profile=config["class_profile"],
                reading_info=config["reading_info"],
                output_dir=output_dir,
            )
            results[reading_id] = final
            print(f"\n✓ Successfully processed: {reading_id}\n")
        except Exception as e:
            print(f"\n✗ Error processing {reading_id}: {e}\n")
            results[reading_id] = {"error": str(e)}
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Processed: {len([r for r in results.values() if 'error' not in r])}/{len(reading_ids)}")
    print(f"Outputs saved to: {output_dir}/")
    print(f"{'='*60}\n")
    
    return results


# ======================================================
# 14. SINGLE READING DEMO (ORIGINAL)
# ======================================================

def run_demo(reading_id: str = "reading01"):
    """
    Run demo for a single reading (original functionality).
    
    Args:
        reading_id: ID of the reading to process (default: "reading01")
    """
    if reading_id not in READING_CONFIGS:
        raise ValueError(f"Reading '{reading_id}' not found in configurations. Available: {list(READING_CONFIGS.keys())}")
    
    config = READING_CONFIGS[reading_id]
    
    # Load reading chunks
    chunks_file = config["chunks_file"]
    if not os.path.exists(chunks_file):
        # Try in pdf/ directory
        pdf_chunks_file = os.path.join("pdf", chunks_file)
        if os.path.exists(pdf_chunks_file):
            chunks_file = pdf_chunks_file
        else:
            raise FileNotFoundError(f"Reading chunks file not found: {chunks_file}")
    
    reading_chunks = load_reading_chunks_from_jsonl(chunks_file)
    
    initial_state: WorkflowState = {
        "reading_chunks": reading_chunks,
        "class_profile": config["class_profile"],
        "reading_info": config["reading_info"],
        "model": "gemini-2.5-flash",
        "temperature": 0.3,
        "max_output_tokens": 8192,
    }

    app = build_workflow()
    final = app.invoke(initial_state)

    print("\n=== MATERIAL REPORT ===\n")
    print(final["material_report_text"])

    print("\n=== FOCUS REPORT JSON ===\n")
    print(final["focus_report_json"])

    print("\n=== RAW SCAFFOLD JSON ===\n")
    print(final["scaffold_json"])

    print("\n=== REVIEW OBJECTS (HITL) ===\n")
    for item in final["annotation_scaffolds_review"]:
        print(json.dumps(item, ensure_ascii=False, indent=2))

    # Example: mark first two as approved
    reviewed = final["annotation_scaffolds_review"]
    if reviewed:
        approve_scaffold(reviewed[0])
    if len(reviewed) > 1:
        approve_scaffold(reviewed[1])

    approved_json = export_approved_scaffolds(reviewed)

    print("\n=== FINAL APPROVED ANNOTATION_SCAFFOLDS JSON ===\n")
    print(json.dumps(approved_json, ensure_ascii=False, indent=2))

'''   # Example class profile
    class_profile = {
    "class_id": "class_001",
    "profile": "undergraduate biology class with mixed prior knowledge of ecology.",
    "design_consideration": "Students with limited background in ecosystems may need reading scaffolds and real-world examples to understand energy flow."
    }

# Example reading info
    reading_info = {
    "assignment_id": "reading02",
    "session_description": "Session 3 of the unit on ecosystems and energy flow.",
    "assignment_description": "Students read about energy transfer, food chains, and food webs in different ecosystems.",
    "assignment_objective": "Students can describe the flow of energy in an ecosystem, identify producers, consumers, and decomposers, and explain the importance of energy transfer efficiency."
    }
'''

    


    


if __name__ == "__main__":
    import sys
    
    # Check if batch mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Batch process all readings
        reading_ids = sys.argv[2:] if len(sys.argv) > 2 else None
        try:
            run_batch_demo(reading_ids=reading_ids)
        except Exception as e:
            print("\n=== UNHANDLED ERROR ===")
            print(repr(e))
            print("=== END ERROR ===\n")
    else:
        # Single reading demo
        reading_id = sys.argv[1] if len(sys.argv) > 1 else "reading01"
        try:
            run_demo(reading_id=reading_id)
        except Exception as e:
            print("\n=== UNHANDLED ERROR ===")
            print(repr(e))
            print("=== END ERROR ===\n")
