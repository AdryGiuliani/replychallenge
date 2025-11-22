from __future__ import annotations

import os
import re
import time
import json
from datetime import datetime
from typing import Literal, TypedDict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


load_dotenv()


class GlucoseState(TypedDict, total=False):
    reading: float
    history: List[float]
    baseline: float
    severity: Literal["normal", "low", "medium", "high"]
    # baseline read from the medical record (may coincide with baseline)
    baseline_from_record: float
    # last relevant log lines for the care agent
    recent_logs: List[str]
    # conversation history between care agent and user (for multi-turn reasoning)
    care_history: List[str]


# --- Baseline agent with LLM: reads baseline from medicalrecord.txt --------

MEDICAL_RECORD_FILE = "medicalrecord.txt"
DEFAULT_BASELINE = 100.0
LOG_FILE = "glucose_log.txt"


def _extract_number_with_llm(text: str) -> Optional[float]:
    """Use an LLM to extract the baseline value from a medical record.

    The model must return ONLY a number (int or float), with no extra text.
    In case of problems, returns None.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[BASELINE-LLM] No OPENAI_API_KEY found in environment; skipping LLM.")
        return None

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    system_prompt = (
        "You are a medical assistant. You are given the text of a medical record. "
        "You MUST RETURN ONLY the numeric value of the patient's baseline blood glucose in mg/dL. "
        "If you are not sure, infer a reasonable baseline value from the text. "
        "Reply ONLY with a number (e.g., 95 or 100.0) with no units or explanations."
    )

    user_prompt = (
        "Medical record text:\n\n" + text + "\n\n" +
        "Extract the baseline blood glucose (mg/dL) and return only the number."
    )

    try:
        resp = llm.invoke([
            ("system", system_prompt),
            ("user", user_prompt),
        ])
    except Exception as e:  # network, quota, etc.
        print(f"[BASELINE-LLM] Error calling LLM: {e}.")
        return None

    content = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

    # Keep ONLY the first number in the LLM response.
    m = re.search(r"(\d+(?:\.\d+)?)", content)
    if not m:
        print(f"[BASELINE-LLM] No number found in LLM response: {content!r}")
        return None

    try:
        value = float(m.group(1))
        return value
    except ValueError:
        print(f"[BASELINE-LLM] Could not convert to float: {content!r}")
        return None


def baseline_agent(state: GlucoseState) -> GlucoseState:
    """Read baseline glucose from medicalrecord.txt using an LLM.

    - First try with an LLM (key read from environment).
    - If anything fails (missing key, API error, invalid answer), use DEFAULT_BASELINE.
    - Store the value both in baseline_from_record and baseline (if not already set).
    """

    if "baseline_from_record" in state:
        print(f"[BASELINE] Baseline already present from record: {state['baseline_from_record']} mg/dL")
        return state

    try:
        with open(MEDICAL_RECORD_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"[BASELINE] WARNING: file '{MEDICAL_RECORD_FILE}' not found. Using default baseline {DEFAULT_BASELINE} mg/dL.")
        baseline_value = DEFAULT_BASELINE
    else:
        # 1) try with LLM
        llm_value = _extract_number_with_llm(text)
        if llm_value is not None:
            baseline_value = llm_value
            print(f"[BASELINE] Baseline extracted from medical record via LLM: {baseline_value} mg/dL")
        else:
            print(f"[BASELINE] LLM did not provide a valid value. Using default baseline {DEFAULT_BASELINE} mg/dL.")
            baseline_value = DEFAULT_BASELINE

    new_state: GlucoseState = {
        **state,
        "baseline_from_record": baseline_value,
    }

    if "baseline" not in new_state:
        new_state["baseline"] = baseline_value

    return new_state


# --- Mock sensor node -----------------------------------------------------

PREDEFINED_READINGS = [80, 85, 90, 110, 150, 200, 75, 95]


def sensor_node(state: GlucoseState) -> GlucoseState:
    """Return next reading from predefined list every call.

    If we exhaust the list, we loop from the start (for a simple demo).
    """
    history = state.get("history", [])
    idx = len(history) % len(PREDEFINED_READINGS)
    reading = float(PREDEFINED_READINGS[idx])

    new_history = history + [reading]

    print(f"[SENSOR] New glucose reading: {reading} mg/dL (idx={idx})")
    # For the demo we keep 5 seconds instead of 10
    time.sleep(5)

    return {
        **state,
        "reading": reading,
        "history": new_history,
    }


# --- Mock RNN node --------------------------------------------------------


def _classify_severity(reading: float, baseline: float) -> Literal["normal", "low", "medium", "high"]:
    """Very simple rule-based classifier standing in for an RNN.

    Uses the percentage difference from baseline to decide severity.
    """
    if baseline <= 0:
        return "normal"

    delta = (reading - baseline) / baseline

    if abs(delta) < 0.1:
        return "normal"
    if delta <= -0.1:
        return "low"
    if 0.1 <= delta < 0.3:
        return "medium"
    return "high"


def rnn_node(state: GlucoseState) -> GlucoseState:
    reading = state["reading"]
    baseline = state["baseline"]
    severity = _classify_severity(reading, baseline)
    print(f"[RNN] Reading {reading} vs baseline {baseline} -> severity: {severity}")
    return {**state, "severity": severity}


# --- Logger node ----------------------------------------------------------


def _read_last_logs(n: int = 5) -> List[str]:
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    return [line.strip() for line in lines[-n:]]


def logger_node(state: GlucoseState) -> GlucoseState:
    timestamp = datetime.utcnow().isoformat()
    baseline_info = state.get("baseline_from_record", state.get("baseline"))
    line = f"{timestamp}\treading={state['reading']}\tseverity={state['severity']}\tbaseline={baseline_info}"

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(f"[LOGGER] {line}")

    # update recent_logs in state so the care agent can use them
    recent = state.get("recent_logs", [])
    recent = (recent + [line])[-5:]

    return {
        **state,
        "recent_logs": recent,
    }


# --- Router based on severity --------------------------------------------

ROUTER_SLEEP_LOW = 5  # seconds to wait when severity is low


def severity_router(state: GlucoseState) -> Literal["back_to_sensor", "care_agent", "end"]:
    """Decide the next step based on detected severity.

    - normal / low   -> go back to sensor (with a short wait if low)
    - medium / high  -> go to care_agent
    """
    severity = state.get("severity", "normal")

    if severity in ("normal", "low"):
        if severity == "low":
            print(f"[ROUTER] LOW severity: only log and wait {ROUTER_SLEEP_LOW}s before going back to sensor.")
            time.sleep(ROUTER_SLEEP_LOW)
        else:
            print("[ROUTER] NORMAL severity: continue monitoring.")
        return "back_to_sensor"

    # medium or high
    print(f"[ROUTER] {severity.upper()} severity: passing to care agent.")
    return "care_agent"


def router_node(state: GlucoseState) -> GlucoseState:
    """Router node that simply forwards the state.

    The actual routing decision is done by `severity_router` via add_conditional_edges.
    """
    return state


# --- Care agent LLM with mock tools ---------------------------------------


def getClinicalInformations() -> str:
    """Mock tool: return user's clinical information.

    In a real system it would read from an EHR; here we return a static string.
    """
    return "Adult patient with type-2 diabetes, no known kidney disease, BMI 28."


def notifyuser(severity: str, message: str) -> str:
    """Mock tool: notify the user and return a textual response.

    In a real system this would be a push notification / SMS; here we log and ask for keyboard input.
    """
    print(f"[TOOL notifyuser] (severity={severity}) -> {message}")
    response = input("User response (simulated): ")
    print(f"[TOOL notifyuser] user_response={response!r}")
    return response


def emergencyCall(message: str) -> str:
    """Mock tool: start an emergency call until the user cancels it.

    Here we only simulate the behavior without blocking indefinitely.
    """
    print(f"[TOOL emergencyCall] {message}")
    response = "simulated call, cancelled by user"
    print(f"[TOOL emergencyCall] user_response={response!r}")
    return response


def notifyDoctor(message: str) -> None:
    """Mock tool: send a report to the preferred doctor (printed as mock)."""
    print(f"[TOOL notifyDoctor] Sending report to doctor:\n{message}")


def care_agent(state: GlucoseState) -> GlucoseState:
    """LLM care agent that decides clinical actions for medium/high severity.

    Uses mock tools: getClinicalInformations, notifyuser, emergencyCall, notifyDoctor.
    The agent can reason over multiple turns: each time it gets user feedback, it
    can plan the next action based on updated context until no further actions
    are needed.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[CARE] No OPENAI_API_KEY found; skipping LLM and ending care flow.")
        return state

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    severity = state.get("severity", "medium")
    reading = state.get("reading")
    baseline = state.get("baseline")
    recent_logs = state.get("recent_logs", [])

    clinical_info = getClinicalInformations()

    # Retrieve previous care-history (if any) so the agent can build on past turns
    care_history = state.get("care_history", [])

    # System prompt: explain tools and required JSON format
    system_prompt = (
        "You are a digital medical assistant that supports blood glucose management. "
        "You have access to these tools:\n"
        "1. getClinicalInformations() -> returns clinical information about the user.\n"
        "2. notifyuser(severity, message) -> sends a message to the user and receives a short reply.\n"
        "3. emergencyCall(message) -> starts an emergency request and waits for confirmation/cancellation.\n"
        "4. notifyDoctor(message) -> sends a textual report to the treating doctor.\n"
        "You MUST decide which tools to use based on severity, context and previous user responses.\n"
        "You MUST reply ONLY with a valid JSON of the form:\n"
        "{\n"
        "  \"actions\": [\n"
        "    {\"tool\": \"notifyuser\" | \"emergencyCall\" | \"notifyDoctor\", \"message\": \"...\"},\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "If no further actions are needed, return {\"actions\": []}.\n"
        "Do not add any text outside the JSON."
    )

    # Summary of recent logs
    logs_text = "\n".join(recent_logs) if recent_logs else "(no recent logs)"

    # Main multi-turn loop: keep planning actions until the LLM returns no actions
    executed_steps: List[str] = []

    while True:
        history_text = "\n".join(care_history) if care_history else "(no previous care-agent conversation)"

        user_prompt = (
            f"Current severity: {severity}\n"
            f"Current reading: {reading} mg/dL\n"
            f"Estimated baseline: {baseline} mg/dL\n"
            f"Clinical information: {clinical_info}\n"
            f"Recent logs (max 5):\n{logs_text}\n\n"
            f"Previous care-agent conversation (most recent last):\n{history_text}\n\n"
            "Decide which actions to take next.\n"
            "- For MEDIUM severity: typically ask the user if they have eaten recently and suggest small adjustments.\n"
            "- For HIGH severity: consider suggesting an emergency call and notifying the doctor.\n"
            "Use the tools by returning JSON actions only. If you think no further actions are needed, return an empty 'actions' array."
        )

        try:
            resp = llm.invoke([
                ("system", system_prompt),
                ("user", user_prompt),
            ])
        except Exception as e:
            print(f"[CARE] Error calling care agent LLM: {e}.")
            break

        content = resp.content if hasattr(resp, "content") else str(resp)
        print("[CARE] Raw LLM output (before JSON parse):\n" + content)

        actions: List[dict] = []
        try:
            data = json.loads(content)
            if isinstance(data, dict) and isinstance(data.get("actions"), list):
                actions = data["actions"]
            else:
                print("[CARE] JSON not in expected format; using fallback behavior.")
        except json.JSONDecodeError as e:
            print(f"[CARE] Error parsing JSON from care agent: {e}. Using fallback behavior.")

        # If no actions, we stop the care loop
        if not actions:
            print("[CARE] No further actions returned by LLM; ending care interaction.")
            break

        # Execute actions and append to care history so the next turn sees them
        for action in actions:
            tool = action.get("tool")
            message = action.get("message", "")
            if tool == "notifyuser":
                user_resp = notifyuser(severity, message)
                care_history.append(f"notifyuser -> msg: {message} | user_response: {user_resp}")
                executed_steps.append("notifyuser")
            elif tool == "emergencyCall":
                call_resp = emergencyCall(message)
                care_history.append(f"emergencyCall -> msg: {message} | response: {call_resp}")
                executed_steps.append("emergencyCall")
            elif tool == "notifyDoctor":
                notifyDoctor(message)
                care_history.append(f"notifyDoctor -> report: {message}")
                executed_steps.append("notifyDoctor")
            else:
                print(f"[CARE] Unknown tool in LLM action: {tool!r}")
                care_history.append(f"unknown_tool -> {tool!r} | message: {message}")

        # Safety: avoid infinite loops if LLM keeps returning actions without end
        if len(executed_steps) > 10:
            print("[CARE] Reached safety limit of executed actions; breaking care loop.")
            break

    # Store updated care history back into the state
    return {**state, "care_history": care_history}


# --- Build LangGraph ------------------------------------------------------


def build_graph():
    graph = StateGraph(GlucoseState)

    graph.add_node("baseline_agent", baseline_agent)
    graph.add_node("sensor", sensor_node)
    graph.add_node("rnn", rnn_node)
    graph.add_node("logger", logger_node)
    graph.add_node("router", router_node)
    graph.add_node("care_agent", care_agent)

    graph.set_entry_point("baseline_agent")

    # baseline -> sensor -> rnn -> logger -> router
    graph.add_edge("baseline_agent", "sensor")
    graph.add_edge("sensor", "rnn")
    graph.add_edge("rnn", "logger")
    graph.add_edge("logger", "router")

    # conditional routing based on severity_router (condition function)
    graph.add_conditional_edges(
        "router",
        severity_router,
        {
            "back_to_sensor": "sensor",
            "care_agent": "care_agent",
            "end": END,
        },
    )

    # after care_agent always go back to sensor
    graph.add_edge("care_agent", "sensor")

    return graph.compile()


def run_demo(iterations: int = 10):
    """Run several graph steps to show the end-to-end flow.

    Baseline glucose is read from medicalrecord.txt via the baseline node.
    The router decides whether to go back to the sensor or involve the care_agent.
    """

    graph = build_graph()

    state: GlucoseState = {
        "reading": 0.0,
        "history": [],
        "severity": "normal",
        "recent_logs": [],
        "care_history": [],
    }

    # Let the graph loop autonomously for a certain number of steps.
    for i in range(iterations):
        print("\n=== Step", i + 1, "===")
        state = graph.invoke(state, {"recursion_limit": 100})

    print("\nDemo finished. Check 'glucose_log.txt' for the full log and the console for care agent actions.")


if __name__ == "__main__":
    run_demo(iterations=10)
