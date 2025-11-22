import time
from datetime import datetime
from typing import Dict, Literal, TypedDict, List

from langgraph.graph import StateGraph, END


class GlucoseState(TypedDict):
    reading: float
    history: List[float]
    baseline: float
    severity: Literal["normal", "low", "medium", "high"]


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

    # Simula la cadenza di 10 secondi tra una lettura e l'altra.
    print(f"[SENSOR] New glucose reading: {reading} mg/dL (idx={idx})")
    time.sleep(1)  # per la demo usiamo 1 secondo invece di 10

    return {
        **state,
        "reading": reading,
        "history": new_history,
    }


# --- Mock RNN node --------------------------------------------------------


def _classify_severity(reading: float, baseline: float) -> Literal["normal", "low", "medium", "high"]:
    """Very simple rule-based classifier standing in for an RNN.

    Usa la differenza percentuale dal baseline per decidere la severità.
    """
    if baseline <= 0:
        return "normal"

    delta = (reading - baseline) / baseline

    # Soglie totalmente arbitrarie, solo per mostrare la logica.
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

LOG_FILE = "glucose_log.txt"


def logger_node(state: GlucoseState) -> GlucoseState:
    timestamp = datetime.utcnow().isoformat()
    line = f"{timestamp}\treading={state['reading']}\tseverity={state['severity']}\n"

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

    print(f"[LOGGER] {line.strip()}")
    return state


# --- Build LangGraph ------------------------------------------------------


def build_graph():
    graph = StateGraph(GlucoseState)

    graph.add_node("sensor", sensor_node)
    graph.add_node("rnn", rnn_node)
    graph.add_node("logger", logger_node)

    graph.set_entry_point("sensor")
    graph.add_edge("sensor", "rnn")
    graph.add_edge("rnn", "logger")
    graph.add_edge("logger", END)

    return graph.compile()


def run_demo(iterations: int = 5, baseline: float = 100.0):
    """Esegue qualche ciclo del grafo per mostrare il flusso end-to-end."""

    graph = build_graph()

    state: GlucoseState = {
        "reading": 0.0,
        "history": [],
        "baseline": baseline,
        "severity": "normal",
    }

    for i in range(iterations):
        print("\n=== Cycle", i + 1, "===")
        # Ogni esecuzione del grafo va da sensor -> rnn -> logger.
        state = graph.invoke(state)

    print("\nDemo terminata. Controlla il file 'glucose_log.txt' per il log completo.")


if __name__ == "__main__":
    # Piccola demo: 5 letture, baseline 100 mg/dL.
    run_demo(iterations=5, baseline=100.0)

