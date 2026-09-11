"""The agent pipeline graph:
START -> inspection_agent -> characterization_agent -> root_cause_agent ->
disposition_agent -> trend_agent -> reporting_agent -> END.

reporting_agent is the sixth and final content-producing node -- it
compiles what every other agent produced into one persisted record, with
no new reasoning of its own. build_graph() remains the single place that
defines how agents connect, should later phases add more.

Every node is wrapped with src.agents.orchestrator.wrap_node before being
added to the graph, so a single agent raising (a malformed image, an
Anthropic API timeout, a ChromaDB error, ...) can't crash graph.invoke()
for the whole image -- see that module's docstring for the full
pipeline-architecture and error-resilience writeup.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.characterization_agent import characterization_agent
from src.agents.disposition_agent import disposition_agent
from src.agents.inspection_agent import inspection_agent
from src.agents.orchestrator import wrap_node
from src.agents.reporting_agent import reporting_agent
from src.agents.root_cause_agent import root_cause_agent
from src.agents.state import InspectionState
from src.agents.trend_agent import trend_agent


def build_graph() -> CompiledStateGraph:
    """Build and compile the agent graph.

    Compilation isn't free and the compiled graph is stateless across
    invoke() calls, so callers should compile once and reuse it (see
    get_graph() below) rather than calling this per image.
    """
    graph = StateGraph(InspectionState)
    graph.add_node("inspection_agent", wrap_node("inspection_agent", inspection_agent))
    graph.add_node("characterization_agent", wrap_node("characterization_agent", characterization_agent))
    graph.add_node("root_cause_agent", wrap_node("root_cause_agent", root_cause_agent))
    graph.add_node("disposition_agent", wrap_node("disposition_agent", disposition_agent))
    graph.add_node("trend_agent", wrap_node("trend_agent", trend_agent))
    graph.add_node("reporting_agent", wrap_node("reporting_agent", reporting_agent))
    graph.add_edge(START, "inspection_agent")
    graph.add_edge("inspection_agent", "characterization_agent")
    graph.add_edge("characterization_agent", "root_cause_agent")
    graph.add_edge("root_cause_agent", "disposition_agent")
    graph.add_edge("disposition_agent", "trend_agent")
    graph.add_edge("trend_agent", "reporting_agent")
    graph.add_edge("reporting_agent", END)
    return graph.compile()


_compiled_graph: CompiledStateGraph = None


def get_graph() -> CompiledStateGraph:
    """Module-level cached compiled graph, reused across calls in a process.

    Mirrors the model cache in src.inference.predict: build once, avoid
    redoing setup work (here, graph compilation) on every call.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
