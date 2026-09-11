"""The agent pipeline graph:
START -> inspection_agent -> characterization_agent -> root_cause_agent -> END.

This is where later phases wire in the next agents (e.g. a Reporting
Agent) after root_cause_agent, so build_graph() is the single place that
defines how agents connect.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.characterization_agent import characterization_agent
from src.agents.inspection_agent import inspection_agent
from src.agents.root_cause_agent import root_cause_agent
from src.agents.state import InspectionState


def build_graph() -> CompiledStateGraph:
    """Build and compile the agent graph.

    Compilation isn't free and the compiled graph is stateless across
    invoke() calls, so callers should compile once and reuse it (see
    get_graph() below) rather than calling this per image.
    """
    graph = StateGraph(InspectionState)
    graph.add_node("inspection_agent", inspection_agent)
    graph.add_node("characterization_agent", characterization_agent)
    graph.add_node("root_cause_agent", root_cause_agent)
    graph.add_edge(START, "inspection_agent")
    graph.add_edge("inspection_agent", "characterization_agent")
    graph.add_edge("characterization_agent", "root_cause_agent")
    graph.add_edge("root_cause_agent", END)
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
