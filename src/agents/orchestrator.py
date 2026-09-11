"""Phase 9 -- Orchestrator: pipeline architecture, error resilience, escalation.

Pipeline architecture
----------------------
One image moves through a fixed, sequential LangGraph pipeline of six
content-producing agents, each reading whatever fields it needs off the
shared `InspectionState` (src.agents.state) and returning only the fields
it adds:

    inspection_agent        -- runs the trained classifier + Grad-CAM
                                (src.inference.predict); no reasoning.
    characterization_agent  -- heuristic, rule-based description of the
                                Grad-CAM pattern (skipped for 'ok').
    root_cause_agent        -- RAG: retrieves from a curated ChromaDB
                                knowledge base and asks Claude to explain
                                likely process causes (skipped for 'ok').
    disposition_agent       -- rule-based accept/rework/scrap/escalate
                                call; always runs, and never depends on
                                root_cause (see "Error resilience" below).
    trend_agent             -- logs the inspection and computes recent
                                defect/scrap-rate drift for its (SIMULATED
                                -- see that module) batch.
    reporting_agent         -- compiles everything above into one
                                persisted record with a short summary.

src.agents.graph.build_graph() wires these into the graph and is the
single place their order is defined; src.agents.run_pipeline runs one
image through it, src.agents.run_batch (this phase) runs many.

Error resilience
------------------
build_graph() wraps every node in `wrap_node()` below before adding it to
the graph. If a node raises -- a malformed image, an Anthropic API
timeout, a ChromaDB error, or anything else -- the wrapper catches it,
appends a record to state['errors'] (which agent, what error) and to
state['agent_outputs'][agent_name] (so the failure shows up in the same
place a normal result would), and returns without setting that node's own
output fields. The graph keeps running rather than raising out of
graph.invoke() for the whole image.

Downstream agents that don't need the failed agent's output are
unaffected: disposition_agent, in particular, only reads label/
confidence/defect_characterization, never root_cause, so a
root_cause_agent failure (e.g. an Anthropic API timeout) still lets
disposition_agent -- and everything after it -- run normally. An agent
that genuinely cannot proceed without the missing data (e.g.
disposition_agent after an inspection_agent failure, with no label to
decide on) will itself raise and get caught the same way, adding its own
record to state['errors'] rather than crashing the batch.

Escalation handling
----------------------
disposition_agent sets state['human_review_required'] = True exactly
when disposition.decision == 'escalate' (False otherwise) -- a top-level
boolean rather than something a caller has to notice by checking
disposition.decision themselves. reporting_agent prefixes summary_text
with '[HUMAN REVIEW REQUIRED] ' whenever it's set, so an escalation reads
as an unmissable flag rather than one value among four in the disposition
field.
"""

import logging
from typing import Any, Callable, Dict

from src.agents.state import InspectionState

logger = logging.getLogger(__name__)

NodeFn = Callable[[InspectionState], Dict[str, Any]]


def wrap_node(agent_name: str, node_fn: NodeFn) -> NodeFn:
    """Wrap a LangGraph node function so an exception can't crash the pipeline.

    On success, returns node_fn's own partial state update unchanged. On
    an exception, logs it, and instead returns a partial update that
    records the failure in state['errors'] and state['agent_outputs']
    without setting any of node_fn's normal output fields -- see the
    module docstring for how downstream nodes cope with that.
    """

    def wrapped(state: InspectionState) -> Dict[str, Any]:
        try:
            return node_fn(state)
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: any agent, any failure
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("Agent %r failed on %r: %s", agent_name, state.get("image_path"), message)
            return {
                "errors": [{"agent": agent_name, "error": message}],
                "agent_outputs": {agent_name: {"error": message, "failed": True}},
            }

    wrapped.__name__ = f"{agent_name}_wrapped"
    return wrapped
