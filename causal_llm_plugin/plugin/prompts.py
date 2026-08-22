"""Frozen prompt templates for the LLM-steering plugin. PROMPT_VERSION is recorded in
every output row; any wording change bumps the version. One global skeleton, per-surface
task lines, block-level ablations only (no per-dataset tuning).
"""

PROMPT_VERSION = "v1"

SKELETON = """{task_line}

You get two kinds of evidence.
SEMANTIC EVIDENCE: phrases decoded from the variable's learned embedding.
Some phrases are noise. Judge by the dominant meaning.
CAUSAL NEIGHBORS: the variable's neighbors in a causal graph, with direction,
sign, strength, and timing. Strong edges matter more than weak ones.

Base the name only on this evidence. Do not invent specifics the evidence
does not support. If the evidence only supports a general description,
give the best general description.

Reply with ONE short label, in the same style as the neighbor labels.
Reply with the label only.

{context_block}{semantic_block}{graph_block}"""

TASK_LINES = {
    "t1": ("One item of a psychological questionnaire has lost its text. "
           "Reconstruct what this item most plausibly measures."),
    "t2": ("A hidden factor of a psychological questionnaire has no name. "
           "It causes the observed items listed below. Name the psychological "
           "construct."),
    "t3": ("One channel of a robot manipulation step log has lost its label. "
           "Reconstruct what this channel measures."),
}

T3_CONTEXT_FACT = ("CONTEXT: one channel of a robot manipulation step log. "
                   "Positions integrate from velocities with a time step of 0.05.\n\n")

# llmhead arm only: the deterministic naming head's evidence-backed proposal.
# Injected only when the head found positive structural evidence (it rewrote
# the decode); other arms never see this line, so their prompts are unchanged.
T3_HEAD_LINE = ('STRUCTURAL PROPOSAL: a deterministic analyzer of the same causal '
                'graph proposes the label "{prop}" for this channel, derived from '
                'integration and actuation evidence. Keep it only if the other '
                'evidence is consistent with it.\n\n')


def build(surface, phrases, graph_lines, with_context_fact=False, head_proposal=None):
    """Render the full prompt. phrases: list[str] or None. graph_lines: list[str] or None.
    Either block may be absent (ablation arms); at least one must be present."""
    assert phrases or graph_lines
    sem = (f"SEMANTIC EVIDENCE: {', '.join(phrases)}\n\n" if phrases else "")
    gb = ("CAUSAL NEIGHBORS:\n" + "\n".join(graph_lines) + "\n" if graph_lines else "")
    ctx = T3_CONTEXT_FACT if (with_context_fact and surface == "t3") else ""
    if head_proposal and surface == "t3":
        ctx += T3_HEAD_LINE.format(prop=head_proposal)
    return SKELETON.format(task_line=TASK_LINES[surface], context_block=ctx,
                           semantic_block=sem, graph_block=gb)
