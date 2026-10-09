"""Registry of planned families (planned mode, see yev.generators.llm.planned).

Tasks 3-6 add their families here: import the family from its module and add it to PLANNED_FAMILIES under its
name. `yev synth planned-plan --family NAME` looks families up in this dict.
"""

from __future__ import annotations

from yev.generators.llm.planned import PlannedFamily
from yev.generators.llm.targets.agent import (
    AGENT_BENIGN_MULTISTEP, AGENT_INJECTION_PAIR, AGENT_OVERREACH, TRACE_AUDIT,
)

PLANNED_FAMILIES: dict[str, PlannedFamily] = {
    f.name: f for f in (AGENT_INJECTION_PAIR, AGENT_OVERREACH, AGENT_BENIGN_MULTISTEP, TRACE_AUDIT)
}
