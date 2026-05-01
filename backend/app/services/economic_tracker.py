"""
Agent Economic State Machine
Tracks financial trajectories for each agent across simulation rounds.

Why: Social media posts and comments are proxies. The real prediction —
especially for career/financial freedom scenarios — requires modelling
actual economic flows: income, burn rate, milestones, and the exact
round when an agent crosses the financial freedom threshold.

This tracker:
1. Assigns each agent an economic starting state based on their role
2. Parses simulation actions to infer economic events (product launch,
   partnership, follower milestone → revenue signal)
3. Projects a financial freedom timeline with month-by-month estimates
4. Flags agents that are burning out or stagnating economically
"""

import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction

logger = get_logger('mirofish.economic_tracker')


# ── Economic profile per role ─────────────────────────────────────────────────

@dataclass
class EconomicState:
    agent_id: int
    agent_name: str
    entity_type: str

    # Starting conditions (USD/month)
    initial_savings: float = 0.0
    monthly_burn: float = 1000.0      # living costs
    monthly_revenue: float = 0.0      # active income
    passive_income: float = 0.0       # revenue not requiring direct time
    savings: float = 0.0              # accumulated savings

    # Freedom threshold (from simulation context)
    freedom_threshold: float = 1500.0  # monthly passive income to be "free"

    # Trajectory
    milestones: List[str] = field(default_factory=list)
    freedom_reached: bool = False
    freedom_round: Optional[int] = None
    projected_freedom_month: Optional[int] = None  # months from start

    # Risk indicators
    runway_months: float = 0.0        # months of savings at current burn
    growth_rate: float = 0.0          # monthly revenue growth %
    risk_level: str = "medium"        # low / medium / high / critical

    def net_monthly(self) -> float:
        return self.monthly_revenue + self.passive_income - self.monthly_burn

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Role → starting economic parameters
_ROLE_ECONOMICS: Dict[str, Dict[str, Any]] = {
    "student": {
        "initial_savings": 500,
        "monthly_burn": 800,
        "monthly_revenue": 200,
        "passive_income": 0,
        "growth_rate": 0.15,   # high growth potential, low base
    },
    "alumni": {
        "initial_savings": 2000,
        "monthly_burn": 1000,
        "monthly_revenue": 1500,
        "passive_income": 50,
        "growth_rate": 0.08,
    },
    "entrepreneur": {
        "initial_savings": 5000,
        "monthly_burn": 2000,
        "monthly_revenue": 1000,
        "passive_income": 200,
        "growth_rate": 0.20,
    },
    "contentcreator": {
        "initial_savings": 1000,
        "monthly_burn": 900,
        "monthly_revenue": 500,
        "passive_income": 100,
        "growth_rate": 0.18,
    },
    "engineer": {
        "initial_savings": 8000,
        "monthly_burn": 2500,
        "monthly_revenue": 5000,
        "passive_income": 0,
        "growth_rate": 0.05,
    },
    "researcher": {
        "initial_savings": 3000,
        "monthly_burn": 1800,
        "monthly_revenue": 3000,
        "passive_income": 0,
        "growth_rate": 0.04,
    },
    "default": {
        "initial_savings": 1000,
        "monthly_burn": 1000,
        "monthly_revenue": 500,
        "passive_income": 0,
        "growth_rate": 0.06,
    },
}

# Keywords in simulation posts/actions that signal economic events
_REVENUE_SIGNALS = [
    (r"launch|ship|release|publish|product", 200),
    (r"partner|sponsor|deal|contract", 300),
    (r"raise|funding|invest|grant", 1000),
    (r"viral|trending|blow up|growth", 150),
    (r"sale|revenue|mrr|profit|earn", 250),
    (r"client|customer|user|subscriber", 100),
]

_BURN_SIGNALS = [
    (r"fail|pivot|shutdown|quit|bankrupt", -500),
    (r"struggle|difficult|hard time|plateau", -100),
    (r"debt|loan|borrow", -200),
]


class EconomicTracker:
    """Tracks and projects agent economic trajectories through a simulation."""

    def __init__(self, freedom_threshold: float = 1500.0):
        self.freedom_threshold = freedom_threshold
        self.llm = LLMClient()

    def initialise_agents(
        self,
        profiles: List[Dict[str, Any]],
        simulation_requirement: str = "",
    ) -> List[EconomicState]:
        """
        Create initial economic states for all agents based on role + context.
        """
        # Try to detect freedom threshold from requirement
        threshold = self._extract_threshold(simulation_requirement) or self.freedom_threshold

        states = []
        for p in profiles:
            et = (p.get("entity_type") or p.get("source_entity_type") or "default").lower()
            role_key = self._map_role(et)
            defaults = _ROLE_ECONOMICS.get(role_key, _ROLE_ECONOMICS["default"])

            state = EconomicState(
                agent_id=p.get("user_id", 0),
                agent_name=p.get("name", "Unknown"),
                entity_type=et,
                initial_savings=defaults["initial_savings"],
                monthly_burn=defaults["monthly_burn"],
                monthly_revenue=defaults["monthly_revenue"],
                passive_income=defaults["passive_income"],
                savings=defaults["initial_savings"],
                freedom_threshold=threshold,
                growth_rate=defaults["growth_rate"],
            )
            state.runway_months = self._calc_runway(state)
            state.risk_level = self._calc_risk(state)
            states.append(state)

        logger.info(f"Initialised economic states for {len(states)} agents "
                    f"(freedom threshold: ${threshold}/month)")
        return states

    def update_from_actions(
        self,
        states: List[EconomicState],
        action_log: List[Dict[str, Any]],
        rounds_elapsed: int,
    ) -> List[EconomicState]:
        """
        Parse simulation action log and apply economic signals to each agent.
        Called after simulation completes.
        """
        # Index states by agent_id
        state_map = {s.agent_id: s for s in states}

        for action in action_log:
            agent_id = action.get("agent_id")
            content = (action.get("content") or action.get("post") or "").lower()
            if agent_id not in state_map or not content:
                continue

            s = state_map[agent_id]
            for pattern, delta in _REVENUE_SIGNALS:
                if re.search(pattern, content):
                    s.monthly_revenue += delta * 0.1  # each post is a small signal
                    if delta > 200:
                        s.passive_income += delta * 0.05
            for pattern, delta in _BURN_SIGNALS:
                if re.search(pattern, content):
                    s.monthly_revenue += delta * 0.1

        # Apply compound growth and update derived fields
        months = max(rounds_elapsed, 1)
        for s in states:
            # Compound monthly revenue by growth rate
            s.monthly_revenue = s.initial_savings * 0.01 + s.monthly_revenue * (
                (1 + s.growth_rate) ** (months / 12)
            )
            s.savings += s.net_monthly() * months
            s.runway_months = self._calc_runway(s)
            s.risk_level = self._calc_risk(s)

            # Check freedom milestone
            if s.passive_income >= s.freedom_threshold and not s.freedom_reached:
                s.freedom_reached = True
                s.freedom_round = rounds_elapsed
                s.milestones.append(
                    f"Financial freedom reached at round {rounds_elapsed} "
                    f"(${s.passive_income:.0f}/month passive income)"
                )

            # Project freedom arrival
            if not s.freedom_reached and s.growth_rate > 0:
                s.projected_freedom_month = self._project_freedom_months(s)

        return states

    def generate_economic_report(
        self,
        states: List[EconomicState],
        simulation_requirement: str,
    ) -> str:
        """
        Use LLM to synthesise economic trajectories into a readable report section.
        """
        states_summary = json.dumps(
            [s.to_dict() for s in states], ensure_ascii=False, indent=2
        )
        lang = get_language_instruction()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial analyst reviewing simulated economic trajectories "
                    "for agents in a prediction simulation. "
                    "Write a concise economic analysis report section covering:\n"
                    "1. Who reaches financial freedom first and when\n"
                    "2. Who is at risk of running out of runway\n"
                    "3. Monthly income progression for each agent\n"
                    "4. Key economic milestones observed\n"
                    "5. Final ranking by economic freedom achieved\n\n"
                    f"{lang}\nBe specific with numbers. Use bullet points."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Prediction Scenario\n{simulation_requirement}\n\n"
                    f"## Agent Economic States\n{states_summary}"
                ),
            },
        ]
        return self.llm.chat(messages=messages, temperature=0.3, max_tokens=1500)

    def save(self, simulation_id: str, states: List[EconomicState]):
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        path = os.path.join(sim_dir, "economic_states.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([s.to_dict() for s in states], f, ensure_ascii=False, indent=2)

    def load(self, simulation_id: str) -> List[EconomicState]:
        path = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id, "economic_states.json")
        if not os.path.exists(path):
            return []
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return [EconomicState(**d) for d in data]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _map_role(entity_type: str) -> str:
        mapping = {
            "student": "student", "alumni": "alumni",
            "entrepreneur": "entrepreneur", "ceo": "entrepreneur",
            "founder": "entrepreneur", "startup": "entrepreneur",
            "contentcreator": "contentcreator", "influencer": "contentcreator",
            "blogger": "contentcreator", "youtuber": "contentcreator",
            "engineer": "engineer", "developer": "engineer",
            "researcher": "researcher", "professor": "researcher",
            "expert": "researcher", "scientist": "researcher",
        }
        return mapping.get(entity_type.lower(), "default")

    @staticmethod
    def _calc_runway(s: EconomicState) -> float:
        burn = s.monthly_burn - s.monthly_revenue - s.passive_income
        if burn <= 0:
            return float('inf')
        return s.savings / burn if s.savings > 0 else 0.0

    @staticmethod
    def _calc_risk(s: EconomicState) -> str:
        runway = s.runway_months
        net = s.net_monthly()
        if net > 500:
            return "low"
        if net > 0 or runway > 12:
            return "medium"
        if runway > 3:
            return "high"
        return "critical"

    @staticmethod
    def _project_freedom_months(s: EconomicState) -> Optional[int]:
        """Estimate months until passive_income >= freedom_threshold under current growth."""
        if s.growth_rate <= 0 or s.passive_income <= 0:
            return None
        gap = s.freedom_threshold - s.passive_income
        if gap <= 0:
            return 0
        # passive_income * (1 + r)^n = threshold  →  n = log(threshold/passive) / log(1+r)
        try:
            n = math.log(s.freedom_threshold / s.passive_income) / math.log(1 + s.growth_rate / 12)
            return max(1, int(math.ceil(n)))
        except (ValueError, ZeroDivisionError):
            return None

    @staticmethod
    def _extract_threshold(requirement: str) -> Optional[float]:
        """Try to parse a dollar threshold from the simulation requirement."""
        match = re.search(r'\$\s*([\d,]+)\s*/?\s*month', requirement, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
        return None
