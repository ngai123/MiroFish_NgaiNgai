"""
Monte Carlo Prediction Engine
Runs the Report Agent N times with varied seeds and aggregates
convergent outcomes into a probability distribution.

Why: A single simulation run is a point estimate with no confidence interval.
Running N times reveals which outcomes are robust (appear in 80%+ of runs)
vs. noise (appear in <30% of runs). This is the difference between a guess
and a statistical prediction.
"""

import json
import re
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction

logger = get_logger('mirofish.monte_carlo')


@dataclass
class RunOutcome:
    run_index: int
    winner: str          # which agent / path won
    reasoning: str       # brief explanation
    key_events: List[str] = field(default_factory=list)
    economic_verdict: str = ""
    time_freedom_verdict: str = ""
    confidence: float = 0.0


@dataclass
class MonteCarloResult:
    monte_carlo_id: str
    simulation_id: str
    project_id: str
    n_runs: int
    completed_runs: int
    created_at: str
    status: str  # running / completed / failed

    runs: List[RunOutcome] = field(default_factory=list)

    # Aggregated results
    winner_distribution: Dict[str, float] = field(default_factory=dict)  # agent -> probability
    consensus_winner: str = ""
    consensus_confidence: float = 0.0
    convergent_events: List[str] = field(default_factory=list)
    divergent_scenarios: List[str] = field(default_factory=list)
    economic_consensus: str = ""
    full_analysis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['runs'] = [asdict(r) for r in self.runs]
        return d


# ── Prompt templates ──────────────────────────────────────────────────────────

_SINGLE_RUN_SYSTEM = """You are an impartial simulation analyst. You will be given:
1. A simulation requirement describing a prediction scenario
2. Key agent profiles (who the agents are)
3. A simulation log summary (what happened during the simulation)

Your job: Analyse the evidence and produce a concise JSON verdict.

Output ONLY valid JSON matching this schema exactly:
{
  "winner": "<name of agent or path that achieved the goal most successfully>",
  "reasoning": "<2-3 sentences explaining why this agent won>",
  "key_events": ["<event 1>", "<event 2>", "<event 3>"],
  "economic_verdict": "<one sentence on which agent achieved financial freedom first>",
  "time_freedom_verdict": "<one sentence on which agent owns their time most>",
  "confidence": <float 0.0-1.0 reflecting your confidence in this verdict given the data>
}

Be direct. Pick exactly ONE winner. Do not hedge."""

_AGGREGATE_SYSTEM = """You are a meta-analyst reviewing results from {n} independent simulation runs of the same prediction scenario.

Your job: Synthesise the runs into a final probability-weighted prediction report.

Output a detailed analytical report (NOT JSON) covering:
1. **Consensus Winner** — which agent/path wins most consistently and at what probability
2. **Why They Win** — the root cause that repeats across runs
3. **Risk Scenarios** — conditions under which the consensus winner fails
4. **Convergent Events** — things that happened in almost every run (high certainty)
5. **Divergent Scenarios** — high-variance outcomes (low certainty, worth monitoring)
6. **Economic Timeline** — when does financial freedom realistically arrive for each agent
7. **Final Verdict** — direct answer to the original prediction question with confidence %

Write in clear English. Be brutally honest about uncertainty."""


class MonteCarloRunner:
    """Runs N independent LLM analysis passes and aggregates into a distribution."""

    LEDGER_DIR = None  # set at import time after Config is available

    def __init__(self):
        import os
        self.ledger_dir = os.path.join(
            Config.OASIS_SIMULATION_DATA_DIR, '..', 'monte_carlo'
        )
        os.makedirs(self.ledger_dir, exist_ok=True)
        self.llm = LLMClient()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        simulation_id: str,
        project_id: str,
        simulation_requirement: str,
        simulation_summary: str,   # condensed log / report text
        agent_profiles: List[Dict[str, Any]],
        n_runs: int = 10,
        max_workers: int = 3,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo analysis.

        Args:
            simulation_id: source simulation
            project_id: source project
            simulation_requirement: original prediction question
            simulation_summary: condensed text from simulation log/report
            agent_profiles: list of agent profile dicts (name, type, bio)
            n_runs: number of independent analysis passes
            max_workers: parallel LLM calls

        Returns:
            MonteCarloResult with full distribution
        """
        mc_id = f"mc_{uuid.uuid4().hex[:12]}"
        result = MonteCarloResult(
            monte_carlo_id=mc_id,
            simulation_id=simulation_id,
            project_id=project_id,
            n_runs=n_runs,
            completed_runs=0,
            created_at=datetime.now().isoformat(),
            status="running",
        )
        self._save(result)

        logger.info(f"Starting Monte Carlo {mc_id}: {n_runs} runs, {max_workers} parallel")

        # Build shared context
        context = self._build_context(
            simulation_requirement, simulation_summary, agent_profiles
        )

        # Run N independent analysis passes
        outcomes: List[RunOutcome] = []
        temperatures = self._temperature_schedule(n_runs)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._single_run, i, context, temperatures[i]): i
                for i in range(n_runs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    outcome = future.result()
                    outcomes.append(outcome)
                    result.completed_runs += 1
                    logger.debug(f"MC run {idx} done: winner={outcome.winner}")
                except Exception as exc:
                    logger.warning(f"MC run {idx} failed: {exc}")

        result.runs = sorted(outcomes, key=lambda o: o.run_index)

        # Aggregate
        result = self._aggregate(result, simulation_requirement, context)
        result.status = "completed"
        self._save(result)
        logger.info(f"Monte Carlo {mc_id} complete. Consensus: {result.consensus_winner} "
                    f"({result.consensus_confidence:.0%})")
        return result

    def get(self, mc_id: str) -> Optional[MonteCarloResult]:
        import os
        path = os.path.join(self.ledger_dir, f"{mc_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        r = MonteCarloResult(**{k: v for k, v in d.items() if k != 'runs'})
        r.runs = [RunOutcome(**ro) for ro in d.get('runs', [])]
        return r

    def list_for_simulation(self, simulation_id: str) -> List[Dict[str, Any]]:
        import os
        results = []
        for fname in os.listdir(self.ledger_dir):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.ledger_dir, fname), encoding='utf-8') as f:
                    d = json.load(f)
                if d.get('simulation_id') == simulation_id:
                    results.append(d)
            except Exception:
                pass
        return sorted(results, key=lambda x: x.get('created_at', ''), reverse=True)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _single_run(self, idx: int, context: str, temperature: float) -> RunOutcome:
        messages = [
            {"role": "system", "content": _SINGLE_RUN_SYSTEM},
            {"role": "user", "content": context},
        ]
        data = self.llm.chat_json(messages=messages, temperature=temperature, max_tokens=800)
        return RunOutcome(
            run_index=idx,
            winner=str(data.get("winner", "Unknown")),
            reasoning=str(data.get("reasoning", "")),
            key_events=data.get("key_events", []),
            economic_verdict=str(data.get("economic_verdict", "")),
            time_freedom_verdict=str(data.get("time_freedom_verdict", "")),
            confidence=float(data.get("confidence", 0.5)),
        )

    def _aggregate(self, result: MonteCarloResult, requirement: str, context: str) -> MonteCarloResult:
        # Winner distribution
        counts: Dict[str, int] = {}
        all_events: List[str] = []
        for run in result.runs:
            w = run.winner.strip()
            counts[w] = counts.get(w, 0) + 1
            all_events.extend(run.key_events)

        total = max(len(result.runs), 1)
        result.winner_distribution = {k: round(v / total, 3) for k, v in counts.items()}
        if counts:
            result.consensus_winner = max(counts, key=counts.get)
            result.consensus_confidence = counts[result.consensus_winner] / total

        # Convergent events (appear in >60% of runs)
        event_counts: Dict[str, int] = {}
        for ev in all_events:
            ev_key = ev.lower().strip()[:80]
            event_counts[ev_key] = event_counts.get(ev_key, 0) + 1
        result.convergent_events = [
            ev for ev, cnt in event_counts.items()
            if cnt / total >= 0.6
        ][:8]

        # Divergent scenarios (agents that won in some but not most runs)
        result.divergent_scenarios = [
            f"{agent} won in {round(prob*100)}% of runs"
            for agent, prob in result.winner_distribution.items()
            if 0.15 <= prob < result.consensus_confidence
        ]

        # Full synthesis via LLM
        runs_summary = "\n".join(
            f"Run {r.run_index+1}: Winner={r.winner} | {r.reasoning}"
            for r in result.runs
        )
        agg_system = _AGGREGATE_SYSTEM.format(n=total)
        agg_user = (
            f"## Original Prediction Question\n{requirement}\n\n"
            f"## Winner Distribution\n"
            + "\n".join(f"- {k}: {v:.0%}" for k, v in result.winner_distribution.items())
            + f"\n\n## Individual Run Results\n{runs_summary}"
        )
        lang = get_language_instruction()
        result.full_analysis = self.llm.chat(
            messages=[
                {"role": "system", "content": f"{agg_system}\n\n{lang}"},
                {"role": "user", "content": agg_user},
            ],
            temperature=0.4,
            max_tokens=2000,
        )
        return result

    def _build_context(
        self,
        requirement: str,
        summary: str,
        profiles: List[Dict[str, Any]],
    ) -> str:
        profile_text = "\n".join(
            f"- {p.get('name','?')} ({p.get('entity_type','?')}): {p.get('bio','')}"
            for p in profiles[:12]
        )
        return (
            f"## Prediction Question\n{requirement}\n\n"
            f"## Agent Profiles\n{profile_text}\n\n"
            f"## Simulation Evidence\n{summary[:6000]}"
        )

    @staticmethod
    def _temperature_schedule(n: int) -> List[float]:
        """Spread temperatures from 0.2 to 0.8 to get diverse but grounded outputs."""
        if n == 1:
            return [0.4]
        step = 0.6 / (n - 1)
        return [round(0.2 + i * step, 2) for i in range(n)]

    def _save(self, result: MonteCarloResult):
        import os
        path = os.path.join(self.ledger_dir, f"{result.monte_carlo_id}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
