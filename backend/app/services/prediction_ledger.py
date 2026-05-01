"""
Prediction Ledger
Stores every simulation's key predictions with timestamps, then tracks
whether they came true when the user verifies outcomes later.

Why: Without a feedback loop you cannot improve. This ledger is the
telemetry system — it records what MiroFish predicted and scores it
against reality. Over time, accuracy metrics reveal which simulation
configurations, prompt structures, and model choices actually work.

Usage flow:
1. Simulation completes → ledger.record() called automatically
2. User sees predictions in the UI
3. N days later, user marks outcomes verified/falsified
4. Ledger scores accuracy → feeds back into simulation quality metrics
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction

logger = get_logger('mirofish.prediction_ledger')


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PredictionClaim:
    """A single falsifiable prediction extracted from a simulation."""
    claim_id: str
    text: str                    # the actual prediction statement
    category: str                # financial / timeline / behavioral / outcome
    confidence: float            # 0.0-1.0 (from Monte Carlo or LLM)
    agent_subject: str           # which agent this prediction is about
    verification_date: str       # ISO date when to check this claim
    verified: Optional[bool] = None     # True=correct, False=wrong, None=pending
    verified_at: Optional[str] = None
    verification_note: str = ""  # user's note when verifying


@dataclass
class LedgerEntry:
    """One simulation's full prediction record."""
    entry_id: str
    simulation_id: str
    project_id: str
    simulation_requirement: str
    created_at: str
    verification_due: str        # suggested verification date
    status: str                  # pending / partial / verified

    claims: List[PredictionClaim] = field(default_factory=list)
    monte_carlo_id: Optional[str] = None
    consensus_winner: str = ""
    consensus_confidence: float = 0.0

    # Accuracy metrics (filled after verification)
    accuracy_score: Optional[float] = None   # % of claims verified True
    total_claims: int = 0
    verified_claims: int = 0
    correct_claims: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['claims'] = [asdict(c) for c in self.claims]
        return d


# ── Prediction extraction prompt ───────────────────────────────────────────────

_EXTRACT_SYSTEM = """You are extracting falsifiable prediction claims from a simulation report.

A good prediction claim is:
- Specific and measurable ("Agent 1 reaches $3K MRR by month 18")
- NOT vague ("Agent 1 will do well")
- Tied to a verifiable timeframe

From the simulation report, extract 5-8 key prediction claims.

Output ONLY valid JSON:
{
  "claims": [
    {
      "text": "<the specific prediction>",
      "category": "financial|timeline|behavioral|outcome",
      "confidence": <0.0-1.0>,
      "agent_subject": "<agent name or 'all'>",
      "months_to_verify": <int: how many months until this can be checked>
    }
  ],
  "consensus_winner": "<which agent wins overall>",
  "consensus_confidence": <0.0-1.0>
}"""


class PredictionLedger:
    """Stores, retrieves, and scores simulation predictions."""

    def __init__(self):
        self.ledger_dir = os.path.join(
            Config.OASIS_SIMULATION_DATA_DIR, '..', 'ledger'
        )
        os.makedirs(self.ledger_dir, exist_ok=True)
        self.llm = LLMClient()

    # ── Recording ──────────────────────────────────────────────────────────────

    def record(
        self,
        simulation_id: str,
        project_id: str,
        simulation_requirement: str,
        report_text: str,
        monte_carlo_id: Optional[str] = None,
        consensus_winner: str = "",
        consensus_confidence: float = 0.0,
        verification_days: int = 30,
    ) -> LedgerEntry:
        """
        Extract predictions from a report and store in the ledger.

        Args:
            simulation_id: source simulation
            project_id: source project
            simulation_requirement: original question
            report_text: full report text to extract claims from
            monte_carlo_id: linked Monte Carlo run (if any)
            consensus_winner: Monte Carlo consensus winner
            consensus_confidence: confidence in that winner
            verification_days: when to check predictions (default 30 days)

        Returns:
            LedgerEntry
        """
        lang = get_language_instruction()
        messages = [
            {"role": "system", "content": f"{_EXTRACT_SYSTEM}\n\n{lang}"},
            {
                "role": "user",
                "content": (
                    f"## Simulation Requirement\n{simulation_requirement}\n\n"
                    f"## Report\n{report_text[:8000]}"
                ),
            },
        ]

        try:
            data = self.llm.chat_json(messages=messages, temperature=0.2, max_tokens=1000)
        except Exception as exc:
            logger.warning(f"Claim extraction failed: {exc}. Using empty claims.")
            data = {"claims": [], "consensus_winner": consensus_winner,
                    "consensus_confidence": consensus_confidence}

        now = datetime.now()
        entry = LedgerEntry(
            entry_id=f"led_{uuid.uuid4().hex[:12]}",
            simulation_id=simulation_id,
            project_id=project_id,
            simulation_requirement=simulation_requirement,
            created_at=now.isoformat(),
            verification_due=(now + timedelta(days=verification_days)).isoformat(),
            status="pending",
            monte_carlo_id=monte_carlo_id,
            consensus_winner=data.get("consensus_winner", consensus_winner),
            consensus_confidence=data.get("consensus_confidence", consensus_confidence),
        )

        for c in data.get("claims", []):
            months = int(c.get("months_to_verify", 1))
            verify_date = (now + timedelta(days=months * 30)).isoformat()
            entry.claims.append(PredictionClaim(
                claim_id=f"clm_{uuid.uuid4().hex[:8]}",
                text=c.get("text", ""),
                category=c.get("category", "outcome"),
                confidence=float(c.get("confidence", 0.5)),
                agent_subject=c.get("agent_subject", "unknown"),
                verification_date=verify_date,
            ))

        entry.total_claims = len(entry.claims)
        self._save(entry)
        logger.info(f"Ledger entry {entry.entry_id}: {entry.total_claims} claims recorded "
                    f"for simulation {simulation_id}")
        return entry

    # ── Verification ───────────────────────────────────────────────────────────

    def verify_claim(
        self,
        entry_id: str,
        claim_id: str,
        verified: bool,
        note: str = "",
    ) -> LedgerEntry:
        """Mark a single prediction claim as verified or falsified."""
        entry = self.get(entry_id)
        if not entry:
            raise ValueError(f"Ledger entry not found: {entry_id}")

        for claim in entry.claims:
            if claim.claim_id == claim_id:
                claim.verified = verified
                claim.verified_at = datetime.now().isoformat()
                claim.verification_note = note
                break

        self._update_scores(entry)
        self._save(entry)
        return entry

    def verify_all(
        self,
        entry_id: str,
        results: List[Dict[str, Any]],  # [{"claim_id": ..., "verified": bool, "note": ...}]
    ) -> LedgerEntry:
        """Batch verify multiple claims at once."""
        entry = self.get(entry_id)
        if not entry:
            raise ValueError(f"Ledger entry not found: {entry_id}")

        result_map = {r["claim_id"]: r for r in results}
        for claim in entry.claims:
            if claim.claim_id in result_map:
                r = result_map[claim.claim_id]
                claim.verified = r.get("verified")
                claim.verified_at = datetime.now().isoformat()
                claim.verification_note = r.get("note", "")

        self._update_scores(entry)
        self._save(entry)
        return entry

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def get(self, entry_id: str) -> Optional[LedgerEntry]:
        path = os.path.join(self.ledger_dir, f"{entry_id}.json")
        if not os.path.exists(path):
            return None
        return self._load(path)

    def get_for_simulation(self, simulation_id: str) -> Optional[LedgerEntry]:
        for fname in os.listdir(self.ledger_dir):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(self.ledger_dir, fname)
            try:
                with open(path, encoding='utf-8') as f:
                    d = json.load(f)
                if d.get('simulation_id') == simulation_id:
                    return self._load(path)
            except Exception:
                pass
        return None

    def list_all(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = []
        for fname in os.listdir(self.ledger_dir):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.ledger_dir, fname), encoding='utf-8') as f:
                    d = json.load(f)
                if status_filter is None or d.get('status') == status_filter:
                    entries.append(d)
            except Exception:
                pass
        return sorted(entries, key=lambda x: x.get('created_at', ''), reverse=True)

    def accuracy_summary(self) -> Dict[str, Any]:
        """Global accuracy metrics across all verified ledger entries."""
        all_entries = self.list_all()
        verified = [e for e in all_entries if e.get('accuracy_score') is not None]
        if not verified:
            return {"total_entries": len(all_entries), "verified_entries": 0,
                    "overall_accuracy": None, "by_category": {}}

        scores = [e['accuracy_score'] for e in verified if e['accuracy_score'] is not None]
        overall = sum(scores) / len(scores) if scores else 0

        # Per-category accuracy
        cat_correct: Dict[str, int] = {}
        cat_total: Dict[str, int] = {}
        for e in verified:
            for c in e.get('claims', []):
                cat = c.get('category', 'unknown')
                if c.get('verified') is not None:
                    cat_total[cat] = cat_total.get(cat, 0) + 1
                    if c.get('verified'):
                        cat_correct[cat] = cat_correct.get(cat, 0) + 1

        by_cat = {
            cat: round(cat_correct.get(cat, 0) / total, 3)
            for cat, total in cat_total.items()
        }

        return {
            "total_entries": len(all_entries),
            "verified_entries": len(verified),
            "overall_accuracy": round(overall, 3),
            "by_category": by_cat,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update_scores(self, entry: LedgerEntry):
        verified = [c for c in entry.claims if c.verified is not None]
        correct = [c for c in verified if c.verified is True]
        entry.verified_claims = len(verified)
        entry.correct_claims = len(correct)
        if verified:
            entry.accuracy_score = len(correct) / len(verified)
        entry.status = (
            "verified" if len(verified) == entry.total_claims and entry.total_claims > 0
            else "partial" if verified
            else "pending"
        )

    def _save(self, entry: LedgerEntry):
        path = os.path.join(self.ledger_dir, f"{entry.entry_id}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)

    def _load(self, path: str) -> LedgerEntry:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        claims = [PredictionClaim(**c) for c in d.pop('claims', [])]
        entry = LedgerEntry(**d)
        entry.claims = claims
        return entry
