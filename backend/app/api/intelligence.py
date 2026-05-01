"""
Intelligence API
Exposes Monte Carlo, Economic Tracker, News Injector, Prediction Ledger,
and Obsidian Graph Exporter as REST endpoints.

Routes:
  POST /api/intelligence/monte-carlo/run
  GET  /api/intelligence/monte-carlo/<mc_id>
  GET  /api/intelligence/monte-carlo/list/<simulation_id>

  POST /api/intelligence/economics/init
  GET  /api/intelligence/economics/<simulation_id>
  POST /api/intelligence/economics/<simulation_id>/report

  GET  /api/intelligence/obsidian/export/<graph_id>

  POST /api/intelligence/news/fetch
  POST /api/intelligence/news/inject

  POST /api/intelligence/ledger/record
  GET  /api/intelligence/ledger/<entry_id>
  GET  /api/intelligence/ledger/simulation/<simulation_id>
  POST /api/intelligence/ledger/<entry_id>/verify
  GET  /api/intelligence/ledger/accuracy
  GET  /api/intelligence/ledger/list
"""

import traceback
from flask import Blueprint, request, jsonify, send_file
import io

from ..services.monte_carlo import MonteCarloRunner
from ..services.economic_tracker import EconomicTracker
from ..services.news_injector import NewsInjector
from ..services.prediction_ledger import PredictionLedger
from ..services.obsidian_exporter import ObsidianExporter
from ..models.project import ProjectManager
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.intelligence')
intelligence_bp = Blueprint('intelligence', __name__)


def _err(msg: str, code: int = 400):
    return jsonify({"success": False, "error": msg}), code


def _ok(data):
    return jsonify({"success": True, "data": data})


# ════════════════════════════════════════════════════════════════════════════════
# OBSIDIAN EXPORTER
# ════════════════════════════════════════════════════════════════════════════════

@intelligence_bp.route('/obsidian/export/<graph_id>', methods=['GET'])
def obsidian_export(graph_id: str):
    """
    Export a Zep graph as an Obsidian vault zip.

    Query params:
        project_id   str  optional — used to fetch project name & requirement
        name         str  optional — vault name override (default: MiroFish)

    Returns a .zip file download ready to unzip into your Obsidian vault folder.
    """
    try:
        project_id = request.args.get('project_id')
        vault_name = request.args.get('name', 'MiroFish')
        requirement = ''

        if project_id:
            project = ProjectManager.get_project(project_id)
            if project:
                vault_name = project.name or vault_name
                requirement = project.simulation_requirement or ''

        logger.info(f"Obsidian export: graph={graph_id}, project={project_id}")
        exporter = ObsidianExporter()
        zip_bytes, filename = exporter.export_to_zip(
            graph_id=graph_id,
            project_name=vault_name,
            simulation_requirement=requirement,
        )

        return send_file(
            io.BytesIO(zip_bytes),
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename,
        )

    except Exception as exc:
        logger.error(f"Obsidian export failed: {exc}", exc_info=True)
        return _err(str(exc), 500)


# ════════════════════════════════════════════════════════════════════════════════
# MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════════

@intelligence_bp.route('/monte-carlo/run', methods=['POST'])
def monte_carlo_run():
    """
    Run Monte Carlo analysis on a completed simulation.

    Body:
        simulation_id   str   required
        project_id      str   required
        n_runs          int   optional (default 10, max 30)
        simulation_summary  str  required — condensed text from the simulation/report
        agent_profiles  list optional — [{name, entity_type, bio}]
    """
    try:
        data = request.get_json() or {}
        sim_id = data.get('simulation_id')
        proj_id = data.get('project_id')
        summary = data.get('simulation_summary', '')

        if not sim_id or not proj_id:
            return _err('simulation_id and project_id are required')
        if not summary:
            return _err('simulation_summary is required — provide the report text or action log summary')

        n_runs = min(int(data.get('n_runs', 10)), 30)
        profiles = data.get('agent_profiles', [])

        # Pull simulation_requirement from project
        project = ProjectManager.get_project(proj_id)
        requirement = (project.simulation_requirement if project else '') or data.get('simulation_requirement', '')

        runner = MonteCarloRunner()
        result = runner.run(
            simulation_id=sim_id,
            project_id=proj_id,
            simulation_requirement=requirement,
            simulation_summary=summary,
            agent_profiles=profiles,
            n_runs=n_runs,
        )
        return _ok(result.to_dict())

    except Exception as exc:
        logger.error(f"Monte Carlo run failed: {exc}", exc_info=True)
        return _err(str(exc), 500)


@intelligence_bp.route('/monte-carlo/<mc_id>', methods=['GET'])
def monte_carlo_get(mc_id: str):
    """Get a specific Monte Carlo result by ID."""
    try:
        runner = MonteCarloRunner()
        result = runner.get(mc_id)
        if not result:
            return _err(f'Monte Carlo result not found: {mc_id}', 404)
        return _ok(result.to_dict())
    except Exception as exc:
        return _err(str(exc), 500)


@intelligence_bp.route('/monte-carlo/list/<simulation_id>', methods=['GET'])
def monte_carlo_list(simulation_id: str):
    """List all Monte Carlo runs for a simulation."""
    try:
        runner = MonteCarloRunner()
        results = runner.list_for_simulation(simulation_id)
        return _ok(results)
    except Exception as exc:
        return _err(str(exc), 500)


# ════════════════════════════════════════════════════════════════════════════════
# ECONOMIC TRACKER
# ════════════════════════════════════════════════════════════════════════════════

@intelligence_bp.route('/economics/init', methods=['POST'])
def economics_init():
    """
    Initialise economic states for agents and save to simulation dir.

    Body:
        simulation_id   str   required
        profiles        list  required — agent profiles from /simulation/profiles
        simulation_requirement  str  optional — used to detect freedom threshold
        freedom_threshold float optional (default 1500 USD/month)
    """
    try:
        data = request.get_json() or {}
        sim_id = data.get('simulation_id')
        profiles = data.get('profiles', [])
        if not sim_id:
            return _err('simulation_id is required')
        if not profiles:
            return _err('profiles list is required')

        threshold = float(data.get('freedom_threshold', 1500.0))
        requirement = data.get('simulation_requirement', '')

        tracker = EconomicTracker(freedom_threshold=threshold)
        states = tracker.initialise_agents(profiles, requirement)
        tracker.save(sim_id, states)

        return _ok([s.to_dict() for s in states])

    except Exception as exc:
        logger.error(f"Economics init failed: {exc}", exc_info=True)
        return _err(str(exc), 500)


@intelligence_bp.route('/economics/<simulation_id>', methods=['GET'])
def economics_get(simulation_id: str):
    """Get saved economic states for a simulation."""
    try:
        tracker = EconomicTracker()
        states = tracker.load(simulation_id)
        if not states:
            return _err(f'No economic states found for {simulation_id}. '
                        'Call /economics/init first.', 404)
        return _ok([s.to_dict() for s in states])
    except Exception as exc:
        return _err(str(exc), 500)


@intelligence_bp.route('/economics/<simulation_id>/report', methods=['POST'])
def economics_report(simulation_id: str):
    """
    Generate an LLM-written economic analysis report for this simulation.

    Body:
        simulation_requirement  str  optional
        action_log  list  optional — [{agent_id, content}] to update states first
        rounds_elapsed  int  optional
    """
    try:
        data = request.get_json() or {}
        requirement = data.get('simulation_requirement', '')
        action_log = data.get('action_log', [])
        rounds_elapsed = int(data.get('rounds_elapsed', 0))

        tracker = EconomicTracker()
        states = tracker.load(simulation_id)
        if not states:
            return _err('No economic states found. Call /economics/init first.', 404)

        if action_log:
            states = tracker.update_from_actions(states, action_log, rounds_elapsed)
            tracker.save(simulation_id, states)

        report = tracker.generate_economic_report(states, requirement)
        return _ok({"report": report, "states": [s.to_dict() for s in states]})

    except Exception as exc:
        logger.error(f"Economics report failed: {exc}", exc_info=True)
        return _err(str(exc), 500)


# ════════════════════════════════════════════════════════════════════════════════
# NEWS INJECTOR
# ════════════════════════════════════════════════════════════════════════════════

@intelligence_bp.route('/news/fetch', methods=['POST'])
def news_fetch():
    """
    Fetch live news relevant to the simulation requirement.

    Body:
        simulation_requirement  str  required
        max_items  int  optional (default 10)
    """
    try:
        data = request.get_json() or {}
        requirement = data.get('simulation_requirement', '')
        if not requirement:
            return _err('simulation_requirement is required')

        max_items = min(int(data.get('max_items', 10)), 30)
        injector = NewsInjector()
        items = injector.fetch_relevant_news(requirement, max_items=max_items)

        return _ok([{
            "title": i.title,
            "summary": i.summary,
            "url": i.url,
            "published": i.published,
            "source": i.source,
            "relevance_score": i.relevance_score,
        } for i in items])

    except Exception as exc:
        logger.error(f"News fetch failed: {exc}", exc_info=True)
        return _err(str(exc), 500)


@intelligence_bp.route('/news/inject', methods=['POST'])
def news_inject():
    """
    Convert fetched news into simulation event configs ready for injection.

    Body:
        simulation_requirement  str   required
        news_items  list  required — from /news/fetch
        total_rounds  int  required
        inject_at_rounds  list  optional — specific round numbers
    """
    try:
        data = request.get_json() or {}
        requirement = data.get('simulation_requirement', '')
        news_items_raw = data.get('news_items', [])
        total_rounds = int(data.get('total_rounds', 30))
        inject_at = data.get('inject_at_rounds')

        if not requirement or not news_items_raw:
            return _err('simulation_requirement and news_items are required')

        from ..services.news_injector import NewsItem
        news_items = [
            NewsItem(
                title=n.get('title', ''),
                summary=n.get('summary', ''),
                url=n.get('url', ''),
                published=n.get('published', ''),
                source=n.get('source', ''),
                relevance_score=n.get('relevance_score', 0.0),
            )
            for n in news_items_raw
        ]

        injector = NewsInjector()
        events = injector.build_injection_events(
            news_items, requirement, total_rounds, inject_at
        )
        formatted = injector.format_for_simulation_config(events)
        return _ok(formatted)

    except Exception as exc:
        logger.error(f"News inject failed: {exc}", exc_info=True)
        return _err(str(exc), 500)


# ════════════════════════════════════════════════════════════════════════════════
# PREDICTION LEDGER
# ════════════════════════════════════════════════════════════════════════════════

@intelligence_bp.route('/ledger/record', methods=['POST'])
def ledger_record():
    """
    Extract and record predictions from a simulation report.

    Body:
        simulation_id   str  required
        project_id      str  required
        report_text     str  required — full report text
        simulation_requirement  str  optional (auto-fetched from project if omitted)
        monte_carlo_id  str  optional
        consensus_winner  str  optional
        consensus_confidence  float  optional
        verification_days  int  optional (default 30)
    """
    try:
        data = request.get_json() or {}
        sim_id = data.get('simulation_id')
        proj_id = data.get('project_id')
        report_text = data.get('report_text', '')

        if not sim_id or not proj_id:
            return _err('simulation_id and project_id are required')
        if not report_text:
            return _err('report_text is required')

        project = ProjectManager.get_project(proj_id)
        requirement = (
            data.get('simulation_requirement')
            or (project.simulation_requirement if project else '')
            or ''
        )

        ledger = PredictionLedger()
        entry = ledger.record(
            simulation_id=sim_id,
            project_id=proj_id,
            simulation_requirement=requirement,
            report_text=report_text,
            monte_carlo_id=data.get('monte_carlo_id'),
            consensus_winner=data.get('consensus_winner', ''),
            consensus_confidence=float(data.get('consensus_confidence', 0.0)),
            verification_days=int(data.get('verification_days', 30)),
        )
        return _ok(entry.to_dict())

    except Exception as exc:
        logger.error(f"Ledger record failed: {exc}", exc_info=True)
        return _err(str(exc), 500)


@intelligence_bp.route('/ledger/<entry_id>', methods=['GET'])
def ledger_get(entry_id: str):
    """Get a specific ledger entry."""
    try:
        ledger = PredictionLedger()
        entry = ledger.get(entry_id)
        if not entry:
            return _err(f'Ledger entry not found: {entry_id}', 404)
        return _ok(entry.to_dict())
    except Exception as exc:
        return _err(str(exc), 500)


@intelligence_bp.route('/ledger/simulation/<simulation_id>', methods=['GET'])
def ledger_for_simulation(simulation_id: str):
    """Get the ledger entry for a specific simulation."""
    try:
        ledger = PredictionLedger()
        entry = ledger.get_for_simulation(simulation_id)
        if not entry:
            return _err(f'No ledger entry for simulation: {simulation_id}', 404)
        return _ok(entry.to_dict())
    except Exception as exc:
        return _err(str(exc), 500)


@intelligence_bp.route('/ledger/<entry_id>/verify', methods=['POST'])
def ledger_verify(entry_id: str):
    """
    Verify one or more prediction claims.

    Body (verify single claim):
        claim_id   str   required
        verified   bool  required
        note       str   optional

    Body (verify multiple claims):
        results: [{"claim_id": ..., "verified": bool, "note": "..."}]
    """
    try:
        data = request.get_json() or {}
        ledger = PredictionLedger()

        if 'results' in data:
            entry = ledger.verify_all(entry_id, data['results'])
        else:
            claim_id = data.get('claim_id')
            verified = data.get('verified')
            if claim_id is None or verified is None:
                return _err('claim_id and verified are required (or use results[] for batch)')
            entry = ledger.verify_claim(entry_id, claim_id, bool(verified), data.get('note', ''))

        return _ok(entry.to_dict())

    except ValueError as exc:
        return _err(str(exc), 404)
    except Exception as exc:
        logger.error(f"Ledger verify failed: {exc}", exc_info=True)
        return _err(str(exc), 500)


@intelligence_bp.route('/ledger/accuracy', methods=['GET'])
def ledger_accuracy():
    """Get global prediction accuracy metrics across all verified entries."""
    try:
        ledger = PredictionLedger()
        return _ok(ledger.accuracy_summary())
    except Exception as exc:
        return _err(str(exc), 500)


@intelligence_bp.route('/ledger/list', methods=['GET'])
def ledger_list():
    """List all ledger entries. Query param: ?status=pending|partial|verified"""
    try:
        status_filter = request.args.get('status')
        ledger = PredictionLedger()
        return _ok(ledger.list_all(status_filter))
    except Exception as exc:
        return _err(str(exc), 500)
