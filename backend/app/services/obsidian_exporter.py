"""
Obsidian Graph Exporter
Converts a Zep knowledge graph into an Obsidian vault — one markdown note
per node, wiki-linked through edges, with YAML frontmatter and tags.

Vault structure:
  MiroFish Export/
  ├── _index.md               ← vault overview + stats
  ├── _edges_index.md         ← all relationships as a table
  ├── Entities/
  │   ├── Agent 1 (AI Purist).md
  │   ├── DeepSeek V3.md
  │   └── ...
  └── Facts/
      └── _all_facts.md       ← every extracted edge fact in one place

Each entity note contains:
  - YAML frontmatter (type, labels, created_at, uuid)
  - Summary paragraph
  - Attributes table
  - Outgoing relationships (wiki-linked)
  - Incoming relationships (wiki-linked)
  - Raw facts involving this node
"""

import io
import os
import re
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.obsidian_exporter')

# Entity type → Obsidian tag colour hint (just a label, Obsidian styles them)
_TYPE_EMOJI = {
    "Student":        "🎓",
    "Engineer":       "⚙️",
    "ContentCreator": "🎬",
    "Entrepreneur":   "🚀",
    "Investor":       "💰",
    "GovernmentAgency": "🏛️",
    "TechCompany":    "🏢",
    "MediaOutlet":    "📰",
    "AcademicResearcher": "🔬",
    "Person":         "👤",
    "Organization":   "🏗️",
}
_DEFAULT_EMOJI = "📌"

# Edge type → readable verb for prose sentences
_EDGE_VERBS = {
    "WORKS_FOR":        "works for",
    "STUDIES_AT":       "studies at",
    "INVESTS_IN":       "invests in",
    "COLLABORATES_WITH": "collaborates with",
    "COMPETES_WITH":    "competes with",
    "SUPPORTS":         "supports",
    "OPPOSES":          "opposes",
    "REPORTS_ON":       "reports on",
    "COMMENTS_ON":      "comments on",
    "INFLUENCES":       "influences",
    "HAS_PROPERTY":     "has property",
    "FEATURES":         "features",
    "IS_EVALUATED_ON":  "is evaluated on",
    "HAS_COMPLEX_SYSTEM": "has complex system",
}


def _safe_filename(name: str) -> str:
    """Convert a node name to a safe filename (Obsidian-compatible)."""
    name = re.sub(r'[\\/:*?"<>|]', '-', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:100] or "unnamed"


def _wikilink(name: str) -> str:
    return f"[[{_safe_filename(name)}]]"


class ObsidianExporter:
    """Exports a Zep graph to an Obsidian-compatible markdown vault zip."""

    def __init__(self):
        self.zep = Zep(api_key=Config.ZEP_API_KEY)

    def export_to_zip(
        self,
        graph_id: str,
        project_name: str = "MiroFish",
        simulation_requirement: str = "",
    ) -> Tuple[bytes, str]:
        """
        Build and return a zip file of the Obsidian vault.

        Returns:
            (zip_bytes, filename)
        """
        logger.info(f"Starting Obsidian export for graph {graph_id}")

        # Fetch all data
        nodes = fetch_all_nodes(self.zep, graph_id)
        edges = fetch_all_edges(self.zep, graph_id)

        logger.info(f"Fetched {len(nodes)} nodes, {len(edges)} edges")

        # Build lookup maps
        uuid_to_node: Dict[str, Any] = {n.uuid_: n for n in nodes}
        # node_name → list of (edge, direction, other_node)
        node_edges: Dict[str, List] = {n.uuid_: [] for n in nodes}
        for edge in edges:
            src, tgt = edge.source_node_uuid, edge.target_node_uuid
            if src in node_edges:
                node_edges[src].append(("out", edge, tgt))
            if tgt in node_edges:
                node_edges[tgt].append(("in", edge, src))

        # Build zip in memory
        buf = io.BytesIO()
        vault_name = f"MiroFish - {project_name}"

        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Index note
            zf.writestr(
                f"{vault_name}/_index.md",
                self._build_index(nodes, edges, project_name, simulation_requirement, graph_id)
            )

            # Facts index
            zf.writestr(
                f"{vault_name}/Facts/_all_facts.md",
                self._build_facts_index(edges, uuid_to_node)
            )

            # Edge relationship index
            zf.writestr(
                f"{vault_name}/_relationships.md",
                self._build_relationships_index(edges, uuid_to_node)
            )

            # One note per node
            for node in nodes:
                content = self._build_node_note(
                    node, node_edges.get(node.uuid_, []), uuid_to_node
                )
                fname = _safe_filename(node.name or node.uuid_)
                zf.writestr(f"{vault_name}/Entities/{fname}.md", content)

        zip_bytes = buf.getvalue()
        filename = f"MiroFish_Obsidian_{graph_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        logger.info(f"Export complete: {len(zip_bytes)//1024}KB, {len(nodes)} notes")
        return zip_bytes, filename

    # ── Note builders ─────────────────────────────────────────────────────────

    def _build_node_note(
        self,
        node: Any,
        edge_list: List,
        uuid_to_node: Dict[str, Any],
    ) -> str:
        labels = node.labels or []
        entity_type = labels[0] if labels else "Entity"
        emoji = _TYPE_EMOJI.get(entity_type, _DEFAULT_EMOJI)
        attrs = node.attributes or {}
        created = (node.created_at or "")[:10]

        lines = []

        # ── YAML frontmatter ──────────────────────────────────────────────────
        lines += [
            "---",
            f'title: "{node.name}"',
            f"type: {entity_type}",
            f"tags:",
            f"  - MiroFish",
            f"  - {entity_type}",
        ]
        for lbl in labels[1:]:
            lines.append(f"  - {lbl}")
        lines += [
            f'uuid: "{node.uuid_}"',
            f'created: "{created}"',
            "---",
            "",
        ]

        # ── Header ────────────────────────────────────────────────────────────
        lines += [
            f"# {emoji} {node.name}",
            "",
            f"> **Type:** `{entity_type}`   |   **Graph:** MiroFish   |   **Created:** {created}",
            "",
        ]

        # ── Summary ───────────────────────────────────────────────────────────
        if node.summary:
            lines += [
                "## 📋 Summary",
                "",
                str(node.summary),
                "",
            ]

        # ── Attributes ────────────────────────────────────────────────────────
        clean_attrs = {k: v for k, v in attrs.items()
                       if k != "name" and v and v != "null"}
        if clean_attrs:
            lines += ["## 🏷️ Attributes", "", "| Field | Value |", "|---|---|"]
            for k, v in clean_attrs.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        # ── Outgoing relationships ─────────────────────────────────────────────
        outgoing = [(e, tgt) for (d, e, tgt) in edge_list if d == "out"]
        if outgoing:
            lines += ["## ➡️ Outgoing Relationships", ""]
            for edge, tgt_uuid in outgoing:
                tgt_node = uuid_to_node.get(tgt_uuid)
                tgt_name = tgt_node.name if tgt_node else tgt_uuid[:8]
                verb = _EDGE_VERBS.get(edge.name, edge.name.lower().replace("_", " "))
                fact = str(edge.fact or "")[:120]
                lines.append(
                    f"- **{verb}** {_wikilink(tgt_name)}"
                    + (f" — _{fact}_" if fact else "")
                )
            lines.append("")

        # ── Incoming relationships ─────────────────────────────────────────────
        incoming = [(e, src) for (d, e, src) in edge_list if d == "in"]
        if incoming:
            lines += ["## ⬅️ Incoming Relationships", ""]
            for edge, src_uuid in incoming:
                src_node = uuid_to_node.get(src_uuid)
                src_name = src_node.name if src_node else src_uuid[:8]
                verb = _EDGE_VERBS.get(edge.name, edge.name.lower().replace("_", " "))
                fact = str(edge.fact or "")[:120]
                lines.append(
                    f"- {_wikilink(src_name)} **{verb}** this"
                    + (f" — _{fact}_" if fact else "")
                )
            lines.append("")

        # ── All raw facts ─────────────────────────────────────────────────────
        all_facts = [str(e.fact or "") for (_, e, _) in edge_list if e.fact]
        if all_facts:
            lines += ["## 💡 Raw Facts", ""]
            for i, fact in enumerate(all_facts[:20], 1):
                lines.append(f"{i}. {fact}")
            if len(all_facts) > 20:
                lines.append(f"\n_...and {len(all_facts)-20} more facts_")
            lines.append("")

        return "\n".join(lines)

    def _build_index(
        self,
        nodes: List,
        edges: List,
        project_name: str,
        requirement: str,
        graph_id: str,
    ) -> str:
        # Count by type
        type_counts: Dict[str, int] = {}
        for n in nodes:
            t = (n.labels or ["Unlabeled"])[0]
            type_counts[t] = type_counts.get(t, 0) + 1

        edge_counts: Dict[str, int] = {}
        for e in edges:
            edge_counts[e.name] = edge_counts.get(e.name, 0) + 1

        lines = [
            "---",
            "tags:",
            "  - MiroFish",
            "  - Index",
            "---",
            "",
            "# 🐟 MiroFish Knowledge Graph",
            f"> **Project:** {project_name}   |   **Graph ID:** `{graph_id}`   |"
            f"   **Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        if requirement:
            lines += [
                "## 🎯 Simulation Requirement",
                "",
                "> " + requirement[:500].replace("\n", "\n> "),
                "",
            ]

        lines += [
            "## 📊 Graph Statistics",
            "",
            f"- **Total Nodes:** {len(nodes)}",
            f"- **Total Edges:** {len(edges)}",
            "",
            "### Nodes by Type",
            "",
            "| Type | Count | Emoji |",
            "|---|---|---|",
        ]
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            em = _TYPE_EMOJI.get(t, _DEFAULT_EMOJI)
            lines.append(f"| {t} | {c} | {em} |")

        lines += [
            "",
            "### Relationships by Type",
            "",
            "| Relationship | Count |",
            "|---|---|",
        ]
        for r, c in sorted(edge_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {r} | {c} |")

        lines += [
            "",
            "## 🗺️ Quick Navigation",
            "",
            "- [[_relationships]] — All relationships table",
            "- [[Facts/_all_facts]] — Every extracted fact",
            "",
            "### All Entities",
            "",
        ]
        # List all nodes grouped by type
        from collections import defaultdict
        by_type: Dict[str, List] = defaultdict(list)
        for n in nodes:
            t = (n.labels or ["Unlabeled"])[0]
            by_type[t].append(n.name)

        for t, names in sorted(by_type.items()):
            em = _TYPE_EMOJI.get(t, _DEFAULT_EMOJI)
            lines.append(f"### {em} {t}")
            for name in sorted(names):
                lines.append(f"- {_wikilink(name)}")
            lines.append("")

        return "\n".join(lines)

    def _build_facts_index(
        self, edges: List, uuid_to_node: Dict[str, Any]
    ) -> str:
        lines = [
            "---",
            "tags:",
            "  - MiroFish",
            "  - Facts",
            "---",
            "",
            "# 💡 All Extracted Facts",
            "",
            f"Total facts: **{len(edges)}**",
            "",
            "| # | Relationship | Source | Fact | Target |",
            "|---|---|---|---|---|",
        ]
        for i, edge in enumerate(edges, 1):
            src = uuid_to_node.get(edge.source_node_uuid)
            tgt = uuid_to_node.get(edge.target_node_uuid)
            src_name = _wikilink(src.name) if src else "?"
            tgt_name = _wikilink(tgt.name) if tgt else "?"
            fact = str(edge.fact or "")[:100].replace("|", "\\|")
            rel = edge.name
            lines.append(f"| {i} | `{rel}` | {src_name} | {fact} | {tgt_name} |")

        return "\n".join(lines)

    def _build_relationships_index(
        self, edges: List, uuid_to_node: Dict[str, Any]
    ) -> str:
        lines = [
            "---",
            "tags:",
            "  - MiroFish",
            "  - Relationships",
            "---",
            "",
            "# 🔗 All Relationships",
            "",
        ]

        # Group by relationship type
        from collections import defaultdict
        by_rel: Dict[str, List] = defaultdict(list)
        for edge in edges:
            by_rel[edge.name].append(edge)

        for rel_name, rel_edges in sorted(by_rel.items()):
            verb = _EDGE_VERBS.get(rel_name, rel_name.replace("_", " ").lower())
            lines += [f"## {rel_name}", f"_{verb}_", ""]
            for edge in rel_edges:
                src = uuid_to_node.get(edge.source_node_uuid)
                tgt = uuid_to_node.get(edge.target_node_uuid)
                src_name = _wikilink(src.name) if src else "?"
                tgt_name = _wikilink(tgt.name) if tgt else "?"
                fact = str(edge.fact or "")[:100]
                lines.append(f"- {src_name} → {tgt_name}" + (f" _{fact}_" if fact else ""))
            lines.append("")

        return "\n".join(lines)
