"""
Live News Injector
Fetches real-world news and injects them as external shock events
into the simulation at specified rounds.

Why: A simulation seeded only on a static document becomes stale the
moment you press run. Real predictions need real-world events as
perturbations — market crashes, product launches, policy changes.
This service bridges the gap between static seed data and live reality.

Architecture:
- Fetches from RSS feeds (no API key needed)
- Extracts top N relevant headlines using keyword matching
- Returns structured EventConfig objects ready for simulation_config injection
- Supports scheduled injection at specific rounds (e.g., inject at round 10, 20, 30)
"""

import json
import re
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger('mirofish.news_injector')


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    summary: str
    url: str
    published: str
    source: str
    relevance_score: float = 0.0


@dataclass
class InjectedEvent:
    round_number: int          # inject at this simulation round
    event_type: str            # "news_shock" | "market_event" | "policy_change"
    title: str
    description: str
    impact: str                # "positive" | "negative" | "neutral" | "disruptive"
    affected_agent_types: List[str] = field(default_factory=list)
    source_url: str = ""


# ── RSS feed catalogue (all verified working, no API key required) ────────────

_RSS_FEEDS = {
    "tech_ai": [
        "https://hnrss.org/frontpage",                          # Hacker News top stories
        "https://www.artificialintelligence-news.com/feed/",    # AI News
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://www.wired.com/feed/rss",
        "https://hnrss.org/newest?q=AI+startup+founder",        # HN AI/startup filtered
    ],
    "business": [
        "https://www.entrepreneur.com/latest.rss",
        "https://techcrunch.com/category/startups/feed/",
        "https://hnrss.org/newest?q=indie+hacker+solopreneur+saas",
    ],
    "asia_tech": [
        "https://technode.com/feed/",                           # China/Asia tech
        "https://www.techinasia.com/feed",                      # Southeast Asia tech
    ],
    "general": [
        "https://www.theverge.com/rss/index.xml",
        "https://www.technologyreview.com/feed/",               # MIT Tech Review
    ],
}

# Topic → relevant RSS feeds
_TOPIC_FEEDS = {
    "ai":           ["tech_ai"],
    "artificial intelligence": ["tech_ai"],
    "startup":      ["business", "tech_ai"],
    "entrepreneur": ["business"],
    "malaysia":     ["asia_tech", "general"],
    "southeast asia": ["asia_tech"],
    "content creator": ["business", "tech_ai"],
    "hardware":     ["tech_ai"],
    "semiconductor": ["tech_ai"],
    "financial":    ["business"],
}


class NewsInjector:
    """Fetches live news and converts it to simulation-ready event configs.

    Data sources (in priority order):
    1. NewsAPI.org — best quality, needs NEWS_API_KEY in .env (free tier: 100 req/day)
    2. RSS feeds   — works with zero setup, no API key needed
    """

    MAX_ITEMS_PER_FEED = 5
    REQUEST_TIMEOUT = 8
    NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

    def __init__(self):
        self.llm = LLMClient()
        self.news_api_key = getattr(Config, 'NEWS_API_KEY', None) or None

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_relevant_news(
        self,
        simulation_requirement: str,
        max_items: int = 10,
    ) -> List[NewsItem]:
        """
        Fetch news items relevant to the simulation requirement.

        Args:
            simulation_requirement: the prediction question / context
            max_items: max number of news items to return

        Returns:
            List of NewsItem sorted by relevance
        """
        keywords = self._extract_keywords(simulation_requirement)
        all_items: List[NewsItem] = []

        # Try NewsAPI first (much better quality and targeting)
        if self.news_api_key:
            try:
                news_api_items = self._fetch_newsapi(keywords, max_items)
                all_items.extend(news_api_items)
                logger.info(f"NewsAPI returned {len(news_api_items)} items")
            except Exception as exc:
                logger.warning(f"NewsAPI fetch failed, falling back to RSS: {exc}")

        # Fall back to (or supplement with) RSS feeds
        if len(all_items) < max_items:
            feeds = self._select_feeds(simulation_requirement)
            for feed_url in feeds:
                if len(all_items) >= max_items * 2:
                    break
                try:
                    items = self._fetch_rss(feed_url)
                    all_items.extend(items)
                except Exception as exc:
                    logger.warning(f"RSS fetch failed for {feed_url}: {exc}")

        # Score relevance and rank
        for item in all_items:
            item.relevance_score = self._score_relevance(item, keywords)

        ranked = sorted(all_items, key=lambda x: x.relevance_score, reverse=True)
        # Deduplicate by title
        seen, unique = set(), []
        for item in ranked:
            key = item.title[:60].lower()
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique[:max_items]

    def build_injection_events(
        self,
        news_items: List[NewsItem],
        simulation_requirement: str,
        total_rounds: int,
        inject_at_rounds: Optional[List[int]] = None,
    ) -> List[InjectedEvent]:
        """
        Convert news items into InjectedEvent objects scheduled across rounds.

        Args:
            news_items: fetched news
            simulation_requirement: context for LLM analysis
            total_rounds: total simulation rounds
            inject_at_rounds: specific rounds to inject at (auto-spaced if None)

        Returns:
            List of InjectedEvent
        """
        if not news_items:
            return []

        # Auto-space injections across the simulation
        if inject_at_rounds is None:
            n = min(len(news_items), 5)
            step = max(1, total_rounds // (n + 1))
            inject_at_rounds = [step * (i + 1) for i in range(n)]

        # Pair news items with rounds
        pairs = list(zip(inject_at_rounds, news_items[:len(inject_at_rounds)]))

        events = []
        for round_num, item in pairs:
            event = self._news_to_event(item, round_num, simulation_requirement)
            if event:
                events.append(event)

        return events

    def format_for_simulation_config(
        self, events: List[InjectedEvent]
    ) -> List[Dict[str, Any]]:
        """
        Format events for injection into simulation_config.json event list.
        """
        return [
            {
                "round": e.round_number,
                "type": e.event_type,
                "title": e.title,
                "content": e.description,
                "impact": e.impact,
                "affected_types": e.affected_agent_types,
                "source": e.source_url,
                "is_external_shock": True,
            }
            for e in events
        ]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _fetch_newsapi(self, keywords: List[str], max_items: int) -> List[NewsItem]:
        """Fetch from NewsAPI.org using top keywords as query."""
        query = " OR ".join(keywords[:5])  # NewsAPI supports OR queries
        params = urllib.parse.urlencode({
            "q": query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": max_items,
            "apiKey": self.news_api_key,
        })
        url = f"{self.NEWSAPI_ENDPOINT}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "MiroFish/1.0"})
        with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read())

        items = []
        for article in data.get("articles", []):
            title = (article.get("title") or "").strip()
            desc = (article.get("description") or "").strip()[:300]
            if title and title != "[Removed]":
                items.append(NewsItem(
                    title=title,
                    summary=desc,
                    url=article.get("url", ""),
                    published=article.get("publishedAt", ""),
                    source=article.get("source", {}).get("name", "NewsAPI"),
                ))
        return items

    def _fetch_rss(self, url: str) -> List[NewsItem]:
        """Fetch and parse an RSS feed."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MiroFish-NewsInjector/1.0"},
        )
        with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT) as resp:
            content = resp.read()

        root = ET.fromstring(content)
        ns = ""
        items = []

        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()

            # Strip HTML from description
            desc = re.sub(r'<[^>]+>', '', desc)[:300]

            if title:
                items.append(NewsItem(
                    title=title,
                    summary=desc,
                    url=link,
                    published=pub,
                    source=url.split('/')[2],
                ))

            if len(items) >= self.MAX_ITEMS_PER_FEED:
                break

        return items

    def _select_feeds(self, requirement: str) -> List[str]:
        """Select RSS feed URLs based on requirement keywords."""
        req_lower = requirement.lower()
        selected_categories = set()

        for keyword, cats in _TOPIC_FEEDS.items():
            if keyword in req_lower:
                selected_categories.update(cats)

        if not selected_categories:
            selected_categories = {"tech_ai", "general"}

        urls = []
        for cat in selected_categories:
            urls.extend(_RSS_FEEDS.get(cat, []))

        return list(dict.fromkeys(urls))  # deduplicate preserving order

    @staticmethod
    def _extract_keywords(requirement: str) -> List[str]:
        """Extract meaningful keywords for relevance scoring."""
        stop = {"the", "a", "an", "is", "in", "on", "at", "to", "for", "of",
                "and", "or", "with", "from", "by", "will", "be", "are", "was"}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', requirement.lower())
        return [w for w in words if w not in stop]

    @staticmethod
    def _score_relevance(item: NewsItem, keywords: List[str]) -> float:
        """Score 0-1 based on keyword overlap with title + summary."""
        text = (item.title + " " + item.summary).lower()
        hits = sum(1 for kw in keywords if kw in text)
        return min(hits / max(len(keywords), 1), 1.0)

    def _news_to_event(
        self,
        item: NewsItem,
        round_num: int,
        requirement: str,
    ) -> Optional[InjectedEvent]:
        """Use LLM to classify a news item as a simulation event."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Convert the following real-world news headline into a simulation event. "
                        "Output ONLY valid JSON:\n"
                        '{"event_type": "news_shock|market_event|policy_change", '
                        '"title": "<short event title>", '
                        '"description": "<2 sentences on what happened and its implication>", '
                        '"impact": "positive|negative|neutral|disruptive", '
                        '"affected_agent_types": ["<type1>", "<type2>"]}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Simulation context: {requirement[:300]}\n\n"
                        f"News: {item.title}\n{item.summary}"
                    ),
                },
            ]
            data = self.llm.chat_json(messages=messages, temperature=0.2, max_tokens=300)
            return InjectedEvent(
                round_number=round_num,
                event_type=data.get("event_type", "news_shock"),
                title=data.get("title", item.title[:80]),
                description=data.get("description", item.summary[:200]),
                impact=data.get("impact", "neutral"),
                affected_agent_types=data.get("affected_agent_types", []),
                source_url=item.url,
            )
        except Exception as exc:
            logger.warning(f"Failed to convert news item to event: {exc}")
            # Fallback: use raw headline without LLM
            return InjectedEvent(
                round_number=round_num,
                event_type="news_shock",
                title=item.title[:80],
                description=item.summary[:200],
                impact="neutral",
                source_url=item.url,
            )
