"""Makara - Web intelligence platform for scraping, analyzing, and summarizing web content."""

__version__ = "0.1.0"

from makara.core import (
    IntelligencePlatform,
    WebContent,
    ContentAnalysis,
    ContentAnalyzer,
    Summarizer,
)
from makara.sentiment import SentimentAnalyzer, SentimentResult
from makara.topics import TopicExtractor, TopicCluster

__all__ = [
    "IntelligencePlatform",
    "WebContent",
    "ContentAnalysis",
    "ContentAnalyzer",
    "Summarizer",
    "SentimentAnalyzer",
    "SentimentResult",
    "TopicExtractor",
    "TopicCluster",
]
