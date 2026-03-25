"""
Core module for the Makara web intelligence platform.

Provides the main classes for web content analysis including text analysis,
entity extraction, summarization, and the orchestrating IntelligencePlatform.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class WebContent:
    """Represents scraped web content with metadata."""

    url: str
    title: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.url:
            raise ValueError("URL cannot be empty")
        if not self.text:
            raise ValueError("Text content cannot be empty")

    @property
    def word_count(self):
        """Return the number of words in the text."""
        return len(self.text.split())

    def has_metadata(self, key):
        """Check if a specific metadata key exists."""
        return key in self.metadata

    def get_metadata(self, key, default=None):
        """Retrieve a metadata value by key."""
        return self.metadata.get(key, default)


@dataclass
class ContentAnalysis:
    """Result of analyzing web content."""

    summary: str
    topics: List[str]
    sentiment: str
    entities: Dict[str, List[str]]
    word_count: int
    sentiment_score: float = 0.0
    confidence: float = 0.0

    def has_topic(self, topic):
        """Check if a specific topic was found."""
        topic_lower = topic.lower()
        return any(topic_lower in t.lower() for t in self.topics)

    def get_entities_by_type(self, entity_type):
        """Retrieve entities of a specific type."""
        return self.entities.get(entity_type, [])

    @property
    def is_positive(self):
        """Check if the overall sentiment is positive."""
        return self.sentiment == "positive"

    @property
    def is_negative(self):
        """Check if the overall sentiment is negative."""
        return self.sentiment == "negative"

    @property
    def is_neutral(self):
        """Check if the overall sentiment is neutral."""
        return self.sentiment == "neutral"


class ContentAnalyzer:
    """
    Analyzes text content for topics, sentiment, and entities.

    Uses keyword extraction for topics, positive/negative word counting
    for sentiment, and regex patterns for entity recognition.
    """

    # Common positive and negative words for basic sentiment
    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "outstanding", "superb", "brilliant", "love", "happy", "joy",
        "best", "perfect", "beautiful", "success", "successful", "win",
        "positive", "improve", "improved", "benefit", "helpful", "praise",
        "innovative", "remarkable", "impressive", "efficient", "elegant",
    }

    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "worst", "poor", "ugly",
        "hate", "sad", "angry", "fail", "failure", "wrong", "broken",
        "negative", "decline", "declined", "loss", "harmful", "criticize",
        "disappointing", "inefficient", "flawed", "defective", "inferior",
    }

    # Patterns for entity extraction
    PERSON_PATTERN = re.compile(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    )
    DATE_PATTERN = re.compile(
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
        r'|'
        r'\b((?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},?\s*\d{2,4})\b'
        r'|'
        r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
    )
    ORG_PATTERN = re.compile(
        r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*'
        r'(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Foundation|Institute|University'
        r'|Association|Organization|Group|International|Technologies)))\b'
        r'\.?'
    )

    def __init__(self, custom_positive=None, custom_negative=None):
        """
        Initialize the analyzer with optional custom word lists.

        Args:
            custom_positive: Additional positive words to include.
            custom_negative: Additional negative words to include.
        """
        self.positive_words = set(self.POSITIVE_WORDS)
        self.negative_words = set(self.NEGATIVE_WORDS)
        if custom_positive:
            self.positive_words.update(w.lower() for w in custom_positive)
        if custom_negative:
            self.negative_words.update(w.lower() for w in custom_negative)

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of text using word counting.

        Returns a tuple of (label, score, confidence) where label is
        'positive', 'negative', or 'neutral', score is -1.0 to 1.0,
        and confidence is 0.0 to 1.0.
        """
        words = text.lower().split()
        if not words:
            return ("neutral", 0.0, 0.0)

        pos_count = sum(1 for w in words if w.strip(".,!?;:\"'()") in self.positive_words)
        neg_count = sum(1 for w in words if w.strip(".,!?;:\"'()") in self.negative_words)
        total_sentiment_words = pos_count + neg_count

        if total_sentiment_words == 0:
            return ("neutral", 0.0, 0.0)

        score = (pos_count - neg_count) / total_sentiment_words
        confidence = total_sentiment_words / len(words)
        confidence = min(confidence * 5, 1.0)  # Scale up, cap at 1.0

        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return (label, round(score, 4), round(confidence, 4))

    def extract_entities(self, text):
        """
        Extract named entities from text using regex patterns.

        Returns a dictionary with keys 'persons', 'organizations', 'dates'.
        """
        entities = {
            "persons": [],
            "organizations": [],
            "dates": [],
        }

        # Extract persons (sequences of capitalized words)
        for match in self.PERSON_PATTERN.finditer(text):
            name = match.group(0)
            if name not in entities["persons"]:
                entities["persons"].append(name)

        # Extract organizations
        for match in self.ORG_PATTERN.finditer(text):
            org = match.group(0)
            if org not in entities["organizations"]:
                entities["organizations"].append(org)

        # Extract dates
        for match in self.DATE_PATTERN.finditer(text):
            date_str = match.group(0)
            if date_str and date_str not in entities["dates"]:
                entities["dates"].append(date_str)

        return entities

    def extract_keywords(self, text, top_n=10):
        """
        Extract top keywords from text using term frequency.

        Filters out common stopwords and short words.
        """
        from makara.topics import TopicExtractor
        extractor = TopicExtractor()
        return extractor.extract_keywords(text, top_n=top_n)

    def analyze(self, text):
        """
        Perform full content analysis on the given text.

        Returns a ContentAnalysis with summary, topics, sentiment,
        entities, and word count.
        """
        sentiment_label, sentiment_score, confidence = self.analyze_sentiment(text)
        entities = self.extract_entities(text)
        keywords = self.extract_keywords(text, top_n=8)

        summarizer = Summarizer()
        summary = summarizer.summarize(text)

        word_count = len(text.split())

        return ContentAnalysis(
            summary=summary,
            topics=keywords,
            sentiment=sentiment_label,
            entities=entities,
            word_count=word_count,
            sentiment_score=sentiment_score,
            confidence=confidence,
        )


class Summarizer:
    """
    Extractive summarizer that selects top sentences by scoring.

    Scores sentences based on word frequency, position, and length.
    """

    # Common stopwords to ignore in frequency scoring
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "while", "that",
        "this", "these", "those", "it", "its", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them",
        "their", "what", "which", "who", "whom", "how", "where", "there",
    }

    def __init__(self, num_sentences=3):
        """
        Initialize the summarizer.

        Args:
            num_sentences: Maximum number of sentences in the summary.
        """
        self.num_sentences = num_sentences

    def _split_sentences(self, text):
        """Split text into sentences using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _compute_word_frequencies(self, text):
        """Compute word frequencies excluding stopwords."""
        words = re.findall(r'[a-zA-Z]+', text.lower())
        freq = {}
        for word in words:
            if word not in self.STOPWORDS and len(word) > 2:
                freq[word] = freq.get(word, 0) + 1
        # Normalize by max frequency
        if freq:
            max_freq = max(freq.values())
            for word in freq:
                freq[word] = freq[word] / max_freq
        return freq

    def _score_sentence(self, sentence, word_freq, position, total_sentences):
        """
        Score a sentence based on multiple factors.

        Factors: word frequency overlap, sentence position, sentence length.
        """
        words = re.findall(r'[a-zA-Z]+', sentence.lower())
        if not words:
            return 0.0

        # Word frequency score
        freq_score = sum(word_freq.get(w, 0) for w in words) / len(words)

        # Position score: first and last sentences get a boost
        if position == 0:
            position_score = 1.0
        elif position == total_sentences - 1:
            position_score = 0.8
        else:
            position_score = 0.5 - (position / (total_sentences * 2))
            position_score = max(position_score, 0.1)

        # Length score: prefer medium-length sentences
        length = len(words)
        if 8 <= length <= 30:
            length_score = 1.0
        elif length < 8:
            length_score = length / 8.0
        else:
            length_score = 30.0 / length

        # Weighted combination
        score = (freq_score * 0.5) + (position_score * 0.3) + (length_score * 0.2)
        return score

    def summarize(self, text, num_sentences=None):
        """
        Generate an extractive summary of the given text.

        Selects the highest-scoring sentences and returns them in
        their original order.

        Args:
            text: The text to summarize.
            num_sentences: Override the default number of summary sentences.

        Returns:
            A string containing the summary sentences.
        """
        if not text or not text.strip():
            return ""

        n = num_sentences or self.num_sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= n:
            return text.strip()

        word_freq = self._compute_word_frequencies(text)
        total = len(sentences)

        scored = []
        for idx, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, word_freq, idx, total)
            scored.append((idx, sentence, score))

        # Sort by score descending, take top n
        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[:n]

        # Return in original order
        top.sort(key=lambda x: x[0])
        return " ".join(item[1] for item in top)


class IntelligencePlatform:
    """
    Orchestrates web content analysis by combining multiple analyzers.

    The platform coordinates content analysis, sentiment scoring,
    topic extraction, and summarization into a unified pipeline.
    """

    def __init__(self, num_summary_sentences=3):
        """
        Initialize the intelligence platform.

        Args:
            num_summary_sentences: Number of sentences for summaries.
        """
        self.analyzer = ContentAnalyzer()
        self.summarizer = Summarizer(num_sentences=num_summary_sentences)
        self._analysis_cache = {}

    def analyze_content(self, content):
        """
        Analyze a WebContent object and return a ContentAnalysis.

        Args:
            content: A WebContent instance to analyze.

        Returns:
            A ContentAnalysis with all analysis results.
        """
        if not isinstance(content, WebContent):
            raise TypeError("Expected a WebContent instance")

        cache_key = hash(content.text)
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        analysis = self.analyzer.analyze(content.text)
        self._analysis_cache[cache_key] = analysis
        return analysis

    def analyze_text(self, text):
        """
        Analyze raw text directly.

        Args:
            text: A string of text to analyze.

        Returns:
            A ContentAnalysis with all analysis results.
        """
        return self.analyzer.analyze(text)

    def batch_analyze(self, contents):
        """
        Analyze multiple WebContent objects.

        Args:
            contents: A list of WebContent instances.

        Returns:
            A list of ContentAnalysis results.
        """
        return [self.analyze_content(c) for c in contents]

    def summarize(self, text, num_sentences=None):
        """
        Generate a summary for the given text.

        Args:
            text: The text to summarize.
            num_sentences: Optional override for summary length.

        Returns:
            A summary string.
        """
        return self.summarizer.summarize(text, num_sentences=num_sentences)

    def compare_content(self, content_a, content_b):
        """
        Compare two WebContent objects and return analysis for both.

        Args:
            content_a: First WebContent to compare.
            content_b: Second WebContent to compare.

        Returns:
            A dict with analyses and comparison metrics.
        """
        analysis_a = self.analyze_content(content_a)
        analysis_b = self.analyze_content(content_b)

        # Find common topics
        topics_a = set(t.lower() for t in analysis_a.topics)
        topics_b = set(t.lower() for t in analysis_b.topics)
        common_topics = topics_a & topics_b

        return {
            "analysis_a": analysis_a,
            "analysis_b": analysis_b,
            "common_topics": list(common_topics),
            "sentiment_difference": abs(
                analysis_a.sentiment_score - analysis_b.sentiment_score
            ),
        }

    def clear_cache(self):
        """Clear the analysis cache."""
        self._analysis_cache.clear()
