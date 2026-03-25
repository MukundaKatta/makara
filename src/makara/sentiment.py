"""
Sentiment analysis module for the Makara platform.

Provides fine-grained sentiment scoring using curated positive and negative
word lists with contextual modifiers for negation and intensification.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a text."""

    score: float  # -1.0 (most negative) to 1.0 (most positive)
    label: str  # "positive", "negative", or "neutral"
    confidence: float  # 0.0 to 1.0

    @property
    def is_positive(self):
        """Check if sentiment is positive."""
        return self.label == "positive"

    @property
    def is_negative(self):
        """Check if sentiment is negative."""
        return self.label == "negative"

    @property
    def is_neutral(self):
        """Check if sentiment is neutral."""
        return self.label == "neutral"

    def __str__(self):
        return "SentimentResult(label={}, score={:.3f}, confidence={:.3f})".format(
            self.label, self.score, self.confidence
        )


class SentimentAnalyzer:
    """
    Scores text sentiment from -1.0 to 1.0 using curated word lists.

    Contains built-in lists of 50 positive and 50 negative words. Supports
    negation detection (e.g., 'not good' flips polarity) and intensity
    modifiers (e.g., 'very good' amplifies score).
    """

    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful",
        "fantastic", "outstanding", "superb", "brilliant", "love",
        "happy", "joy", "joyful", "delightful", "pleased",
        "best", "perfect", "beautiful", "success", "successful",
        "win", "winning", "positive", "improve", "improved",
        "benefit", "helpful", "praise", "innovative", "remarkable",
        "impressive", "efficient", "elegant", "bright", "cheerful",
        "charming", "creative", "dynamic", "energetic", "fabulous",
        "generous", "graceful", "harmonious", "ideal", "magnificent",
        "noble", "optimistic", "pleasant", "radiant", "splendid",
    }

    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "worst",
        "poor", "ugly", "hate", "sad", "angry",
        "fail", "failure", "wrong", "broken", "negative",
        "decline", "declined", "loss", "harmful", "criticize",
        "disappointing", "inefficient", "flawed", "defective", "inferior",
        "dreadful", "miserable", "pathetic", "tragic", "disastrous",
        "appalling", "atrocious", "abysmal", "lousy", "wretched",
        "grim", "painful", "frustrating", "annoying", "boring",
        "careless", "corrupt", "cruel", "dangerous", "destructive",
        "hostile", "offensive", "repulsive", "shameful", "toxic",
    }

    NEGATION_WORDS = {
        "not", "no", "never", "neither", "nobody", "nothing",
        "nowhere", "nor", "cannot", "without", "hardly", "barely",
    }

    INTENSIFIERS = {
        "very": 1.5,
        "extremely": 2.0,
        "incredibly": 1.8,
        "really": 1.3,
        "absolutely": 1.7,
        "quite": 1.2,
        "highly": 1.5,
        "deeply": 1.4,
        "utterly": 1.8,
        "remarkably": 1.5,
    }

    def __init__(self, custom_positive=None, custom_negative=None):
        """
        Initialize the sentiment analyzer.

        Args:
            custom_positive: Optional set of additional positive words.
            custom_negative: Optional set of additional negative words.
        """
        self.positive_words = set(self.POSITIVE_WORDS)
        self.negative_words = set(self.NEGATIVE_WORDS)
        if custom_positive:
            self.positive_words.update(w.lower() for w in custom_positive)
        if custom_negative:
            self.negative_words.update(w.lower() for w in custom_negative)

    def _clean_word(self, word):
        """Strip punctuation from a word for matching."""
        return word.lower().strip(".,!?;:\"'()[]{}")

    def _score_words(self, words):
        """
        Score a list of words considering negation and intensifiers.

        Returns a tuple of (total_score, sentiment_word_count).
        """
        total_score = 0.0
        sentiment_count = 0
        negated = False
        intensity = 1.0

        for word in words:
            clean = self._clean_word(word)

            # Check for negation
            if clean in self.NEGATION_WORDS:
                negated = True
                continue

            # Check for intensifiers
            if clean in self.INTENSIFIERS:
                intensity = self.INTENSIFIERS[clean]
                continue

            # Score the word
            word_score = 0.0
            if clean in self.positive_words:
                word_score = 1.0
                sentiment_count += 1
            elif clean in self.negative_words:
                word_score = -1.0
                sentiment_count += 1

            if word_score != 0.0:
                if negated:
                    word_score = -word_score
                word_score *= intensity
                total_score += word_score

            # Reset modifiers after each sentiment word
            negated = False
            intensity = 1.0

        return total_score, sentiment_count

    def analyze(self, text):
        """
        Analyze the sentiment of the given text.

        Args:
            text: The text to analyze.

        Returns:
            A SentimentResult with score, label, and confidence.
        """
        if not text or not text.strip():
            return SentimentResult(score=0.0, label="neutral", confidence=0.0)

        words = text.split()
        total_score, sentiment_count = self._score_words(words)

        if sentiment_count == 0:
            return SentimentResult(score=0.0, label="neutral", confidence=0.0)

        # Normalize score to -1.0 to 1.0 range
        normalized = total_score / sentiment_count
        normalized = max(-1.0, min(1.0, normalized))

        # Confidence based on proportion of sentiment words
        confidence = sentiment_count / len(words)
        confidence = min(confidence * 5, 1.0)

        # Determine label
        if normalized > 0.1:
            label = "positive"
        elif normalized < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            score=round(normalized, 4),
            label=label,
            confidence=round(confidence, 4),
        )

    def analyze_sentences(self, text):
        """
        Analyze sentiment per sentence.

        Args:
            text: Text containing multiple sentences.

        Returns:
            A list of (sentence, SentimentResult) tuples.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        results = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                result = self.analyze(sentence)
                results.append((sentence, result))
        return results

    def get_overall_sentiment(self, texts):
        """
        Compute average sentiment across multiple texts.

        Args:
            texts: A list of text strings.

        Returns:
            A SentimentResult representing the aggregate sentiment.
        """
        if not texts:
            return SentimentResult(score=0.0, label="neutral", confidence=0.0)

        results = [self.analyze(t) for t in texts]
        avg_score = sum(r.score for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)

        if avg_score > 0.1:
            label = "positive"
        elif avg_score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            score=round(avg_score, 4),
            label=label,
            confidence=round(avg_confidence, 4),
        )
