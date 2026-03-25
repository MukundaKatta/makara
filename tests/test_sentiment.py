"""Tests for the makara.sentiment module."""

import pytest
from makara.sentiment import SentimentAnalyzer, SentimentResult


class TestSentimentResult:
    """Tests for the SentimentResult dataclass."""

    def test_positive_result(self):
        result = SentimentResult(score=0.8, label="positive", confidence=0.9)
        assert result.is_positive
        assert not result.is_negative
        assert not result.is_neutral

    def test_negative_result(self):
        result = SentimentResult(score=-0.6, label="negative", confidence=0.7)
        assert result.is_negative
        assert not result.is_positive

    def test_neutral_result(self):
        result = SentimentResult(score=0.0, label="neutral", confidence=0.1)
        assert result.is_neutral

    def test_str_representation(self):
        result = SentimentResult(score=0.5, label="positive", confidence=0.8)
        text = str(result)
        assert "positive" in text
        assert "0.500" in text


class TestSentimentAnalyzer:
    """Tests for the SentimentAnalyzer class."""

    def test_positive_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("This is absolutely wonderful and amazing!")
        assert result.label == "positive"
        assert result.score > 0
        assert result.confidence > 0

    def test_negative_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("This is terrible and awful, the worst ever.")
        assert result.label == "negative"
        assert result.score < 0

    def test_neutral_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("The table is made of wood and has four legs.")
        assert result.label == "neutral"
        assert result.score == 0.0

    def test_empty_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("")
        assert result.label == "neutral"
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_negation_detection(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("This is not good at all.")
        assert result.score <= 0

    def test_intensifier_effect(self):
        analyzer = SentimentAnalyzer()
        normal = analyzer.analyze("The product is good.")
        intensified = analyzer.analyze("The product is extremely good.")
        # Intensified should have equal or greater magnitude
        assert intensified.score >= normal.score

    def test_custom_word_lists(self):
        analyzer = SentimentAnalyzer(
            custom_positive={"groovy", "tubular"},
            custom_negative={"bogus"},
        )
        result = analyzer.analyze("That is totally groovy and tubular!")
        assert result.label == "positive"

    def test_analyze_sentences(self):
        analyzer = SentimentAnalyzer()
        text = "This is great! But that was terrible."
        results = analyzer.analyze_sentences(text)
        assert len(results) == 2
        assert results[0][1].label == "positive"
        assert results[1][1].label == "negative"

    def test_overall_sentiment_positive(self):
        analyzer = SentimentAnalyzer()
        texts = [
            "Great product!",
            "Wonderful experience!",
            "Excellent quality!",
        ]
        result = analyzer.get_overall_sentiment(texts)
        assert result.label == "positive"

    def test_overall_sentiment_empty(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.get_overall_sentiment([])
        assert result.label == "neutral"
        assert result.score == 0.0

    def test_score_range(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "amazing wonderful excellent great fantastic superb brilliant"
        )
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
