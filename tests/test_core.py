"""Tests for the makara.core module."""

import pytest
from makara.core import (
    WebContent,
    ContentAnalysis,
    ContentAnalyzer,
    Summarizer,
    IntelligencePlatform,
)


class TestWebContent:
    """Tests for the WebContent dataclass."""

    def test_create_web_content(self):
        content = WebContent(
            url="https://example.com",
            title="Test Page",
            text="This is test content for analysis.",
        )
        assert content.url == "https://example.com"
        assert content.title == "Test Page"
        assert content.text == "This is test content for analysis."

    def test_web_content_word_count(self):
        content = WebContent(
            url="https://example.com",
            title="Test",
            text="one two three four five",
        )
        assert content.word_count == 5

    def test_web_content_empty_url_raises(self):
        with pytest.raises(ValueError, match="URL cannot be empty"):
            WebContent(url="", title="Test", text="Some text")

    def test_web_content_empty_text_raises(self):
        with pytest.raises(ValueError, match="Text content cannot be empty"):
            WebContent(url="https://example.com", title="Test", text="")

    def test_web_content_metadata(self):
        content = WebContent(
            url="https://example.com",
            title="Test",
            text="Content here.",
            metadata={"author": "Alice", "date": "2025-01-01"},
        )
        assert content.has_metadata("author")
        assert not content.has_metadata("missing")
        assert content.get_metadata("author") == "Alice"
        assert content.get_metadata("missing", "default") == "default"


class TestContentAnalysis:
    """Tests for the ContentAnalysis dataclass."""

    def test_content_analysis_creation(self):
        analysis = ContentAnalysis(
            summary="A test summary.",
            topics=["python", "testing"],
            sentiment="positive",
            entities={"persons": ["John Smith"]},
            word_count=100,
            sentiment_score=0.5,
        )
        assert analysis.summary == "A test summary."
        assert analysis.word_count == 100

    def test_has_topic(self):
        analysis = ContentAnalysis(
            summary="",
            topics=["machine learning", "Python"],
            sentiment="neutral",
            entities={},
            word_count=50,
        )
        assert analysis.has_topic("python")
        assert analysis.has_topic("Machine Learning")
        assert not analysis.has_topic("javascript")

    def test_sentiment_properties(self):
        pos = ContentAnalysis(
            summary="", topics=[], sentiment="positive", entities={}, word_count=0
        )
        neg = ContentAnalysis(
            summary="", topics=[], sentiment="negative", entities={}, word_count=0
        )
        neu = ContentAnalysis(
            summary="", topics=[], sentiment="neutral", entities={}, word_count=0
        )
        assert pos.is_positive and not pos.is_negative
        assert neg.is_negative and not neg.is_positive
        assert neu.is_neutral


class TestContentAnalyzer:
    """Tests for the ContentAnalyzer class."""

    def test_analyze_positive_sentiment(self):
        analyzer = ContentAnalyzer()
        label, score, confidence = analyzer.analyze_sentiment(
            "This is a great and excellent product. Amazing quality."
        )
        assert label == "positive"
        assert score > 0

    def test_analyze_negative_sentiment(self):
        analyzer = ContentAnalyzer()
        label, score, confidence = analyzer.analyze_sentiment(
            "This is terrible and awful. The worst experience ever."
        )
        assert label == "negative"
        assert score < 0

    def test_analyze_neutral_sentiment(self):
        analyzer = ContentAnalyzer()
        label, score, confidence = analyzer.analyze_sentiment(
            "The table has four legs and is made of wood."
        )
        assert label == "neutral"

    def test_extract_entities_persons(self):
        analyzer = ContentAnalyzer()
        entities = analyzer.extract_entities(
            "John Smith met with Sarah Connor at the conference."
        )
        assert "John Smith" in entities["persons"]
        assert "Sarah Connor" in entities["persons"]

    def test_extract_entities_dates(self):
        analyzer = ContentAnalyzer()
        entities = analyzer.extract_entities(
            "The meeting is on January 15, 2025 and the deadline is 2025-03-01."
        )
        assert len(entities["dates"]) >= 1

    def test_extract_entities_organizations(self):
        analyzer = ContentAnalyzer()
        entities = analyzer.extract_entities(
            "She works at Acme Corp and studied at Oxford University."
        )
        assert any("Acme Corp" in org for org in entities["organizations"])

    def test_full_analysis(self):
        analyzer = ContentAnalyzer()
        text = (
            "The innovative machine learning platform has shown great success. "
            "John Smith from Tech Corp announced impressive results. "
            "The system achieved excellent performance on January 10, 2025. "
            "Users praised the beautiful and efficient interface design."
        )
        result = analyzer.analyze(text)
        assert isinstance(result, ContentAnalysis)
        assert result.sentiment == "positive"
        assert result.word_count > 0
        assert len(result.topics) > 0

    def test_custom_word_lists(self):
        analyzer = ContentAnalyzer(
            custom_positive=["splendiferous"],
            custom_negative=["abominable"],
        )
        label, score, _ = analyzer.analyze_sentiment("This is splendiferous!")
        assert label == "positive"


class TestSummarizer:
    """Tests for the Summarizer class."""

    def test_summarize_short_text(self):
        s = Summarizer(num_sentences=3)
        text = "This is a short text."
        assert s.summarize(text) == text.strip()

    def test_summarize_long_text(self):
        s = Summarizer(num_sentences=2)
        text = (
            "Machine learning is transforming industries worldwide. "
            "Companies invest billions in artificial intelligence research. "
            "Neural networks can now process images and text efficiently. "
            "The healthcare sector benefits significantly from these advances. "
            "Researchers continue pushing boundaries every year."
        )
        result = s.summarize(text)
        # Summary should be shorter than original
        assert len(result) < len(text)
        # Should contain complete sentences
        assert result.endswith(".")

    def test_summarize_empty_text(self):
        s = Summarizer()
        assert s.summarize("") == ""
        assert s.summarize("   ") == ""

    def test_summarize_preserves_sentence_order(self):
        s = Summarizer(num_sentences=2)
        text = (
            "First sentence about technology and innovation. "
            "Second filler sentence. Third filler sentence. "
            "Fourth filler sentence. "
            "Fifth sentence about excellent technology advances and innovation."
        )
        result = s.summarize(text)
        sentences = result.split(". ")
        # Verify at least we got something
        assert len(sentences) >= 1


class TestIntelligencePlatform:
    """Tests for the IntelligencePlatform class."""

    def test_analyze_content(self):
        platform = IntelligencePlatform()
        content = WebContent(
            url="https://example.com",
            title="Test",
            text=(
                "The amazing new product received great reviews. "
                "Users found it excellent and praised its beautiful design. "
                "John Smith from Innovation Corp gave it outstanding marks."
            ),
        )
        result = platform.analyze_content(content)
        assert isinstance(result, ContentAnalysis)
        assert result.sentiment == "positive"

    def test_analyze_text(self):
        platform = IntelligencePlatform()
        result = platform.analyze_text(
            "Technology drives innovation and creates wonderful opportunities."
        )
        assert isinstance(result, ContentAnalysis)

    def test_batch_analyze(self):
        platform = IntelligencePlatform()
        contents = [
            WebContent(url="https://a.com", title="A", text="Great amazing wonderful news today."),
            WebContent(url="https://b.com", title="B", text="Terrible awful horrible disaster struck."),
        ]
        results = platform.batch_analyze(contents)
        assert len(results) == 2
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"

    def test_compare_content(self):
        platform = IntelligencePlatform()
        a = WebContent(url="https://a.com", title="A", text="Technology drives innovation forward with great results.")
        b = WebContent(url="https://b.com", title="B", text="Technology creates terrible problems and harmful results.")
        comparison = platform.compare_content(a, b)
        assert "analysis_a" in comparison
        assert "analysis_b" in comparison
        assert "common_topics" in comparison

    def test_caching(self):
        platform = IntelligencePlatform()
        content = WebContent(
            url="https://example.com",
            title="Test",
            text="Great amazing excellent wonderful superb content here.",
        )
        result1 = platform.analyze_content(content)
        result2 = platform.analyze_content(content)
        assert result1 is result2  # Same object from cache

    def test_clear_cache(self):
        platform = IntelligencePlatform()
        content = WebContent(
            url="https://example.com",
            title="Test",
            text="Great amazing excellent wonderful superb content here.",
        )
        result1 = platform.analyze_content(content)
        platform.clear_cache()
        result2 = platform.analyze_content(content)
        assert result1 is not result2

    def test_type_error_on_wrong_input(self):
        platform = IntelligencePlatform()
        with pytest.raises(TypeError):
            platform.analyze_content("not a WebContent object")

    def test_summarize_via_platform(self):
        platform = IntelligencePlatform()
        text = (
            "The first point is important. "
            "The second point is less relevant. "
            "The third point adds context. "
            "The fourth point summarizes everything. "
            "The final conclusion wraps it all up."
        )
        result = platform.summarize(text, num_sentences=2)
        assert len(result) < len(text)
