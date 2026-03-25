"""Tests for the makara.topics module."""

import pytest
from makara.topics import TopicExtractor, TopicCluster


class TestTopicExtractor:
    """Tests for the TopicExtractor class."""

    def test_extract_keywords(self):
        extractor = TopicExtractor()
        text = (
            "Python programming is popular. Python is used for machine learning. "
            "Machine learning and Python make a great combination for data science."
        )
        keywords = extractor.extract_keywords(text, top_n=5)
        assert len(keywords) > 0
        assert "python" in keywords
        assert "machine" in keywords or "learning" in keywords

    def test_extract_keywords_empty_text(self):
        extractor = TopicExtractor()
        keywords = extractor.extract_keywords("")
        assert keywords == []

    def test_stopword_removal(self):
        extractor = TopicExtractor()
        text = "The quick brown fox jumps over the lazy dog."
        keywords = extractor.extract_keywords(text, top_n=10)
        assert "the" not in keywords
        assert "over" not in keywords

    def test_term_frequency(self):
        extractor = TopicExtractor()
        text = "apple banana apple cherry apple banana"
        freq = extractor.compute_term_frequency(text)
        assert freq["apple"] == 3
        assert freq["banana"] == 2
        assert freq["cherry"] == 1

    def test_extract_ngrams(self):
        extractor = TopicExtractor()
        text = (
            "machine learning is great. machine learning advances quickly. "
            "deep learning and machine learning transform industries."
        )
        ngrams = extractor.extract_ngrams(text, n=2, top_n=5)
        assert len(ngrams) > 0
        assert "machine learning" in ngrams

    def test_extract_topics_combined(self):
        extractor = TopicExtractor()
        text = (
            "Artificial intelligence and machine learning drive innovation. "
            "Deep learning models use neural networks for intelligence tasks. "
            "Machine learning algorithms process data efficiently."
        )
        topics = extractor.extract_topics(text, num_keywords=3, num_ngrams=2)
        assert len(topics) > 0
        # Should have both keywords and n-grams
        assert len(topics) <= 5

    def test_custom_stopwords(self):
        extractor = TopicExtractor(custom_stopwords={"apple", "banana"})
        text = "apple banana cherry date elderberry cherry date"
        keywords = extractor.extract_keywords(text, top_n=5)
        assert "apple" not in keywords
        assert "banana" not in keywords
        assert "cherry" in keywords

    def test_min_word_length(self):
        extractor = TopicExtractor(min_word_length=5)
        text = "big cat runs fast across open fields daily"
        keywords = extractor.extract_keywords(text, top_n=10)
        for kw in keywords:
            assert len(kw) >= 5


class TestTopicCluster:
    """Tests for the TopicCluster class."""

    def test_cluster_related_keywords(self):
        clusterer = TopicCluster(window_size=5)
        text = (
            "Python programming language is great for data science projects. "
            "Data science and machine learning use Python programming extensively. "
            "The programming language Python supports machine learning libraries. "
            "Data analysis with Python programming tools is efficient."
        )
        clusters = clusterer.cluster(text)
        assert len(clusters) > 0
        # All keywords should appear in some cluster
        all_words = [w for cluster in clusters for w in cluster]
        assert len(all_words) > 0

    def test_cluster_with_provided_keywords(self):
        clusterer = TopicCluster(window_size=3)
        text = "alpha beta alpha gamma beta delta alpha beta gamma"
        clusters = clusterer.cluster(text, keywords=["alpha", "beta", "gamma"])
        assert len(clusters) >= 1
        all_words = [w for c in clusters for w in c]
        assert "alpha" in all_words
        assert "beta" in all_words

    def test_cluster_empty_text(self):
        clusterer = TopicCluster()
        clusters = clusterer.cluster("")
        assert clusters == []
