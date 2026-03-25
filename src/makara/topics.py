"""
Topic extraction module for the Makara platform.

Provides TF-based keyword extraction with stopword removal and n-gram
support, plus clustering of related keywords into topic groups.
"""

import re
from typing import List, Dict, Tuple, Optional


class TopicExtractor:
    """
    Extracts topics from text using term frequency analysis.

    Removes stopwords, scores words by frequency, and supports
    extraction of both unigrams and bigram n-grams.
    """

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
        "also", "about", "up", "like", "many", "much", "well", "back",
        "even", "still", "new", "now", "way", "use", "one", "two",
    }

    def __init__(self, min_word_length=3, custom_stopwords=None):
        """
        Initialize the topic extractor.

        Args:
            min_word_length: Minimum word length to consider.
            custom_stopwords: Additional stopwords to filter out.
        """
        self.min_word_length = min_word_length
        self.stopwords = set(self.STOPWORDS)
        if custom_stopwords:
            self.stopwords.update(w.lower() for w in custom_stopwords)

    def _tokenize(self, text):
        """Tokenize text into lowercase words."""
        return re.findall(r'[a-zA-Z]+', text.lower())

    def _filter_words(self, words):
        """Remove stopwords and short words."""
        return [
            w for w in words
            if w not in self.stopwords and len(w) >= self.min_word_length
        ]

    def compute_term_frequency(self, text):
        """
        Compute term frequency for all non-stopwords.

        Args:
            text: The input text.

        Returns:
            A dictionary of word to frequency count.
        """
        words = self._tokenize(text)
        filtered = self._filter_words(words)
        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        return freq

    def extract_keywords(self, text, top_n=10):
        """
        Extract the top-N keywords by term frequency.

        Args:
            text: The input text.
            top_n: Number of keywords to return.

        Returns:
            A list of keyword strings sorted by frequency (descending).
        """
        freq = self.compute_term_frequency(text)
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]

    def extract_ngrams(self, text, n=2, top_n=10):
        """
        Extract top n-grams from text.

        Args:
            text: The input text.
            n: Size of n-grams (default 2 for bigrams).
            top_n: Number of n-grams to return.

        Returns:
            A list of n-gram strings sorted by frequency.
        """
        words = self._tokenize(text)
        filtered = self._filter_words(words)

        if len(filtered) < n:
            return []

        ngram_freq = {}
        for i in range(len(filtered) - n + 1):
            ngram = " ".join(filtered[i:i + n])
            ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1

        sorted_ngrams = sorted(
            ngram_freq.items(), key=lambda x: x[1], reverse=True
        )
        return [ngram for ngram, count in sorted_ngrams[:top_n]]

    def extract_topics(self, text, num_keywords=5, num_ngrams=3):
        """
        Extract a combined set of topics (keywords + n-grams).

        Args:
            text: The input text.
            num_keywords: Number of single-word keywords.
            num_ngrams: Number of bigram phrases.

        Returns:
            A list of topic strings (keywords first, then n-grams).
        """
        keywords = self.extract_keywords(text, top_n=num_keywords)
        ngrams = self.extract_ngrams(text, n=2, top_n=num_ngrams)
        return keywords + ngrams


class TopicCluster:
    """
    Groups related keywords into topic clusters based on co-occurrence.

    Builds a co-occurrence matrix from text windows and clusters
    keywords that frequently appear near each other.
    """

    def __init__(self, window_size=5):
        """
        Initialize the topic clusterer.

        Args:
            window_size: Number of words to consider for co-occurrence.
        """
        self.window_size = window_size

    def _build_cooccurrence(self, words, target_words):
        """
        Build a co-occurrence dictionary for target words.

        Args:
            words: All tokenized words from the text.
            target_words: Keywords to track co-occurrence for.

        Returns:
            A dict mapping each keyword to a dict of co-occurring keywords
            and their counts.
        """
        target_set = set(target_words)
        cooccurrence = {w: {} for w in target_words}

        for i, word in enumerate(words):
            if word not in target_set:
                continue
            start = max(0, i - self.window_size)
            end = min(len(words), i + self.window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                neighbor = words[j]
                if neighbor in target_set:
                    cooccurrence[word][neighbor] = (
                        cooccurrence[word].get(neighbor, 0) + 1
                    )

        return cooccurrence

    def cluster(self, text, keywords=None, threshold=1):
        """
        Cluster keywords based on co-occurrence in the text.

        Args:
            text: The source text.
            keywords: Optional list of keywords to cluster. If None,
                      extracts them automatically.
            threshold: Minimum co-occurrence count to form a cluster link.

        Returns:
            A list of lists, each inner list being a cluster of related
            keywords.
        """
        extractor = TopicExtractor()
        if keywords is None:
            keywords = extractor.extract_keywords(text, top_n=15)

        if not keywords:
            return []

        words = extractor._tokenize(text)
        filtered = extractor._filter_words(words)
        cooccurrence = self._build_cooccurrence(filtered, keywords)

        # Simple greedy clustering
        clustered = set()
        clusters = []

        for keyword in keywords:
            if keyword in clustered:
                continue
            cluster = [keyword]
            clustered.add(keyword)

            # Add co-occurring words above threshold
            for neighbor, count in cooccurrence.get(keyword, {}).items():
                if neighbor not in clustered and count >= threshold:
                    cluster.append(neighbor)
                    clustered.add(neighbor)

            clusters.append(cluster)

        # Add any unclustered keywords as singletons
        for keyword in keywords:
            if keyword not in clustered:
                clusters.append([keyword])

        return clusters
