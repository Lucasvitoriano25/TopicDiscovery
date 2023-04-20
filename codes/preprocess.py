# -*- coding: utf-8 -*-
import re
import gensim
import spacy
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser

class ArXivPreprocessor:

    """A text pre-processor for arXiv abstracts.

    Attributes
    ----------
    stopwords : array_like
        List of stopwords.

    nlp : spacy.lang.en.English
        The SpaCy English model used for lemmatization.

    n_gram_models : array_like
        List containing n-gram models.

    """

    def __init__(self):
        pass

    def fit_transform(self,
                      documents,
                      additional_stopwords=[],
                      max_n=3,
                      n_gram_threshold=100,
                      pos_tags=["NOUN", "ADJ", "PROPN"]):
        """Fit to documents and transform them.

        Parameters
        ----------
        documents : array_like
            Sequence of document strings.

        additional_stopwords : array_like, default=[]
            List of stopwords (in addition to gensim stopwords).

        max_n : int, default=3
            Maximum n value for n-gram phrase learning. Enables phrases up
            to n words in length.

        n_gram_threshold : int, default=100
            Minimum n-gram frequency threshold. All n-grams with a frequency
            lower than the threshold will be ignored.

        pos_tags : array_like, default=["NOUN", "ADJ", "PROPN"]
            Part-of-speech tags extracted from distinct tokens.

        Returns
        -------
        documents : array_like
            Tokenized and pre-processed documents.

        """

        # set instance attributes
        self.max_n = max_n
        self.n_gram_threshold = n_gram_threshold
        self.pos_tags = pos_tags
        self.stopwords = stopwords.words("english")
        self.stopwords.extend(additional_stopwords)
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        # fit-transform documents
        print(" [1/9] Removing headings...")
        documents = self.remove_heading(documents)
        print(" [2/9] Removing gradings, resources and course support...")
        documents = self.remove_useless_info(documents)
        print(" [3/9] Removing LaTex equations...")
        documents = self.remove_latex_equations(documents)
        print(" [4/9] Removing newlines and extra spaces...")
        documents = self.remove_newlines(documents)
        print(" [5/9] Tokenizing documents...")
        documents = self.tokenize(documents)
        print(" [6/9] Removing stopwords...")
        documents = self.remove_stopwords(documents, self.stopwords)
        print(" [7/9] Identifying n-gram phrases...")
        documents = self.identify_phrases(documents)
        print(" [8/9] Lemmatizing...")
        documents = self.lemmatize(documents, self.pos_tags)
        print(" [9/9] Removing common words...")
        documents = self.remove_words(documents)
        print(" Done.")
        return documents

    def transform(self, documents):
        """Transform documents.

        Parameters
        ----------
        documents : array_like
            Sequence of document strings.

        Returns
        -------
        documents : array_like
            Tokenized and pre-processed documents.
        """
        documents = self.remove_heading(documents)
        documents = self.remove_useless_info(documents)
        documents = self.remove_latex_equations(documents)
        documents = self.remove_newlines(documents)
        documents = self.tokenize(documents)
        documents = self.remove_stopwords(documents, self.stopwords)
        documents = self.identify_phrases(documents, fit=False)
        documents = self.lemmatize(documents, self.pos_tags)
        documents = self.remove_words(documents)
        return documents

    def remove_latex_equations(self, documents):
        """Remove LaTex equations."""

        def _remove_latex(doc):
            """Remove text between every two consecutive occurences of "$"."""
            indices = [match.start() for match in re.finditer("\$", doc)]
            if len(indices) % 2 == 0:
                parsed = doc
                for idx in range(0, len(indices), 2):
                    substring = doc[indices[idx]:indices[idx+1]+1]
                    parsed = parsed.replace(substring, "")
                return parsed
            else:
                return doc  # cannot process since there are an odd number of "$"s

        return [_remove_latex(doc) for doc in documents]

    def remove_newlines(self, documents):
        """Remove newline characters and extra spaces."""
        return [re.sub("\s+", " ", doc) for doc in documents]

    def tokenize(self, documents):
        """Tokenize a document using Gensim pre-processing."""
        return [simple_preprocess(str(doc)) for doc in documents]

    def remove_stopwords(self, documents, stop_words):
        """Remove stopwords."""
        return [[word for word in doc if word not in stop_words]
                for doc in documents]

    def identify_phrases(self, documents, fit=True):
        """Identify and transform phrases using n-grams."""
        processed = documents
        if fit:
            self.n_gram_models = []
            for n in range(2, self.max_n):
                n_grams = Phrases(processed, threshold=self.n_gram_threshold)
                n_gram_model = Phraser(n_grams)
                self.n_gram_models.append(n_gram_model)
                processed = [n_gram_model[doc] for doc in processed]
        else:
            for model in self.n_gram_models:
                processed = [model[doc] for doc in processed]
        return processed

    def lemmatize(self, documents, pos_tags):
        """Lemmatize documents and extract words by POS tag."""
        lemmatized = []
        for doc in documents:
            tokens = self.nlp(" ".join(doc))
            lemmatized.append([token.lemma_ for token in tokens
                               if token.pos_ in pos_tags])
        return lemmatized
    def remove_heading(self, documents):
        new_lst = []
        for i in range(len(documents)):
            text = documents[i].strip().replace('\\n', '\n').replace('\\xa0', ' ')
            lines = text.splitlines()
            # Remove the entire line containing 'Instructors:'
            lines = [line for line in lines if "Instructors" not in line and "Department" not in line and "Campus" not in line and "Language\xa0of\xa0instruction" not in line and"Workload" not in line and "On\xadsite\xa0hours" not in line]
            # Convert the list back to string
            text = '\n'.join(lines)
            # Append the modified string to a new list
            new_lst.append(text)
        return new_lst
    
    def remove_useless_info(self, documents):
        new_lst = []
        for i in range(len(documents)):
            text = documents[i].strip().replace('\\n', '\n').replace('\\xa0', ' ')
            result = re.sub(r'(Class\xa0components\xa0\(lecture,\xa0labs,\xa0etc.\))(.*?)(Grading)', r'\1\n\3', text, flags=re.DOTALL)
            result = re.sub(r'(Grading)(.*?)(Resources)', r'\1\n\3', result, flags=re.DOTALL)
            result = re.sub(r'(Grading)(.*?)(Course\xa0support)', r'\1\n\3', result, flags=re.DOTALL)
            result = re.sub(r'(Course\xa0support)(.*?)(Resources)', r'\1\n\3', result, flags=re.DOTALL)
            result = re.sub(r'(Resources)(.*?)(Learning\xa0outcomes\xa0covered\xa0on\xa0the\xa0course)', r'\1\n\3', result, flags=re.DOTALL)
            # Append the modified string to a new list
            new_lst.append(result)
        return new_lst
    
    def remove_words(self,documents):
        newdoc = []
        wordstoremove = ["course","student", "end", "day", "campus", "group", "part","grading","class","components","resources","learning","outcomes","covered","support"]
        for doc in documents:
            newdoc.append([i for i in doc if not (i in wordstoremove)])
        return newdoc
    
