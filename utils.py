# -*- coding: utf-8 -*-
import json
import pickle
import requests
from bs4 import BeautifulSoup
from gensim.corpora import Dictionary
from PyPDF2 import PdfReader


def load_arxiv_metadata(path):

    reader = PdfReader(path)
    page = reader.pages[19]
    text = (page.extract_text())

    page2 = reader.pages[18]
    text2 = (page.extract_text())
    
    return text.replace("\n",""),text2.replace("\n","")

def extract_abstracts(metadata,
                      categories=["cs.AI", "cs.GT", "cs.CV", "cs.IR",
                                  "cs.LG", "cs.MA", "cs.NE", "stat.ML",
                                  "stat.ME", "stat.CO", "stat.TH"]):
    """Extract paper abstracts from arXiv metadata by category."""
    abstracts = []
    for item in metadata:
        paper = json.loads(item)
        for category in categories:
            if category in paper["categories"]:
                title = str(paper["title"])
                abstract = str(paper["abstract"])
                text = title + " " + abstract
                abstracts.append(text)
                break
    return abstracts


def export_documents(documents, filepath="documents.txt"):
    """Export documents to txt."""
    with open(filepath, "w") as fp:
        fp.write(json.dumps(documents))


def load_documents(filepath="documents.txt"):
    """Load documents from txt."""
    with open(filepath, "r") as fp:
        documents = json.loads(fp.read())
        return documents


def export_object(obj, filepath):
    """Export Python object."""
    pickle.dump(obj, open(filepath, "wb"))


def load_object(filepath):
    """Load saved Python object."""
    obj = pickle.load(open(filepath, "rb"))
    return obj


def create_directory_if_not_exists(dir_path):
    """Create directory if it does not exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def scrape_arxiv_abstract(paper_url):
    """Scrape arXiv abstract from url."""
    try:
        page = requests.get(paper_url)
        soup = BeautifulSoup(page.content, "html.parser")
        abstract = soup.find("blockquote", {"class": "abstract mathjax"})
        title = soup.find("h1", {"class": "title mathjax"})
        abstract.span.decompose()
        title.span.decompose()
        return title.text + "\n" + abstract.text

    except Exception as e:
        print(e)
        raise
