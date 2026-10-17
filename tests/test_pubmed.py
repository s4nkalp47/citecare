"""PubMedAbstractReader tests against a fake NCBI server (httpx.MockTransport)."""

import httpx

from src.plugins.medrag.pubmed import PubMedAbstractReader

EFETCH_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
          <Title>The Lancet</Title>
        </Journal>
        <ArticleTitle>Blood pressure lowering in <i>type 2</i> diabetes</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Hypertension is common.</AbstractText>
          <AbstractText Label="RESULTS">ACE inhibitors reduced events.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Meta-Analysis</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def make_reader(requests: list[httpx.Request]) -> PubMedAbstractReader:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, json={"esearchresult": {"idlist": ["111"]}})
        return httpx.Response(200, content=EFETCH_XML)

    reader = PubMedAbstractReader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    reader.request_delay = 0
    return reader


def test_reader_parses_abstract():
    docs = make_reader([]).load_data("hypertension diabetes", max_results=1)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["pmid"] == "111"
    assert doc.metadata["title"] == "Blood pressure lowering in type 2 diabetes"
    assert doc.metadata["journal"] == "The Lancet"
    assert doc.metadata["year"] == "2021"
    assert doc.metadata["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert "BACKGROUND: Hypertension is common." in doc.text
    assert "RESULTS: ACE inhibitors reduced events." in doc.text


def test_reader_searches_pubmed_by_relevance_with_filter():
    requests: list[httpx.Request] = []
    make_reader(requests).load_data("hypertension", extra_filter="AND english[lang]")

    search = requests[0].url.params
    assert search["db"] == "pubmed"
    assert search["sort"] == "relevance"
    assert search["term"] == "(hypertension) AND english[lang]"


def test_reader_returns_empty_when_no_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})

    reader = PubMedAbstractReader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    reader.request_delay = 0
    assert reader.load_data("nothing matches") == []
