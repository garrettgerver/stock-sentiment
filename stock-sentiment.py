import json
import argparse
import feedparser
import requests
from xml.sax import SAXParseException
from requests.exceptions import MissingSchema
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

model = BertForSequenceClassification.from_pretrained(
    "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    num_labels=3
)
tokenizer = BertTokenizer.from_pretrained(
    "ahmedrachid/FinancialBERT-Sentiment-Analysis", 
    clean_up_tokenization_spaces=True
)
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

class ModelObject():
    def __init__(self):
        self._container: dict[str, [dict[str, float]]] = {}
        self._elements: dict[str, float] = {}

    @property
    def get_article_sentiments(self) -> dict[str, float]:
        return self._elements

    @property
    def get_ticker_sentiments(self) -> dict[str, [dict[str, float]]]:
        return self._container

    def add_article_sentiment(self, article: str, sentiment: float) -> None:
        self._elements.update({article: sentiment})

    def add_ticker_sentiment(self, ticker: str, sentiment: dict[str, float]) -> None:
        self._container.update({ticker: self._elements})
        self._elements = {}

def model(tickers: list[str]) -> ModelObject: 
    data = ModelObject()
    for ticker in tickers:
        visited_links = []
        match (args.source):
            case "Yahoo": rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
            case _: break #will be used in future for incorporating more websites
        try:
            feed = feedparser.parse(rss_url)
            if feed.status != 200:
                if feed.status == 404:
                    print(f"No ticker {ticker}. Proceeding.. \n")
                else:
                    print(f"Unknown error for {ticker}. Proceeding.. \n")
                continue
        except SAXParseException:
            print(f"Error parsing RSS feed for {ticker}. Proceeding.. \n")
            continue

        for link in feed.entries:
            title = link.title
            url = link.link
            if url in visited_links:
                continue
            visited_links.append(url)
            try:
                req = requests.get(url)
            except MissingSchema:
                print(f"Article not found. Proceeding.. \n")
                continue
            if req.status_code != 200:
                if req.status_code == 404:
                    print(f"Article not found. Proceeding.. \n")
                else:
                    print(f"Unknown error. Proceeding.. \n")
                continue
           
            total_sentiment = 0
            soup = BeautifulSoup(req.content, "html.parser")
            for para in soup.find_all("p"):
                text = BeautifulSoup(str(para), "lxml").text
                sentiment_list = nlp(text)
                sentiment = (
                        sentiment_list[0]["score"] * 
                        (1 if sentiment_list[0]["label"]=="positive" else
                        -1 if sentiment_list[0]["label"]=="negative" else 0)
                )
                total_sentiment += sentiment
            print(title)
            print(f"{total_sentiment}\n")
            data.add_article_sentiment(title, total_sentiment)
        data.add_ticker_sentiment(ticker, data.get_article_sentiments)
    return data.get_ticker_sentiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--source", choices=["Yahoo"], default="Yahoo", help="source of news articles"
    )
    parser.add_argument(
        "-f", "--file", help="file to write to (.txt, .json)"
    )
    parser.add_argument("ticker", nargs="+", help="ticker symbols")
    args = parser.parse_args()
    if args.file:
        if not(args.file.split(".")[-1] == "json" or "txt"):
            raise AssertionError("Invalid file format (.txt, .json)")

    tickers = [x.upper() for x in args.ticker] 
    data = model(tickers)
    if args.file:
        with open(args.file, "w") as f:
            json.dump(data, f, indent=4)
            print(f"Wrote data to {args.file}")
    print("Done")
