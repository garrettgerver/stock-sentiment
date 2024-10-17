import sys
import feedparser
import requests
from xml.sax import SAXParseException
from requests.exceptions import MissingSchema
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

if len(sys.argv) < 2:
    raise AssertionError("No arguments passed")

def main():
    model = BertForSequenceClassification.from_pretrained(
            "ahmedrachid/FinancialBERT-Sentiment-Analysis",
            num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(
            "ahmedrachid/FinancialBERT-Sentiment-Analysis", 
            clean_up_tokenization_spaces=True)
    nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

    tickers = [x.upper() for x in sys.argv[1:]]
    for ticker in tickers:
        visited_links = []
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
        try:
            feed = feedparser.parse(rss_url)
            if feed.status != 200:
                if feed.status == 404:
                    print(f"No ticker {ticker}. Proceeding.. ")
                else:
                    print(f"Unknown error for {ticker}. Proceeding.. ")
                continue
        except SAXParseException:
            print(f"Error parsing RSS feed for {ticker}. Proceeding.. ")
            continue

        for link in feed.entries:
            title = link.title
            desc = link.description
            url = link.link
            if url in visited_links:
                continue
            visited_links.append(url)
            try:
                req = requests.get(url)
            except MissingSchema:
                print(f"Article not found. Proceeding.. ")
                continue
            if req.status_code != 200:
                if req.status_code == 404:
                    print(f"Article not found. Proceeding.. ")
                else:
                    print(f"Unknown error. Proceeding..")
                continue
           
            print(title)
            text = ""
            sentiments = 0
            soup = BeautifulSoup(req.content, "html.parser")
            for para in soup.find_all("p"):
                text = BeautifulSoup(str(para), "lxml").text
                sentiment_list = nlp(text)
                sentiment = (
                        sentiment_list[0]["score"] * 
                        (1 if sentiment_list[0]["label"]=="positive" else
                        -1 if sentiment_list[0]["label"]=="negative" else 0)
                )
                sentiments += sentiment
            print(sentiments)

if __name__ == "__main__":
    main()

