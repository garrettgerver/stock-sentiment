# Stock Sentiment Script in Python
This is a simple python script that uses a pretrained LLM model [FinancialBERT-Sentiment-Analysis](https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis) to calculate the general sentiment of a list of given stocks per the most recent Yahoo Finance articles about said stocks.

The current way to run this script, as per `python stock-sentiment -h`:
```
usage: stock-sentiment.py [-h] [-s {Yahoo}] [-f FILE] ticker [ticker ...]

positional arguments:
  ticker                ticker symbols

options:
  -h, --help            show this help message and exit
  -s {Yahoo}, --source {Yahoo}
                        source of news articles (default: Yahoo)
  -f FILE, --file FILE  file to write to (.txt, .json)
```

This script can also be imported as a module and returns the results as a JSON object.

Current points of development are timing out requests if they take too long and the ability to pass through alternative websites to Yahoo Finance that may potentially yield more acurate results, such as Barron's. 
