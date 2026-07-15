import pandas as pd
signals = dict()
news = []
signal_keywords = {
    "layoff_mentions": [
        "layoff", "layoffs", "cuts", "workforce", "job cuts",
        "staff reduction", "headcount reduction", "restructuring"
    ],
    "hiring_freeze_mentions": [
        "hiring freeze", "pauses hiring", "slows hiring", "recruitment freeze"
    ],
    "executive_exit_count": [
        "ceo resigns", "cfo resigns", "executive leaves", "steps down",
        "departure of", "chief operating officer leaves"
    ],
    "lawsuit_mentions": [
        "lawsuit", "sued", "class action", "legal claim", "court case"
    ],
    "regulatory_mentions": [
        "regulator", "investigation", "compliance", "antitrust", "inquiry"
    ],
    "customer_complaints": [
        "customers complain", "backlash", "poor support", "complaints",
        "users report"
    ],
    "outage_mentions": [
        "outage", "service disruption", "platform down", "downtime"
    ],
    "financial_stress_mentions": [
        "misses earnings", "lowers guidance", "cash runway", "liquidity",
        "widening losses", "revenue slowdown", "slowing demand"
    ],
    "positive_growth_mentions": [
        "expansion", "strong demand", "raises guidance", "partnership",
        "new markets", "record revenue"
    ]
}
def detect_signals(text):
    text = text.lower()
    result = {}

    for signal, keywords in signal_keywords.items():
        result[signal] = 0

        for keyword in keywords:
            if keyword in text:
                result[signal] = 1
                break

    return result


rows = []

for item in news:
    signals = detect_signals(item["text"])

    rows.append({
        "date": item["date"],
        "company": item["company"],
        "text": item["text"],
        **signals
    })

df = pd.DataFrame(rows)

df["date"] = pd.to_datetime(df["date"])
df["week"] = df["date"].dt.isocalendar().week

negative_words = [
    "cut", "cuts", "decline", "loss", "losses", "lawsuit", "investigation",
    "resigns", "outage", "complaints", "slowdown", "collapse", "risk",
    "restructuring", "slowing demand"
]

positive_words = [
    "growth", "expansion", "partnership", "profit", "strong", "record",
    "raises guidance", "new markets", "demand"
]


def sentiment_score(text):
    text = text.lower()

    negative_count = sum(word in text for word in negative_words)
    positive_count = sum(word in text for word in positive_words)

    return positive_count - negative_count


df["sentiment_score"] = df["text"].apply(sentiment_score)
df["is_negative"] = (df["sentiment_score"] < 0).astype(int)

