import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH")
TOP_K = 3

NEWS_URLS = [
     "https://edition.cnn.com/sport/",
     "https://www.news.com.au/sport/olympics",
     "https://olympics.com/en/paris-2024/news",
     "https://olympics.com/en/paris-2024/medals",
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
