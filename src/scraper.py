import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

years = range(2003, 2009)
all_text = []

for year in years:
    index_url = f"https://www.jaxa.jp/press/{year}/index_j.html"
    try:
        r = requests.get(index_url, timeout=10)
        if r.status_code != 200:
            print(f"Skipping {year}: status {r.status_code}")
            continue
        r.encoding = 'shift_jis'  # <-- force correct encoding
    except Exception as e:
        print(f"Skipping {year}: {e}")
        continue

    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "_j.html" in href:
            full_url = urljoin(index_url, href)
            try:
                page = requests.get(full_url, timeout=10)
                page.encoding = 'shift_jis'  # <-- also here
                page_soup = BeautifulSoup(page.text, "html.parser")
                body = page_soup.get_text(separator="\n", strip=True)
                all_text.append(body)
                print(f"  Scraped: {full_url}")
            except Exception as e:
                print(f"  Failed {full_url}: {e}")
            time.sleep(0.5)

with open("intro-to-nlp-project/src/ja.txt", "a", encoding="utf8") as f:
    f.write("\n".join(all_text))

print(f"Done. Total pages scraped: {len(all_text)}")