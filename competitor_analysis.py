#!/usr/bin/env python
# coding: utf-8

# In[28]:


# import asyncio
# import json
# from playwright.async_api import async_playwright

# async def get_reviews():
#     reviews = []

#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=False)  # Set to True to run headless
#         context = await browser.new_context()
#         page = await context.new_page()
#         await page.goto("https://www.coursereport.com/schools/le-wagon", wait_until="networkidle")

#         page_num = 1
#         while True:
#             print(f"📄 Scraping page {page_num}...")

#             # Wait for reviews to load
#             await page.wait_for_selector("ul.divide-y.divide-gray.divide-solid", timeout=10000)

#             # Get review blocks
#             review_blocks = page.locator("ul.divide-y.divide-gray.divide-solid > li.py-6.space-y-4")
#             count = await review_blocks.count()
#             print(f"📝 Found {count} reviews")

#             for i in range(count):
#                 review = review_blocks.nth(i)
#                 try:
#                     # Click 'Read More' if it exists
#                     read_more = review.locator("a:text('Read More')")
#                     if await read_more.count() > 0:
#                         await read_more.click()
#                         await asyncio.sleep(0.2)  # slight delay to allow content to expand

#                     # Get full review text
#                     text_el = review.locator("div.text-gray-dark.break-words")
#                     text = await text_el.inner_text()

#                     # Get review date
#                     date_el = review.locator(
#                         "div.flex.text-sm.leading-relaxed.justify-between.gap-2 >> "
#                         "div.text-gray-medium.flex-shrink-0")

#                     date = await date_el.inner_text()

#                     reviews.append({
#                         "date": date.strip(),
#                         "text": text.strip()})

#                 except Exception as e:
#                     print(f"❌ Error extracting review #{i}: {e}")
#                     continue

#             # Stop after 5 pages
#             if page_num >= 5:
#                 print("⛔ Reached max page limit (5).")
#                 break

#             # Try clicking the "Next" button
#             next_button = page.locator("#reviews span.page.next a[rel='next']")
#             if await next_button.count() > 0:
#                 print("➡️ Moving to next page...")
#                 await next_button.first.click()
#                 await page.wait_for_selector("ul.divide-y.divide-gray.divide-solid", timeout=10000)
#                 page_num += 1
#                 await asyncio.sleep(1)
#             else:
#                 print("✅ No more pages.")
#                 break

#         await browser.close()
#     return reviews

# async def main():
#     data = await get_reviews()
#     with open("le_wagon_reviews_full.json", "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     print(f"💾 Saved {len(data)} reviews to le_wagon_reviews_full.json")

# await main()


# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import asyncio
import json
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import gensim
from playwright.async_api import async_playwright
plt.style.use('ggplot')
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from deep_translator import GoogleTranslator
import re
from selenium import webdriver
from langdetect import detect, DetectorFactory
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
sia = SentimentIntensityAnalyzer()


# In[ ]:


import pandas as pd
import os
downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
os.chdir(downloads_path)
bootcamps = pd.read_csv('bootcamps.csv', index_col=0)
bootcamps = bootcamps.apply(pd.to_numeric, errors="coerce").astype("Int64")
bootcamps = bootcamps.fillna(0)


# Coursereport

# In[30]:


# async def get_reviews():
#     reviews = []

#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=False)  # Change to True if you want headless
#         context = await browser.new_context()
#         page = await context.new_page()
#         await page.goto("https://www.coursereport.com/schools/le-wagon", wait_until="networkidle")

#         page_num = 1
#         while True:
#             print(f"📄 Scraping page {page_num}...")

#             # Wait for reviews list to load
#             await page.wait_for_selector("ul.divide-y.divide-gray.divide-solid", timeout=10000)

#             # Get review blocks
#             review_blocks = page.locator("ul.divide-y.divide-gray.divide-solid > li.py-6.space-y-4")
#             count = await review_blocks.count()
#             # print(f"📝 Found {count} reviews")

#             for i in range(count):
#                 review = review_blocks.nth(i)
#                 try:
#                     # Click 'Read More' if present to expand full text
#                     read_more_button = review.locator("button:text('Read More')")
#                     if await read_more_button.count() > 0:
#                         await read_more_button.first.click()
#                         await asyncio.sleep(0.3)

#                     content_blocks = review.locator("div.text-gray-dark.break-words > div[data-toggle-target='content']")
#                     count_contents = await content_blocks.count()
#                     # print(f"Review #{i} has {count_contents} content blocks")

#                     text = ""

#                     if count_contents > 1:
#                         text = await content_blocks.nth(1).inner_text()
#                         if not text.strip():
#                             text = await content_blocks.first.inner_text()
#                     elif count_contents == 1:
#                         text = await content_blocks.first.inner_text()
#                     else:
#                         # Fallback to main container text
#                         text_el = review.locator("div.text-gray-dark.break-words")
#                         text = await text_el.inner_text()

#                     if not text.strip():
#                         print(f"⚠️ Empty review text for review #{i}, skipping.")
#                         continue

#                     # Extract date
#                     date_el = review.locator("div.flex.text-sm.leading-relaxed.justify-between.gap-2 >> div.text-gray-medium.flex-shrink-0")
#                     date = await date_el.inner_text()

#                     # Extract bootcamp info if available
#                     bootcamp_info_el = review.locator("div.flex-grow > div.flex.text-gray-medium.space-x-4 > span")
#                     bootcamp_info = ""
#                     if await bootcamp_info_el.count() > 0:
#                         bootcamp_info = await bootcamp_info_el.inner_text()

#                     reviews.append({
#                         "date": date.strip(),
#                         "review": text.strip(),
#                         "bootcamp_info": bootcamp_info.strip(),
#                         "source": 'coursereport' # added now, might have to delete
#                     })

#                 except Exception as e:
#                     print(f"❌ Error extracting review #{i}: {e}")
#                     continue

#             # Stop after 5 pages (adjust as needed)
#             if page_num >= 5:
#                 print("⛔ Reached max page limit.")
#                 break

#             # Dismiss modal if present
#             modal_close_btn = page.locator("turbo-frame#get_matched_modal button, turbo-frame#get_matched_modal [data-action*='close']")
#             if await modal_close_btn.count() > 0:
#                 print("🧼 Dismissing modal...")
#                 try:
#                     await modal_close_btn.first.click()
#                     await asyncio.sleep(1)
#                 except Exception as e:
#                     print(f"⚠️ Could not dismiss modal: {e}")

#             # Click next page button if available
#             next_button = page.locator("#reviews span.page.next a[rel='next']")
#             if await next_button.count() > 0:
#                 print("➡️ Moving to next page...")
#                 await next_button.first.click()
#                 await page.wait_for_selector("ul.divide-y.divide-gray.divide-solid", timeout=10000)
#                 page_num += 1
#                 await asyncio.sleep(1)
#             else:
#                 print("✅ No more pages.")
#                 break

#         await browser.close()
#     return reviews

# async def main():
#     data = await get_reviews()
#     with open("lewagon_coursereport.json", "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     # print(f"💾 Saved {len(data)} reviews to lewagon_coursereport.json")

# await main()


# Trustpilot

# In[31]:


# options = webdriver.ChromeOptions()
# options.add_argument('--headless')
# options.add_argument('--no-sandbox')
# options.add_argument('--disable-dev-shm-usage')

# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# def extract_reviews_from_page():
#     soup = BeautifulSoup(driver.page_source, 'html.parser')
#     reviews = []

#     # Loop through each full review card (article)
#     review_cards = soup.find_all('article', attrs={'data-service-review-card-paper': True})

#     for card in review_cards:
#         # Review section inside the card
#         section = card.find('section', class_='styles_reviewContentwrapper__K2aRu')

#         # Review text
#         review_tag = section.find('p', attrs={'data-service-review-text-typography': True}) if section else None
#         review_text = review_tag.get_text(separator="\n").strip() if review_tag else 'No review text'

#         # Date
#         date_p = section.find('p', attrs={'data-service-review-date-of-experience-typography': True}) if section else None
#         date_span = date_p.find('span') if date_p else None
#         date_text = date_span.text.strip() if date_span else 'No date'

#         # ✅ Country code (correct scope now!)
#         country_span = card.find('span', attrs={'data-consumer-country-typography': True})
#         country_code = country_span.get_text(strip=True) if country_span else 'unknown'

#         if review_text != 'No review text' and date_text != 'No date' and country_code == 'DE':
#             reviews.append({
#                 'country_code': country_code,
#                 'review': review_text,
#                 'date': date_text,
#                 'source': 'trustpilot'
#             })
#         else:
#             continue
#             # print("Skipped empty or incomplete review")

#     return reviews

# from selenium.common.exceptions import NoSuchElementException

# def scrape_all_reviews():
#     all_reviews = []
#     base_url = 'https://de.trustpilot.com/review/lewagon.com'
#     page = 1

#     while True:
#         url = f"{base_url}?page={page}" if page > 1 else base_url
#         print(f"🔍 Loading {url}")
#         driver.get(url)
#         time.sleep(3)  # Wait for JS to load

#         reviews = extract_reviews_from_page()
#         print(f"📥 Page {page}: Found {len(reviews)} DE reviews.")
#         all_reviews.extend(reviews)

#         try:
#             next_button = driver.find_element("css selector", 'a[data-pagination-button-next-link="true"]')
#             # If found and visible, go to the next page
#             if 'disabled' in next_button.get_attribute('class'):
#                 print("⛔️ Next page button is disabled.")
#                 break
#         except NoSuchElementException:
#             print("⛔️ No next page button found.")
#             break

#         page += 1
#         # Optional limit for testing
#         # if page == 10:
#         #     break

#     return all_reviews

# all_reviews = scrape_all_reviews()
# driver.quit()

# with open('lewagon_trustpilot.json', 'w', encoding='utf-8') as f:
#     json.dump(all_reviews, f, ensure_ascii=False, indent=4)

# print(f"\n✅ Saved {len(all_reviews)} reviews to lewagon_trustpilot.json")


# In[32]:


df_google = pd.read_csv('lewagon_google.csv', names=['date', 'review', 'source'])
df_trustpilot = pd.read_json('lewagon_trustpilot.json')
df_coursereport_raw = pd.read_json('lewagon_coursereport.json')
def filter_bootcamp_info(df):
    keywords = ['online', 'Berlin', 'Germany']
    mask = df['bootcamp_info'].str.contains('|'.join(keywords), case=False, na=False)
    return df[mask]
df_coursereport = filter_bootcamp_info(df_coursereport_raw)
df_lewagon = pd.concat([df_coursereport, df_google, df_trustpilot], ignore_index=True)
df_lewagon


# In[33]:


def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
df_lewagon['review'].apply(clean_text)    


# In[34]:


DetectorFactory.seed = 0
df_lewagon['language'] = df_lewagon['review'].apply(lambda x: detect(x))
translator = GoogleTranslator(source='auto', target='en') 
def safe_translate(text):
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Translation failed for text: {text[:30]}... — Error: {e}")
        return None
df_lewagon['translated'] = df_lewagon.apply(lambda row: safe_translate(row['review']) if row['language'] != 'en' else None, axis=1)


# In[35]:


sia.polarity_scores("I am so happy") # {'neg': 0.0, 'neu': 0.334, 'pos': 0.666, 'compound': 0.6115}
sia.polarity_scores('I am very sad')['compound'] # {'neg': 0.629, 'neu': 0.371, 'pos': 0.0, 'compound': -0.5256}
# Compound goes from -1 to 1
# df['compound'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df_lewagon['compound'] = df_lewagon.apply(lambda row: sia.polarity_scores(row['translated'] if pd.notnull(row['translated']) else row['review'])['compound'],axis=1)


# In[36]:


import locale
locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
mask = (df_lewagon['country_code'] == 'DE') | (df_lewagon['source'] == 'google')
df_lewagon.loc[mask, 'date'] = pd.to_datetime(
    df_lewagon.loc[mask, 'date'],
    format='%d. %B %Y',
    errors='coerce')  # Optional: prevents crash on invalid dates
df_lewagon['date'] = pd.to_datetime(df_lewagon['date'], errors='coerce')


# In[37]:


df = df_lewagon.sort_values('date').reset_index(drop=True)
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['compound'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Compound Score')
plt.title('Compound Score Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[38]:


print(df.review[df['compound'] == df['compound'].min()].to_list())


# In[39]:


# remove stopwords, punctuation, and normalize the corpus
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

df_lewagon['cleaned'] = df.apply(lambda row: clean(row['translated']) if pd.notnull(row['translated']) else clean(row['review']), axis=1)
unwanted = {'le', 'wagon', 'bootcamp', 'lewagon', 'bootcamps', 'wagon’s'}
df_lewagon['cleaned'] = df_lewagon['cleaned'].apply(lambda x: " ".join([word for word in x.split() if word not in unwanted]))
df['tokens'] = df_lewagon['cleaned'].apply(lambda x: x.split())
# Creating document-term matrix 
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]


# In[ ]:


from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# === Train LDA Model ===
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    random_state=42,
    update_every=1,
    chunksize=10,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

# === Coherence Score for current model ===
coherence_model = CoherenceModel(model=lda_model, texts=df['tokens'], dictionary=dictionary, coherence='c_v')
coherence = coherence_model.get_coherence()
print(f'Coherence Score: {coherence:.4f}')
# Higher coherence → The top words in a topic are more related to each other (e.g., "apple", "banana", "grape", "fruit").
# Lower coherence → The words seem unrelated (e.g., "government", "banana", "code", "policy").

# === Coherence over different topic counts ===
coherence_values = []
for k in range(2, 10):
    model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=42)
    cm = CoherenceModel(model=model, texts=df['tokens'], dictionary=dictionary, coherence='c_v')
    coherence_values.append((k, cm.get_coherence()))

# Plot coherence vs number of topics
k_values, scores = zip(*coherence_values)
plt.plot(k_values, scores)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score (c_v)")
plt.title("Coherence Score by Number of Topics")
plt.show()

# === Print Top N Words per Topic (Clean Format) ===
topn = 5
threshold = 0.015

print("\nTop Words per Topic:")
topics = lda_model.show_topics(num_topics=3, num_words=topn, formatted=False)
for topic_id, terms in topics:
    print(f"\nTopic {topic_id}:")
    for word, weight in terms:
        if weight >= threshold:
            print(f"  {word}: {round(weight, 4)}")
# > 0.60	Very good — topics are highly coherent and interpretable.
# 0.50 – 0.60	Good — topics are generally coherent and useful.
# 0.40 – 0.50	Moderate — topics may be mixed or noisy. May need tuning.
# < 0.40	Low — topics are likely incoherent or too broad/noisy.

# === Visualization with pyLDAvis ===
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)


# In[46]:


competitors = ['Ironhack','Le Wagon','Stackfuel','Careerfoundry','Code institute', 'Masterschool','WBS coding school','Codeworks','Digital career institute']
courses = ['Web Development', 'Data Analytics', 'UX/UI Design', 'Digital Marketing', 'Data Science and ML', 'AI', 'DevOps & Cloud Computing', 'Cybersecurity']
features = pd.DataFrame(columns=competitors, index=courses)


# In[ ]:


import gensim
from gensim.utils import simple_preprocess
from gensim.models import Phrases, Phraser
from nltk.corpus import stopwords
import spacy
import nltk

# Normalization
# Download resources if needed
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

stop_words = set(stopwords.words('english'))

def preprocess_texts(texts):
    """
    texts: list of raw documents (strings)
    returns: list of tokenized, cleaned, lemmatized texts
    """
    # Tokenize and clean
    def tokenize(text):
        return [token for token in simple_preprocess(text, deacc=True) if token not in stop_words]

    tokenized_texts = [tokenize(doc) for doc in texts]

    # Build bigrams and trigrams
    bigram = Phrases(tokenized_texts, min_count=5, threshold=100)
    trigram = Phrases(bigram[tokenized_texts], threshold=100)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    texts_bigrams = [bigram_mod[doc] for doc in tokenized_texts]
    texts_trigrams = [trigram_mod[bigram_mod[doc]] for doc in texts_bigrams]

    # Lemmatize
    def lemmatize(doc):
        doc = nlp(" ".join(doc))
        return [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']]

    return [lemmatize(doc) for doc in texts_trigrams] # add

