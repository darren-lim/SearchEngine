# Search Engine

Created By Darren Lim, Paul Nguyen, Thu Nguyen, Mehrdad Ekbatani

### Description
Basic indexing documentation and search engine implementation as a class project. The corpus used for indexing in from a subsection of websites taken from uci.edu. The search engine will search words from an input and parse through the indexed websites. It returns websites based on relevance to the words inputted. Numbers, punctuation, and stop words are removed from the search query. The search engine will take words from the query and look through a file that contains words and its corresponding documents (websites). It will match the query with words in that file. After going through an algorithm that uses tf-idf (term frequency inverse document frequency) scoring, it returns documents that are relevant to the query, and shows the website URLs to the user.

### How To Run
The Corpus is unavailable, therefore index creation is unavailable. However, the search engine is runnable. Pull the repo, and run the gui.py file to run the search engine.

### Implementation
Search Engine

- tf-idf scoring
- cosine similarity
- duplication detection
- inverted index creation
- threading
- word stemming
- gui interface
- response time mostly < 300ms with short queries
