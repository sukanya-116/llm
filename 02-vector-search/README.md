# Vector Search

In module 1 we used keyword search with minsearch and sqlitesearch.
It matches exact words. If you search for "Docker", the document has
to contain "Docker" to come back.

But look at these two questions:

- "Can I still join the course after the start date?"
- "Is it possible to enroll late?"

They mean the same thing, yet they share almost no words. A keyword
engine struggles to match them. We need something that works on
meaning, not on the exact words.

That something is vector search. Instead of matching words, it matches
ideas.

## The vector search process

We run vector search in two stages.

1. Offline (indexing): we convert all documents into vectors (arrays
   of numbers) and store them in an index.
2. Online (querying): we convert the user's query into a vector with
   the same model, then find the closest document vectors by similarity.

An embedding model produces these vectors. It's a neural network
trained to capture meaning, so texts that mean similar things land on
similar vectors. We measure how close two vectors are with a distance
metric. The most common one is cosine similarity.

Cosine similarity measures the angle between two vectors:

- Vectors pointing in the same direction: similarity close to 1
  (similar)
- Vectors at right angles: similarity close to 0 (unrelated)
- Vectors pointing in opposite directions: similarity close to -1
  (opposite meaning)

The larger the cosine similarity, the more similar the two texts are
in meaning.

## Keyword search vs vector search

Here's how the two approaches differ:

- Keyword search matches exact words. Vector search matches meaning.
- Keyword search suits specific terms, IDs, and names. Vector search
  suits paraphrased questions and natural language.
- Keyword search example: "pandas dataframe". Vector search example:
  "How do I work with tabular data?"
- Keyword search uses an inverted index (BM25, TF-IDF). Vector search
  uses a vector index based on cosine similarity.
- Keyword search misses synonyms and paraphrases. Vector search misses
  exact term matches.

In practice the two work best together. Hybrid search combines them.

## Building vector search

We'll take the same FAQ dataset from module 1 and build vector search
with three tools:

1. minsearch - in-memory vector search (simplest, good for
   experiments)
2. sqlitesearch - persistent vector search backed by SQLite
   (production-friendly, same API as minsearch)
3. PGVector - vector search in PostgreSQL (scalable, runs in
   Docker)

Then we'll plug vector search into our RAG pipeline.

## Prerequisites

In module 1 we set up a project with several libraries. Here we also
install sentence-transformers. It pulls in PyTorch and is heavy.
```bash
mkdir 02-vector-search
cd 02-vector-search
uv init
uv add requests minsearch openai groq jupyter python-dotenv sqlitesearch
```

You also need a `.env` file with your API key. 

---
# Embeddings

Before we can do vector search, we need to turn our text into vectors.
We call this process embedding: we embed text into a vector space. The
vectors we get back are also called "embeddings."

## Word embeddings and sentence embeddings

This idea comes from
[word2vec](https://en.wikipedia.org/wiki/Word2vec). The model learns to
place words as points in a multi-dimensional space. Words with similar
meanings land close to each other.

Imagine a 2D space where "enroll" and "join" are near each other and
"Docker" is far away:

```text
        · enroll
       · join
                   · Docker
```

The same idea works for entire sentences:

```text
Q1: "I just discovered the course. Can I still join it?"
Q2: "I just found out about the program. Can I still enroll?"

These two are close - they mean the same thing.

Q3: "How do I run Docker on Windows?"

This one is far away from Q1 and Q2.
```

Now imagine all 1200 documents in our FAQ dataset. Each one becomes a
point in this space. When a user asks a question, we embed it into the
same space and find the closest documents. Those nearest neighbors are
our search results.

The model encodes the whole sentence, not the words in isolation. So it
can tell apart the same word in different contexts.

Take the word "judge." In "the judge ruled out the possibility of crime"
(legal) it gets one vector. In "LLM-as-a-judge approach to evaluate
LLMs" (ML evaluation) it gets a different one. The surrounding context
changes the embedding.

So an embedding model takes text in and returns a fixed-length array of
numbers. We train it so that texts with similar meanings get similar
vectors.

We'll use [sentence-transformers](https://www.sbert.net/), a popular
open-source library for embeddings. It runs locally on your machine, so
there are no API costs.

## Installing sentence-transformers

Install the library:

```bash
uv add sentence-transformers
```

This also pulls in PyTorch under the hood, so it downloads a lot. You'll
see CUDA and other Nvidia packages go by. 

## Choosing a model

Sentence-transformers supports many models. The right one depends on
your task, your language, and the resources you have. Larger models are
usually slower, so for our FAQ dataset of short English texts a small
model is enough. Try a few on your own data and keep the one that works
best.

We'll use `all-MiniLM-L6-v2`:

- 384-dimensional vectors (compact)
- Fast on CPU
- Good quality for general English text
- Uses cosine similarity (we'll explain this below)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

The first time you run this, it downloads the model (~80 MB) and the
tokenizer from HuggingFace. The tokenizer turns text into something the
model can read. After that, both load from a local cache.

## Trying it with simple examples

Let's see how embeddings work on a few examples.

We'll start with a query:

```python
q1 = "Can I still join the course after the start date?"
v1 = model.encode(q1)
```

`v1` is a vector, an array of 384 numbers. Each number stands for some
concept the model learned. We can't read off what any one of them means.
But two vectors with similar values point to texts about similar things.

Encode our document:

```python
d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
dv = model.encode(d)
```

Next, we compare the query against the document using dot product:

```python
v1.dot(dv)
```

We get 0.32.

Now we try an unrelated query:

```python
q2 = "How to install Docker on Windows?"
v2 = model.encode(q2)
```

This time the similarity with the document should be much smaller:

```python
v2.dot(dv)
```

And we get 0.01.

The first score for `q1` vs `d` (0.32) is higher, so that query is more
similar to the document about registration. The second score for `q2`
vs `d` sits near 0, because installing Docker has nothing to do with
registration. A score near 0 means the two vectors are about as
different as they can be.

That's the whole idea behind vector search: similar texts get similar
vectors, and a dot product tells us how similar.

## Cosine similarity

The `all-MiniLM-L6-v2` model outputs normalized vectors - vectors with
unit length. When both vectors are normalized, the dot product equals
cosine similarity. That's why the model documentation says it "uses
cosine similarity."

Cosine similarity measures the angle between two vectors, ignoring
their length:

- 1.0 = same direction (similar)
- 0.0 = perpendicular (unrelated)
- -1.0 = opposite direction (opposite meaning)

Formally, if `theta` is the angle between two vectors, cosine similarity
is `cos(theta)`:

- `cos(0) = 1` - vectors point in the same direction
- `cos(90) = 0` - vectors are perpendicular
- `cos(180) = -1` - vectors point in opposite directions

Because our vectors are normalized, the dot product gives us cosine
similarity directly. This is why we can use `v1.dot(dv)` to compare
texts.

In practice, we rarely get cosine similarity below 0. The embedding
model maps text to a region of the vector space where most vectors
have positive components. There's no concept of "opposite meaning"
that maps to a vector pointing the other way.

---

# Embedding Our Dataset

## Loading the data

In [module 1](../01-agentic-rag/) we created [`ingest.py`](../01-agentic-rag/ingest.py) for loading the
FAQ data.

Copy into your project 

```python
from ingest import load_faq_data

documents = load_faq_data()
```

## Generating embeddings

Each document is a Python dictionary with a question and an answer. We
embed both together. That way a query can match against the question
text and the answer text in our index.

Build one text per document:

```python
texts = []

for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)
```

Now we generate the embeddings. We have about 1200 texts here. We won't
hand the model all of them at once. That takes a long time, and we can't
see what's happening inside. Instead we split them into batches.

First we import `tqdm` so we can watch the progress:

```python
from tqdm.auto import tqdm
```

Next we chunk the dataset into batches of 50 and encode each batch:

```python
batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

len(vectors)
```

We end up with 1208 vectors. On a GPU this is fast. Most of us run on
Codespaces without a GPU, so it takes a bit, but it's a one-off.

We turn them into a 2-dimensional array (matrix) where

- rows are documents (vectors)
- columns are dimensions of the vectors

```python
import numpy as np
X = np.array(vectors)
```

Calling `X.shape` returns (1350, 384) - number of documents vs number of dimensions.

---
# Vector Search

In the previous lesson we embedded our FAQ dataset into a matrix `X`
with 1350 document vectors. Here we see how vector search works under
the hood.

## Scoring documents

We have a matrix `X` with all document embeddings. We take a query,
compare it against every document, and keep the most similar ones.

When a query comes in, we embed it:

```python
query = "Can I still join the course after the start date?"
v_query = model.encode(query)
```

Next, we compute the dot product against all documents:

```python
scores = X.dot(v_query)
```

This is matrix-vector multiplication. Each element `i` of `scores` is
the cosine similarity between document `i` (row `i` of `X`) and
`v_query`.

We could compute the same thing with a for loop:

```python
scores = [v_query.dot(X[i]) for i in range(len(X))]
```

But `X.dot(v_query)` is much faster. Numpy runs optimized C code instead
of looping in Python, so the matrix version is hard to beat. The outcome
is the same either way: one score per document.

## Best match

The highest score is the most similar document:

```python
idx = np.argmax(scores)
idx, scores[idx]
```

This returns document 2 with score 0.76.

Let's see which document it is:

```python
documents[idx]
```

## Top 5 results

Usually we want more than the single best match, so let's pull the top
5.

`np.argsort` sorts from lowest to highest, so the last 5 are the top
ones:

```python
top5 = np.argsort(scores)[-5:]
```

They come out smallest-first, so we reverse them to get the highest
first:

```python
top5 = top5[::-1]
top5
```

Now we can read off the top 5 scores:

```python
scores[top5]
```

There's a shorter trick. We can negate the scores
first, so the largest becomes the smallest. Then `argsort` puts the best
matches at the front.

Here it is in one line:

```python
top5 = np.argsort(-scores)[:5]
```

It looks cryptic the first time you see it. But it's a common way to
turn a min-sort into a max-sort.

Let's read off the actual documents:

```python
for idx in top5:
    print(scores[idx])
    print(documents[idx])
    print()
```

This is vector search in its simplest form. We embed the query, compute
dot products against all documents, and return the highest-scoring ones.

We return 5 and not the single best for a reason. The answer to a
question can be spread across several documents. One holds part of it,
another fills in the rest. Sometimes the top result isn't the right one
but the second is. We send all 5 to the LLM and let it combine them.

The number 5 is a starting point, picked on gut feeling. Later, when we
evaluate search quality, we can test whether 3 or 10 works better for
our data.

Doing this by hand with numpy is fine for a small dataset. A larger one
needs a library that also handles filtering and ranking. 

---
# Vector Search with minsearch

In the previous section we did vector search by hand with numpy. We
embedded the query, computed dot products, and found the best matches.
Writing the argsort and matrix code every time gets old, and it can't
filter by course. So instead we'll use a library that wraps all of it.

We'll use [minsearch](https://github.com/alexeygrigorev/minsearch), the
small in-memory search library we already used in module 1 for text
search. It has a `VectorSearch` class for vector search.

Both classes share the same API:

- `fit` to index data
- `search` to query
- `filter_dict` in `search` to filter by keyword

It's the simplest way to get started with vector search.

## Creating the index

We already have our documents and vectors from the previous section.

Index them:

```python
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, documents)
```

We pass the numpy array `X` with all embeddings and the list of
documents as payload. The `keyword_fields` parameter works the same as
in the text `Index`, so we can filter by course later.

## Searching

Let's search for a question:

```python
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vindex.search(query_vector, num_results=5)
```

Under the hood it does the same thing we just did by hand. It computes
the dot product between each vector (after filtering) and our query
vector.

Look at the top result:

```python
results[0]
```

It should return the document about joining the course late:

## Filtering by course

Like the text index, we can filter by keyword fields. This matters for
user experience. A student in LLM Zoom Camp doesn't care about answers
from the data engineering course. So we narrow to their course first,
then score only within it.

Pass a `filter_dict`:

```python
results = vindex.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

Now that we can run vector search, let's use it in RAG.

---
# RAG with Vector Search

In module 1, we built a RAG pipeline with three steps:

```python
def rag(question):
    search_results = search(question)
    user_prompt = build_prompt(question, search_results)
    return llm(user_prompt)
```

The search step used keyword search. Now we swap in vector search.
Because RAG is modular, search is the only step we touch. Build prompt
and the LLM call stay exactly as they were.

## Using RAGBase

In [module 1](../01-agentic-rag/) we put all the RAG logic into a
[`RAGBase`](../01-agentic-rag/code/rag_helper.py) helper class. It
has `search`, `build_prompt`, and `llm` methods, so we only need to
override `search`.

Download `rag_helper.py` into your project


First, create the Groq client:

```python
from dotenv import load_dotenv
from groq import Groq
import os
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
```

Next, download and index the data:

```python
from ingest import load_faq_data, build_index

documents = load_faq_data()
index = build_index(documents)
```

Then use the `RAGBase` class:

```python
from rag_helper import RAGBase

assistant = RAGBase(
    index=index,
    llm_client=client,
)
```

Ask it a question:

```python
query = "I just found out about the program, can I still sign up?"
assistant.rag(query)
```

This still uses keyword search. Text search isn't bad here, so the
answer may already look right. Next we replace search with vector
search.

We already have:

- All the indexed documents `documents`
- The embeddings matrix `X` with all these documents
- The vector search engine `vindex`

We can't pass `vindex` to RAG as-is. Text search takes the query string
directly, but vector search needs the query as a vector first. So we 
subclass `RAGBase` and override `search` to encode the query before
searching.

The subclass overrides `search`:

```python

class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )
```

The `__init__` method adds one extra argument, `embedder`, for the
sentence transformer. Inside `search` we use it to turn the query into a
vector. Then we query `vindex` with that vector instead of the raw text.
Everything else is inherited from `RAGBase`.

## Using it

Let's init it:

```python
vector_assistant = RAGVector(
    embedder=model,
    index=vindex,
    llm_client=openai_client,
)
```

Try it with different queries:

```python
vector_assistant.rag("the program has already begun, can I still sign up?")
```

The answers should be close to what we got with keyword search, but
vector search handles rephrased questions better. The swap was trivial
because RAG has three clear steps. The same trick lets us change the LLM
provider later by overriding just the `llm` step.

---

# Vector Search with sqlitesearch

In the previous section we used minsearch for vector search.

It works, but it has three problems:

- It rebuilds the index on every startup
- It keeps everything in memory
- It searches by brute force


With text search we never felt these. Indexing was fast because we
didn't embed anything. With vector search, indexing runs a neural
network over every document, so it takes a minute on our dataset.
Keeping everything in memory is fine here, but a larger dataset would
need too much space.

The third problem is brute-force search. For every query we compare the
query vector against every single document. With 1,000 documents this is
fine, probably even faster than anything smarter. But as the dataset
grows past 10,000 or so, it gets slow, and we'll want an approximate
method instead.

What we've done so far is exact nearest neighbor (NN) search. We score
the query against every document and pick the top ones. It always finds
the true top matches, but it pays for that by touching everything.

Approximate nearest neighbor (ANN) search takes a shortcut. Instead of
comparing against everything, it first narrows down to a region of
likely matches. Then it scores only within that region. It may miss the
absolute best match, but the results are still good and it's much
faster.

```text
NN (exact):    compare query against ALL documents -> top 5
ANN (approx):  narrow down to a region -> compare within region -> top 5
```

## sqlitesearch

sqlitesearch is the persistent sibling of minsearch, and it solves both
problems at once.

We already used it in module 1 for persistent text search. It also does
vector search through its `VectorSearchIndex` class. It stores vectors
in SQLite, a real on-disk database, and uses ANN strategies for
retrieval. Because the data lives on disk, one process can write the
vectors and another can read them back.


## Creating the index

Initialize it:

```python
from sqlitesearch import VectorSearchIndex

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)
```

sqlitesearch supports three ANN modes:

- `lsh` (default): up to 100K vectors, random hyperplane projections
- `ivf`: 10K-500K vectors, K-means clustering
- `hnsw`: 10K-1M+ vectors, proximity graph (highest recall)

For our small dataset, `lsh` is fine. All modes use two-phase search:
approximate candidate retrieval, then exact cosine similarity
reranking.

## Indexing the data

Fit the index with our vectors and documents:

```python
vs_index.fit(vectors, documents)
```

The index is saved to `faq_vectors2.db`. Unlike minsearch, this file
persists on disk. You can search immediately after indexing, or reopen
the index later without re-indexing.

## Searching

Search works the same way as with minsearch. We always encode the query
into a vector first. This is one thing that makes vector search heavier
than text search. With text search we'd throw the raw query straight at
the engine.

Encode, then search:

```python
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vs_index.search(query_vector, num_results=5)
```

Look at the results:

```python
results
```

## Filtering by course

Filtering works the same way:

```python
results = vs_index.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

## Closing the connection

When you're done with the index:

```python
vs_index.close()
```


## Reopening the index

In a new Python session, you can reopen the index without re-computing
embeddings:

```python
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

model = SentenceTransformer("all-MiniLM-L6-v2")

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)
```

Now we can search:

```python
query_vector = model.encode("How do I run Kafka?")
results = vs_index.search(query_vector, num_results=5)
```

We still load the embedding model to encode the query, but we don't
re-embed all the documents. No `fit` call needed, because the index is
already built and waiting on disk.

This is the same two-process split we used for text search in module 1.
One process ingests and builds the index, another queries it.

It matters more here than with text search. Embedding the whole dataset
takes about a minute. We don't want a user waiting that long when the
app starts up. We pay that cost once during ingestion, and the query
side starts up instantly.

## Using sqlitesearch vector search in RAG

Let's use our persistent vector index in RAG.

In a new notebook, set up the model and open the index (same as
the "Reopening the index" section above):

```python
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

model = SentenceTransformer("all-MiniLM-L6-v2")

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)
```

We'll use the `RAGVector` class. It overrides the `search` method
to embed the query and use vector search.

Set up the GROQ client and create the assistant:

```python
from rag_helper import RAGBase
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )

vector_assistant = RAGVector(
    embedder=model,
    index=vs_index,
    llm_client=openai_client,
)
```

Try it:

```python
vector_assistant.rag("the program has already begun, can I still sign up?")
```

When you're done, close the connection:

```python
vs_index.close()
```

## Comparing minsearch and sqlitesearch for vector search

Here is how the two compare:

- minsearch `VectorSearch`: in-memory (numpy), exact cosine similarity,
  must re-compute embeddings on startup, good for experiments and
  notebooks
- sqlitesearch `VectorSearchIndex`: persistent (SQLite `.db` file), ANN
  (LSH/IVF/HNSW) with exact rerank, can open an existing index, good
  for projects and persistence

Its only dependencies are SQLite and numpy. So it runs on any host that 
offers a free SQLite database, where a dedicated vector database would cost 
extra.

