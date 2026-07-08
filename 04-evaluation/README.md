# Evaluation

In the previous modules, we built search engines and RAG pipelines.
We tried different approaches: keyword search with minsearch, vector
search, agents with function calling. But we never answered the obvious
question of which one is actually better.

We could try a few queries by hand and see what looks good. That's fine
for a quick sanity check, but it doesn't scale, and it doesn't give us a
number to compare. We need a systematic way to tell whether one approach
beats another.

That's what evaluation is for. And it's worth saying up front: of
everything in this course, evaluation is the part that matters most. It's
also the most tedious. But it's the only way to be sure your system
works. And it's how you keep it working as you change prompts and swap
models.

## The evaluation setup

For search evaluation, we need a dataset of questions where we know
which document is the correct answer. We'll use an LLM to generate
these questions from our FAQ data.

The approach works like this:

- A = the original answer in the FAQ
- Q* = a question generated from that answer by an LLM
- We send Q* through our search and check if the original document
  appears in the results

For RAG evaluation, we go one step further:

- A = the original answer in the FAQ
- Q* = a question generated from that answer by an LLM
- A' = the answer produced by our RAG system when given Q*
- We compare A' with A to see if the system produced the right answer

This is the A → Q* → A' pattern. We know the answer for each generated
question because we created the question from that answer.

With evaluation, we can:

- Compare different search methods (minsearch vs vector search vs hybrid)
- Tune parameters (boost values, number of results, prompt templates)
- Compare different LLMs (gpt-5.4-mini vs others)
- Track improvements over time

There are two types of evaluation:

- Offline evaluation: run the system on a test dataset and compute metrics
- Online evaluation: collect feedback from real users in production

Offline evaluation is what we do before putting changes in front of
users. It lets us compare search settings, prompts, or models on the
same dataset. Online evaluation happens after deployment. It uses real
traffic, feedback, logs, and dashboards to monitor quality.

In this module, we focus on offline evaluation. We'll generate a test
dataset, run our search and RAG systems on it, and measure how well they
perform.

Synthetic data is a good starting point when you don't have real user
data. But generated questions can be too similar to the original FAQ
text, which inflates the metrics. As soon as you can, start collecting
real user queries and use them to validate your evaluation framework.

We'll cover three levels of evaluation:

1. Search evaluation: does the search return the right documents?
2. RAG evaluation: does the LLM generate good answers?
3. Agent evaluation: does the agent use tools efficiently?

Most of our time goes to search, and that's on purpose. Everything else
depends on it: if retrieval brings back the wrong documents, no prompt or
model can rescue the answer. So we test search on its own first, then
evaluate the full pipeline on top of it.

For search, we'll use two metrics: Hit Rate and MRR (Mean Reciprocal
Rank). For RAG quality, we'll use LLM-as-a-judge. For agents, we'll
look at the final answer and the tool-call trajectory.

Let's start with generating the test data we need.

---

# Generating Ground Truth Data

To evaluate search, we need a dataset of queries where we know which
document is the correct answer. This is called ground truth (or gold
standard) data.

For each query in our ground truth dataset, we know which document in
the knowledge base is relevant. When we run a search, we check whether
the results include the correct document.

There are several ways to get ground truth data:

- Human annotators look at documents and write queries (best quality, expensive)
- Collect real user queries and label them (requires a running system)
- Generate synthetic data with an LLM (what we'll do)

We don't have a production system yet, so we'll use an LLM to generate
questions. For each FAQ document, we ask the LLM to create 5 questions
that this document would answer. Then we know that for each generated
question, the source document is the correct answer.

## Loading the documents

We'll use helper files from module 01 and this module.

If you don't have them in your notebook directory, download them:

```bash
/01-agentic-rag/ingest.py
/01-agentic-rag/rag_helper.py
evaluation_utils.py
```

Then load the FAQ data:

```python
from ingest import load_faq_data
documents = load_faq_data()
```

We'll generate questions only for the LLM Zoomcamp FAQ. The full FAQ
dataset contains documents from multiple courses. Generating five
questions for every document would take longer and cost more.

```python
documents_llm = []

for doc in documents:
    if doc["course"] == "llm-zoomcamp":
        documents_llm.append(doc)

len(documents_llm)
```

We'll use these documents from now on so let's name them as `documents`

```python
documents = documents_llm
```

Each document already has an `id` field:

```python
doc = documents[0]
print(doc["id"])
print(doc["question"])
print(doc["answer"])
```

The ID becomes the label in our ground truth dataset. We generate
questions from a document, so we know that this document holds the
answer. Later, search evaluation checks whether search brings back the
document with this ID.

This is why every record needs a stable ID. If you can't uniquely
identify a document, you can't tell whether search retrieved the right
one. When you build your own evaluation set, assign an ID to each record
in your knowledge base first.

## Generating questions with structured output

We use an LLM to generate questions for each document.

With structured output, we ask the LLM to return data in a specific
format instead of free-form text. For example, instead of getting a
paragraph that contains questions, we can ask for a Python object with
a `questions` field.

This is useful when code will process the output. The model returns the
same structure every time. We can access the generated questions
directly instead of parsing text manually.

We want the output as a list of strings, so we define that structure
with a Pydantic model:

```python
from pydantic import BaseModel

class Questions(BaseModel):
    questions: list[str]
```

The instructions for the LLM:

```python
data_gen_instructions = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
""".strip()

structured_instructions = data_gen_instructions + """
Return ONLY a valid JSON object with the following structure:
{
    "questions": ["question1", "question2", "question3", "question4", "question5"]
}
Do not include any other text, markdown, or formatting outside the JSON.
"""
```

We ask the LLM to use different wording from the original document.
This makes the evaluation more realistic - real users won't phrase
their questions the same way as the FAQ.

Call the LLM for one document:

```python
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

```

Prepare the document as JSON:

```python
import json

user_prompt = json.dumps(doc)
```

Create the messages:

```python
messages = [
    {"role": "developer", "content": data_gen_instructions},
    {"role": "user", "content": user_prompt}
]
```

Call the model:

```python
response = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",  
    messages=messages,
    response_format={"type": "json_object"},
    temperature=0.7,
)
```

Parse the response

```python
import json
result_text = response.choices[0].message.content
result_data = json.loads(result_text)
questions = result_data["questions"]
questions
```

You should see 5 questions that relate to the first FAQ document.

## Reusable utilities

`evaluation_utils.py` contains helper functions we'll reuse in this module:

- `llm_structured`: calls the Groq API with structured output
- `llm_structured_retry`: retries structured-output calls when a
  request fails
- `calc_price`: calculates the price from token usage
- `calc_total_price`: calculates the total price from multiple usage
  objects
- `map_progress`: runs work in parallel and tracks progress.

Import the structured-output helper:

```python
from evaluation_utils import llm_structured
```

Use it on the same document:

```python
result, usage = llm_structured(
    groq_client,
    structured_instructions,
    user_prompt,
    Questions
)

print(result.questions)
```

## Tracking cost

The response also contains token usage:

```python
usage.input_tokens, usage.output_tokens
```

As in the agents module, we calculate the price from `response.usage`.

Import the price helper:

```python
from evaluation_utils import calc_price
```

Calculate the cost of this call:

```python
cost = calc_price(usage)

cost
```

Now convert these questions into ground truth records:

```python
records = []

for q in result.questions:
    records.append({
        "question": q,
        "document": doc["id"]
    })

records
```

Each record has two fields:

- `question`: the question generated by the LLM
- `document`: the ID of the FAQ document that should answer the question

The `document` field connects the generated question to the document
that contains the answer. Later, when we evaluate search, we'll ask the
search engine the generated question. Then we'll check if it retrieves
the document with this ID.

We now know how to generate and store questions for one document. In
the next lesson, we'll run this for all LLM Zoomcamp FAQ documents and
save the full ground truth dataset.

