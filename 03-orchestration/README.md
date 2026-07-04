# AI Orchestration

AI-powered workflow orchestration using Kestra.

### Why AI for Workflows?

When building LLM applications and workflows, we often spend significant time writing boilerplate code, searching documentation, and structuring pipelines. AI tools can help us:

- Generate workflows faster: describe what you want in natural language instead of writing YAML from scratch
- Avoid errors: get syntax-correct, up-to-date code that follows best practices
- Automate complex decisions: let AI agents dynamically orchestrate tasks based on changing conditions
- Ground responses in data: use RAG to ensure AI provides accurate, contextual information

However, AI is only as good as the context we provide. This module teaches you how to engineer that context for reliable, production-ready workflows.

### What Makes This Different from AI Assistants?

Traditional AI assistants (like ChatGPT or Gemini in a browser) don't have context about your codebase and workflow patterns, real-time data from your systems, or the latest documentation and best practices.

By integrating AI directly into Kestra and using techniques like RAG and specialized agents, we can provide this context and get much better results.

See the [full list of supported providers](https://kestra.io/plugins/plugin-ai/provider)

---
## Context Enginnering

### Experiment: ChatGPT Without Context

1. Open ChatGPT in a private browser window (to avoid any existing chat context)

2. Enter this prompt:
   ```
   Create a Kestra flow that loads NYC taxi data from a CSV file to BigQuery. The flow should extract data, upload to GCS, and load to BigQuery.
   ```

3. Observe the results. ChatGPT will generate a Kestra flow, but it likely contains:
   - Outdated plugin syntax (e.g., old task types that have been renamed)
   - Incorrect property names (e.g., properties that don't exist in current versions)
   - Hallucinated features (e.g., tasks, triggers, or properties that never existed)

### Why Does This Happen?

Large Language Models like GPT are trained on data up to a specific point in time. They don't automatically know about software updates and new releases, renamed plugins or changed APIs, new best practices in your organisation, or specific configurations for your infrastructure.

This is the fundamental challenge of using AI: the model can only work with information it has access to.

### Context is Everything

Without proper context, generic AI assistants hallucinate outdated or incorrect code that you can't trust for production use. With proper context, AI generates accurate, current, production-ready code you can iterate on quickly. The same principle applies whether you're generating flows or answering questions from your own data.

---

## Setting up Kestra

### Prerequisites

This module requires [Docker](https://docs.docker.com/get-started/get-docker/) with Docker Compose to run Kestra locally. [Docker Desktop](https://www.docker.com/products/docker-desktop/) is the easiest way to get both on Mac and Windows. If you don't have Docker installed, set that up before proceeding.

### Step 1: Start Kestra

This module includes a `docker-compose.yml` with Kestra pre-configured:

```bash
cd 03-orchestration
docker compose up -d
```

Once the container starts, access the Kestra UI at http://localhost:8080.

To shut down Kestra:

```bash
docker compose down
```

### Step 2: Obtain API Keys

**Gemini API Key (Required)**

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key" and copy your key

The free tier is sufficient for light use, but rate limits are relatively low — you may hit quota quickly if you run the agent and multi-agent flows repeatedly. If you run into `429 Resource Exhausted` errors, wait a minute before retrying, or consider upgrading to a paid tier.

**OpenAI API Key (Required for flow 3)**

1. Visit [platform.openai.com](https://platform.openai.com/home) and sign in or create an account
2. Go to **API keys** and create a new key

**Tavily API Key (Required for web search: flows 3, 5, and 6)**

1. Visit [Tavily](https://tavily.com/)
2. Sign up for the free tier
3. Get your API key from the dashboard

The free tier includes 1,000 searches/month.

### Step 3: Configure API Keys in Kestra

Kestra reads secrets from environment variables prefixed with `SECRET_` where the value is base64-encoded. Export your keys before starting Kestra:

```bash

# GEMINI
export GEMINI_API_KEY="your-gemini-api-key-here" 
export SECRET_GEMINI_API_KEY=$(echo -n $GEMINI_API_KEY | base64) 

# OPENAI
export OPENAI_API_KEY="your-openai-api-key-here"
export SECRET_OPENAI_API_KEY=$(echo -n $OPENAI_API_KEY | base64)   

# TAVILY
export TAVILY_API_KEY="your-tavily-api-key-here"
export SECRET_TAVILY_API_KEY=$(echo -n $TAVILY_API_KEY | base64)   
```

Then start (or restart) Kestra:

```bash
docker compose up -d
```

In flows, reference secrets with `{{ secret('GEMINI_API_KEY') }}` — omit the `SECRET_` prefix when calling `secret()`.

> [!WARNING]
> Never commit API keys to Git!

### Step 4: Import Example Flows

```bash
cd 03-orchestration

# Adjust username and password to match your Kestra setup
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/1_chat_without_rag.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/2_chat_with_rag.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/3_rag_with_websearch.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/4_simple_agent.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/5_web_research_agent.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/6_multi_agent_research.yaml
```

Alternatively, copy-paste the flow YAML directly into Kestra's UI.

### Step 5: Run Your First Agent

1. Open Kestra UI at http://localhost:8080
2. Navigate to the `zoomcamp` namespace
3. Find the `4_simple_agent` flow and click "Execute"
4. Leave default inputs or customize them
5. Watch the execution and review the outputs
6. Then run `5_web_research_agent` and `6_multi_agent_research` and analyze the logs and outputs

---
## AI Copilot

Building workflows manually can be slow. You need to know which plugin to use, look up the exact property names, remember the right syntax, and connect each task together in the correct order. For a non-trivial flow, this can take a long time before you even run it once. While Kestra's autocomplete helps, you still have to build each task one at a time.

AI Copilot changes the approach. Instead of building each step manually, you describe your inputs and your goal — and the AI Copilot generates the flow structure for you. You then tweak the last 5% to get the exact behaviour you want. The AI Copilot handles the boilerplate; you focus on the logic that's specific to your use case.

This works reliably because Kestra's AI Copilot is grounded in the current plugin documentation, valid property names, and best practices for your running version of Kestra — unlike a generic AI assistant, which guesses.

### Setup

> Note: In Kestra's Open Source edition, the AI Copilot only supports Gemini as its AI provider.

Before using AI Copilot, you need to configure Gemini API access in your Kestra instance. 

Access AI Copilot:

1. Open Kestra UI at http://localhost:8080
2. Create a new flow or open an existing one
3. Click the AI Copilot button (sparkle icon ✨) in the top-right corner of the Flow Editor

### Hands-On: Compare Copilot vs. Raw ChatGPT

1. Click the AI Copilot button in Kestra's Flow Editor
2. Enter:
   ```
   Create a Kestra flow that loads NYC taxi data from a CSV file to BigQuery. The flow should extract data, upload to GCS, and load to BigQuery.
   ```
3. Observe the results — correct, up-to-date plugin types, valid property names, and working executable YAML.

### The 5% Rule

Copilot gets you to a working flow quickly, but it won't know everything about your environment. After generation, review the output and make the small adjustments that are specific to your setup - your environment variables, your secrets, your error handling preferences, or a task that needs a slightly different configuration than the default.

The bulk of the structure is done. You're just closing the gap between a general solution and your exact requirements.

### Iterative Refinement

AI Copilot helps with both creating new flows and refining existing ones. The conversation is cumulative — each follow-up preserves the existing flow structure and only modifies what's needed.

Example conversation:

1. "Create a flow that downloads a CSV file and loads it to BigQuery" → Copilot generates a basic flow
2. "Add a task that checks data quality in BigQuery" → Copilot adds SQL validation tasks
3. "Schedule the flow to run daily at 9 AM UTC" → Copilot adds a `Schedule` trigger
4. "Send a Slack notification if it fails" → Copilot adds a `SlackIncomingWebhook` task in an `errors` branch

Then you make the final tweaks manually - adjusting the SQL query, setting your specific Slack channel, or adding a retry count that matches your SLA. You're collaborating with AI, not starting from scratch each time.

### Example Use Cases

- Generate new flows: "Create a flow that syncs data from Postgres to GCS"
- Add tasks: "Add an If-task performing conditional branching"
- Configure triggers: "Add a webhook trigger"
- Add error handling: "Add retry logic with exponential backoff"

### Alternative: Agent Skills

If you're using an AI coding assistant (such as Claude or Cursor), Kestra's [agent-skills](https://github.com/kestra-io/agent-skills) repository gives your AI assistant the same grounding that AI Copilot has inside the UI — current plugin documentation, valid property names, and best practices. This means you can generate reliable, correct Kestra flows directly from your editor without switching to the Kestra UI.

---
