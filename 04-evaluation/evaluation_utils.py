import time
import json
from tqdm.auto import tqdm
from rag_helper import RAGBase


def calc_price(usage):
    # Groq pricing for openai/gpt-oss-120b
  
    input_price_per_million = 0.15  
    output_price_per_million = 0.60 

    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(client, instructions, user_prompt, output_type, model="openai/gpt-oss-120b"):
    """
    Groq version of structured LLM call.
    Uses JSON mode with schema enforcement via system prompt.
    """
    # Get the JSON schema from the Pydantic model
    schema = output_type.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    
    # Add schema instructions to the system prompt
    enhanced_instructions = (
        instructions + 
        "\n\nIMPORTANT: Return ONLY a valid JSON object matching this exact schema:\n"
        f"{schema_str}\n"
        "Do not include any other text, markdown, or formatting outside the JSON object."
    )
    
    messages = [
        {"role": "system", "content": enhanced_instructions},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    
    # Parse the JSON response
    result_text = response.choices[0].message.content
    
    # Handle potential markdown code blocks
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    try:
        result_dict = json.loads(result_text)
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response if it's embedded
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result_dict = json.loads(json_match.group())
        else:
            raise ValueError(f"Failed to parse JSON from response: {result_text[:200]}...") from e
    
    # Validate with Pydantic
    validated_result = output_type(**result_dict)
    
    # Create a usage object compatible with the expected format
    class Usage:
        def __init__(self, prompt_tokens, completion_tokens, total_tokens):
            self.input_tokens = prompt_tokens
            self.output_tokens = completion_tokens
            self.total_tokens = total_tokens
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            
        def __repr__(self):
            return f"ResponseUsage(input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, total_tokens={self.total_tokens})"
    
    usage = Usage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens
    )
    
    return validated_result, usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="openai/gpt-oss-120b",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {2 ** attempt}s...")
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages,
            temperature=0.7
        )

        # Create a usage object compatible with the expected format
        class Usage:
            def __init__(self, prompt_tokens, completion_tokens, total_tokens):
                self.input_tokens = prompt_tokens
                self.output_tokens = completion_tokens
                self.total_tokens = total_tokens
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens
                
            def __repr__(self):
                return f"ResponseUsage(input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, total_tokens={self.total_tokens})"
        
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )

        self.last_usage = usage
        self.usages.append(usage)

        return response.choices[0].message.content

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results