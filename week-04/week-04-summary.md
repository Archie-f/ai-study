# Week 04 — Prompt Engineering & LLM API Fundamentals

## What I Built
- prompt_anatomy.py — Builds prompt text with the given parameters in the given format (task, context, output format)
- few_shot.py — Builds a prompt with the given examples and new input using the context
- chain_of_thought.py — CoT prompt builder to make the model response the input through the given instructions
- eval_mindset.py — Builds a result summary for the test cases using manually created results
- prompt_runner.py — multi-variant harness: Runs the same input with three different prompt variants,
                     generates a table of results' comparison with in and out token counts, costs, and latencies.
- structured_outputs.py — Analyzes responses for different inputs using Pydantic model validation
- batch_eval.py — batch evaluator: runs harness over multiple inputs, scores results
- conversation.py — ConversationManager class implementing sliding window truncation based on token counts

## Key Techniques Learned
- Prompt anatomy: system / task / context / format layers
- Few-shot: how many examples, ordering effect, domain matching
- Chain-of-thought: the magic phrase, when to use, debuggability
- JSON mode: prompt pattern 'Return only valid JSON', parsing with json.loads()
- Pydantic v2: BaseModel, Field, ValidationError, model_validate()
- Conversation history: alternating user/assistant roles, system vs messages
- Token counting: client.messages.count_tokens() API
- Sliding window: remove oldest pairs until token count < max_tokens

## How This Feeds llm-compare
prompt_runner.py is the core of llm-compare. It will be extended in Week 05
to call OpenAI and Ollama alongside Claude.

## Dependencies
anthropic>=0.25, pydantic>=2.0, python-dotenv>=1.0