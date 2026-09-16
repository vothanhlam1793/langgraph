# LangGraph Starter

A deliberately small Python project for learning how a LangGraph agent works.
It has one LLM, one tool, conversation memory, and an explicit graph loop:

```text
START -> agent -> tools -> agent -> END
                 \-> END
```

## Run it

1. Create a local environment and install dependencies:

   ```bash
   uv sync
   ```

2. Add an OpenAI API key:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and replace `your_api_key_here`.

3. Start the chat:

   ```bash
   uv run python main.py
   ```

Try these prompts:

```text
What is (128 * 7) / 4?
Explain what LangGraph does.
What was the result of the calculation I just asked for?
```

## What to inspect

- `AgentState`: the data that flows through graph nodes. `add_messages` appends new messages instead of replacing the conversation.
- `call_model`: invokes an OpenAI chat model with the available tools.
- `route_after_model`: chooses `tools` when the model emits a tool call, otherwise ends the run.
- `ToolNode`: executes the requested tool and sends its result back to `agent`.
- `MemorySaver` and `thread_id`: retain the conversation state in memory while this program runs.

## LangSmith tracing

Tracing is configured entirely through environment variables, so no application
code is needed. Add these to `.env` to view each graph run, model call, and tool
call in the LangSmith project:

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=langgraph-starter
```

Open [LangSmith](https://smith.langchain.com/) and select `langgraph-starter`
after running the program. Keep `.env` out of version control.

## Next experiments

1. Add a second tool, for example a weather or database lookup.
2. Print `result["messages"]` to see each model and tool message in the loop.
3. Replace `MemorySaver` with a persistent checkpointer when you need durable conversation state.
4. Put this graph behind an API before adding Temporal for durable, multi-step business workflows.
