# Gemini Tool Calls Guide for This Repo

This guide explains how to enable and use Gemini tool calls in the existing llm_demo flow.

## What Tool Calls Mean

Tool calls allow Gemini to use external capabilities during generation, such as:

- google_search: web-backed retrieval during response generation
- code_execution: execute code snippets in a controlled runtime

## Files Involved

- src/models/gemini_client.py: sends tools and tool_config to Gemini API
- llm_demo.py: reads tool settings from config and passes them to client
- configs/llm/gemini.yaml: place to enable tool definitions
- configs/llm/gemini_search.yaml: search-only demo config
- configs/llm/gemini_code_execution.yaml: code-execution-only demo config
- configs/llm/gemini_tools_all.yaml: combined tools demo config

## How Tool Calls Are Wired

1. llm_demo.py loads generation_params from configs/llm/gemini.yaml.
2. If tools or tool_config are present, they are passed to GeminiClient.generate.
3. GeminiClient uses GenerateContentConfig with optional tools and tool_config.
4. params.json stores whether tools were enabled for that run.
5. metadata.json stores tool usage evidence under tool_usage.

## Enable Tool Calls in Config

Edit configs/llm/gemini.yaml and uncomment or add:

generation_params:
  temperature: 0.7
  max_tokens: 4096
  tools:
    - google_search: {}
    - code_execution: {}

Note:
- Do not set function_calling_config unless you are using custom function declarations.
- Setting function_calling_config without function declarations can cause INVALID_ARGUMENT errors.

## Ready-to-Run Demos

Search-only:

python llm_demo.py --config gemini_search.yaml --prompt-config search_weather_brief.yaml --query "What is the weather in Bangalore right now?"

Code execution only:

python llm_demo.py --config gemini_code_execution.yaml --prompt-config code_execution_demo.yaml --query "Compute the CAGR from 120 to 185 over 6 years"

Search + code execution:

python llm_demo.py --config gemini_tools_all.yaml --prompt-config search_plus_compute.yaml --query "Find current Bangalore temperature in Celsius and convert it to Fahrenheit"

Shorthand note:
- llm_demo.py now accepts shorthand file names for config and prompt-config.
- It resolves from configs/llm and configs/prompts when direct paths are not found.

## Run Command

python llm_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/model_explanation.yaml --query "Explain random forest with a recent example"

## Behavior Notes

- Tool calls are model and account capability dependent.
- If tools are not supported for a chosen model, Gemini can return an API argument error.
- Start with one tool at a time for easier debugging.

## How to Verify Tool Usage from Artifacts

After each run, inspect runs/<run_name>/metadata.json:

- tool_usage.tool_calls_detected: high-level true/false signal
- tool_usage.has_grounding_metadata: usually true for web-grounded search responses
- tool_usage.execution_events: includes executable_code and code_execution_result when code tool runs
- tool_usage.function_calls: populated when custom function calling is used

## Troubleshooting

1. API key errors:
- Verify GEMINI_API_KEY in shell or .env.

2. INVALID_ARGUMENT errors:
- Validate tool schema and supported tools for your model.
- Temporarily disable tools to isolate issue.
- If error mentions function_declarations, remove function_calling_config or add proper function declarations.

3. Non-deterministic outputs:
- Lower temperature for task-like prompts.

## Best Practices for Teaching Demos

- Keep prompts structured with explicit output format.
- Log tool-enabled and tool-disabled runs separately for comparison.
- Use runs artifacts to compare latency and response quality.
- Treat tool calls as optional augmentation, not guaranteed behavior.

## Minimal Experiment Plan

1. Baseline run without tools.
2. Run with google_search only.
3. Run with code_execution only.
4. Compare response quality, latency, and consistency.

## Custom Function Calling (Advanced)

Custom functions are also possible, but require:

1. Function declarations in the tool definitions.
2. Optional function_calling_config for routing mode.
3. App-side handling for function execution and returning tool outputs.

This repo currently demonstrates built-in tools (google_search and code_execution) out of the box.
