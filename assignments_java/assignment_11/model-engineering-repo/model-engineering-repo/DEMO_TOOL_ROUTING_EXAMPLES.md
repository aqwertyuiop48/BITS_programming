# Tool Routing Examples (Auto Mode) – LLM Demo

This document shows how `llm_demo.py` with `--budget-mode auto` automatically detects query intent and routes tools (search vs code execution).

## How Auto Tool Routing Works

The `infer_tool_routing()` function detects patterns in your query:

**Search tool triggers on:**
- Keywords: "today", "latest", "current", "news", "weather", "price", "stock", "recent", "update"
- Actions: "search", "look up", "find online", "web"
- Questions: "who is", "what happened", "when did"

**Code tool triggers on:**
- Actions: "calculate", "compute", "sum", "average", "mean", "median", "std", "variance"
- Technologies: "python", "pandas", "dataframe", "numpy", "code"
- Math: "regression", "correlation", "simulate", "equation"
- Numeric expressions: patterns like `123 + 456` or `x * 2.5`

---

## Example 1: SEARCH TOOL TRIGGERED ❌ ➡️ ✅

### Query
```
What is the latest Nvidia stock price?
```

### Command
```powershell
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/general_assistant.yaml \
                   --query "What is the latest Nvidia stock price?" \
                   --budget-mode auto
```

### Console Output (Key Lines)
```
[general_assistant_20260516_120000] Calling Gemini API...
  Model: gemini-2.0-flash
  Task: general_assistant
  Budget mode: balanced
  Auto route: True
  Routing reason: detected freshness/search intent
  Routed tools: ['google_search']
  Max tokens: 700
```

### metadata.json Snippet
```json
{
  "task": "general_assistant",
  "query": "What is the latest Nvidia stock price?",
  "budget_mode": "balanced",
  "auto_route": true,
  "routing_reason": "detected freshness/search intent",
  "routed_tools": ["google_search"],
  "prompt_tokens": 145,
  "response_tokens": 87,
  "total_tokens": 232
}
```

**Why this triggered search:**
- Keyword "latest" detected
- Query asks for current real-world data (stock price)
- System automatically routes to `google_search` tool
- Budget bumped to `balanced` (700 tokens) because tool is in use

---

## Example 2: CODE TOOL TRIGGERED

### Query
```
Calculate the average of [10, 20, 30, 45, 50] and find the standard deviation
```

### Command
```powershell
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/general_assistant.yaml \
                   --query "Calculate the average of [10, 20, 30, 45, 50] and find the standard deviation" \
                   --budget-mode auto
```

### Console Output (Key Lines)
```
[general_assistant_20260516_120015] Calling Gemini API...
  Model: gemini-2.0-flash
  Task: general_assistant
  Budget mode: balanced
  Auto route: True
  Routing reason: detected computation/code intent
  Routed tools: ['code_execution']
  Max tokens: 700
```

### metadata.json Snippet
```json
{
  "task": "general_assistant",
  "query": "Calculate the average of [10, 20, 30, 45, 50] and find the standard deviation",
  "budget_mode": "balanced",
  "auto_route": true,
  "routing_reason": "detected computation/code intent",
  "routed_tools": ["code_execution"],
  "prompt_tokens": 162,
  "response_tokens": 124,
  "total_tokens": 286
}
```

**Why this triggered code:**
- Keywords "calculate", "average", "standard deviation" detected
- Numeric expression `[10, 20, 30, ...]` found
- System routes to `code_execution` tool
- Budget set to `balanced` (700 tokens)

---

## Example 3: BOTH TOOLS TRIGGERED (Search + Code)

### Query
```
What is the latest S&P 500 price? Calculate what 5% of that value is.
```

### Command
```powershell
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/general_assistant.yaml \
                   --query "What is the latest S&P 500 price? Calculate what 5% of that value is." \
                   --budget-mode auto
```

### Console Output (Key Lines)
```
[general_assistant_20260516_120030] Calling Gemini API...
  Model: gemini-2.0-flash
  Task: general_assistant
  Budget mode: detailed
  Auto route: True
  Routing reason: detected freshness/search intent; detected computation/code intent
  Routed tools: ['google_search', 'code_execution']
  Max tokens: 1200
```

### metadata.json Snippet
```json
{
  "task": "general_assistant",
  "query": "What is the latest S&P 500 price? Calculate what 5% of that value is.",
  "budget_mode": "detailed",
  "auto_route": true,
  "routing_reason": "detected freshness/search intent; detected computation/code intent",
  "routed_tools": ["google_search", "code_execution"],
  "prompt_tokens": 198,
  "response_tokens": 156,
  "total_tokens": 354
}
```

**Why both were triggered:**
- "latest" keyword → triggers search
- "Calculate" + "5%" → triggers code
- Routing reason combined: `"detected freshness/search intent; detected computation/code intent"`
- Budget upgraded to `detailed` (1200 tokens) because 2 tools in use

---

## Example 4: NO TOOLS TRIGGERED (Pure Knowledge)

### Query
```
What is the Transformer architecture in machine learning?
```

### Command
```powershell
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/general_assistant.yaml \
                   --query "What is the Transformer architecture in machine learning?" \
                   --budget-mode auto
```

### Console Output (Key Lines)
```
[general_assistant_20260516_120045] Calling Gemini API...
  Model: gemini-2.0-flash
  Task: general_assistant
  Budget mode: low
  Auto route: True
  Routing reason: no tool intent detected; using direct generation
  Routed tools: []
  Max tokens: 350
```

### metadata.json Snippet
```json
{
  "task": "general_assistant",
  "query": "What is the Transformer architecture in machine learning?",
  "budget_mode": "low",
  "auto_route": true,
  "routing_reason": "no tool intent detected; using direct generation",
  "routed_tools": [],
  "prompt_tokens": 89,
  "response_tokens": 267,
  "total_tokens": 356
}
```

**Why no tools triggered:**
- No search keywords ("latest", "news", "weather", etc.)
- No computation keywords ("calculate", "average", "sum", etc.)
- Query is asking for general knowledge
- Budget set to `low` (350 tokens) for cost efficiency
- No tools needed; direct generation sufficient

---

## Example 5: NO TOOLS, BUT LONGER QUERY → Budget Bumped

### Query
```
Explain how deep learning neural networks differ from traditional machine learning algorithms, including the history of their development, key innovations that made them practical, current limitations, and potential future directions in the field.
```

### Command
```powershell
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/general_assistant.yaml \
                   --query "Explain how deep learning neural networks differ from traditional machine learning algorithms, including the history of their development, key innovations that made them practical, current limitations, and potential future directions in the field." \
                   --budget-mode auto
```

### Console Output (Key Lines)
```
[general_assistant_20260516_120100] Calling Gemini API...
  Model: gemini-2.0-flash
  Task: general_assistant
  Budget mode: balanced
  Auto route: True
  Routing reason: no tool intent detected; using direct generation
  Routed tools: []
  Max tokens: 700
```

### metadata.json Snippet
```json
{
  "task": "general_assistant",
  "query": "Explain how deep learning neural networks differ from traditional machine learning algorithms...",
  "budget_mode": "balanced",
  "auto_route": true,
  "routing_reason": "no tool intent detected; using direct generation",
  "routed_tools": [],
  "prompt_tokens": 112,
  "response_tokens": 398,
  "total_tokens": 510
}
```

**Why budget was bumped to balanced:**
- Query length > 180 characters (complex multi-part question)
- No tool intent detected
- Heuristic: longer question → needs more tokens for thoughtful answer
- Budget raised from `low` (350) to `balanced` (700) automatically

---

## Decision Tree (How Auto Routing Works)

```
Query → 
  Is "latest"|"news"|"weather"|"price"|"search" etc? 
    YES → Add google_search to tools, budget ≥ balanced
    NO → Continue

  Is "calculate"|"compute"|"average"|"python" etc?
    YES → Add code_execution to tools, budget ≥ balanced
    NO → Continue

  Two tools selected?
    YES → Set budget = detailed (1200 tokens)
    NO → Continue

  One tool selected?
    YES → Set budget = balanced (700 tokens)
    NO → Continue

  Query length > 180 chars?
    YES → Set budget = balanced (700 tokens)
    NO → Set budget = low (350 tokens)
```

---

## Quick Reference: Tool Trigger Keywords

### Search Tool Triggers
- **Freshness**: "latest", "today", "current", "recent", "update", "now"
- **Specific info**: "weather", "price", "stock", "news", "headline"
- **Actions**: "search", "look up", "find online", "web"
- **Questions**: "who is", "what happened", "when did"

### Code Tool Triggers
- **Math**: "calculate", "compute", "sum", "average", "mean", "median", "std"
- **Stats**: "variance", "correlation", "regression"
- **Tech**: "python", "pandas", "dataframe", "numpy", "code"
- **Simulation**: "simulate", "equation", "formula", "+ - * /"

---

## How to Disable Auto Routing (If Needed)

To manually override auto routing and always use `config` tools instead:

```powershell
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/general_assistant.yaml \
                   --query "What is the latest Nvidia stock price?" \
                   --disable-auto-route \
                   --budget-mode balanced
```

When `--disable-auto-route` is used:
- `Routed tools: []` (empty, uses config defaults)
- `Auto route: False`
- `Routing reason: auto-route disabled`
- Budget uses your manual `--budget-mode` setting

---

## Teaching Insight

Show students these examples live to make visible:
1. **Cost awareness**: Compare token counts across modes
2. **Intelligent automation**: System "reads" the question and picks the right tool
3. **Budget profiles**: Different questions get different resource allocations
4. **No magic**: Routing is simple pattern matching; students can predict behavior

This makes the concept concrete and memorable for non-ML audiences.
