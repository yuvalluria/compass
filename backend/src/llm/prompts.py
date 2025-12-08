"""Prompt templates for LLM interactions."""

# =============================================================================
# FEW-SHOT EXAMPLES FOR IMPROVED ACCURACY (v2 - Expanded)
# =============================================================================
# These examples help the LLM understand the exact output format expected.
# 10 diverse examples covering edge cases for better accuracy.

FEW_SHOT_EXAMPLES = """
### Example 1: Basic chatbot (conversational, no documents)
Input: "chatbot for 500 users"
Output: {"use_case": "chatbot_conversational", "user_count": 500, "priority": null, "hardware_preference": null}

### Example 2: RAG system (document Q&A, knowledge base, retrieval)
Input: "RAG system for 200 users on H100 GPUs"
Output: {"use_case": "document_analysis_rag", "user_count": 200, "priority": null, "hardware_preference": "H100"}

### Example 3: Document Q&A (NOT chatbot - has documents/knowledge base)
Input: "document Q&A for 600 users"
Output: {"use_case": "document_analysis_rag", "user_count": 600, "priority": null, "hardware_preference": null}

### Example 4: Knowledge base search (RAG, NOT chatbot)
Input: "knowledge base Q&A for 1000 employees"
Output: {"use_case": "document_analysis_rag", "user_count": 1000, "priority": null, "hardware_preference": null}

### Example 5: Semantic search (RAG category)
Input: "semantic search assistant for 450 knowledge workers"
Output: {"use_case": "document_analysis_rag", "user_count": 450, "priority": null, "hardware_preference": null}

### Example 6: SHORT summarization (articles, news, brief docs)
Input: "summarization for 1000 users, budget is tight"
Output: {"use_case": "summarization_short", "user_count": 1000, "priority": "cost_saving", "hardware_preference": null}

### Example 7: LONG document summarization (reports, books, 10+ pages)
Input: "summarize long reports for 80 analysts, documents are 50+ pages"
Output: {"use_case": "long_document_summarization", "user_count": 80, "priority": null, "hardware_preference": null}

### Example 8: Book/chapter summarization (LONG)
Input: "book chapter summarization for 50 students"
Output: {"use_case": "long_document_summarization", "user_count": 50, "priority": null, "hardware_preference": null}

### Example 9: Report condensation (LONG documents)
Input: "report condensation for 70 managers"
Output: {"use_case": "long_document_summarization", "user_count": 70, "priority": null, "hardware_preference": null}

### Example 10: Code completion with speed priority
Input: "code completion for 300 developers, need fast response"
Output: {"use_case": "code_completion", "user_count": 300, "priority": "low_latency", "hardware_preference": null}

### Example 11: Translation with quality priority
Input: "translation service for legal documents, 50 lawyers"
Output: {"use_case": "translation", "user_count": 50, "priority": null, "hardware_preference": null}

### Example 12: Legal/research analysis (multi-document)
Input: "multi-document analysis for 120 researchers"
Output: {"use_case": "research_legal_analysis", "user_count": 120, "priority": null, "hardware_preference": null}
"""

INTENT_EXTRACTION_SCHEMA = """
## EXACT USE CASE MAPPING (choose ONE):

| Use Case | When to Use | Keywords |
|----------|-------------|----------|
| chatbot_conversational | Interactive chat, Q&A bots, customer support | chatbot, bot, assistant, Q&A, support, help desk |
| code_completion | IDE autocomplete, short code suggestions | autocomplete, completion, copilot, suggestions |
| code_generation_detailed | Full code with docs/tests | code generation, generate code, write code |
| translation | Language translation | translate, translation, multilingual, localization |
| content_generation | Marketing, blog, creative writing | content, marketing, blog, writing, copy |
| summarization_short | Brief summaries (<10 pages input) | summarize, summary, condense, brief |
| document_analysis_rag | Document Q&A with retrieval | RAG, document Q&A, knowledge base, search |
| long_document_summarization | Long docs (10+ pages) | long document, lengthy, extensive, reports |
| research_legal_analysis | Legal/academic analysis | legal, research, academic, contract, compliance |

## PRIORITY DETECTION (choose ONE or null):

| Priority | Keywords to Look For |
|----------|---------------------|
| low_latency | fast, quick, instant, real-time, speed, latency, sub-second, millisecond, snappy, responsive |
| cost_saving | budget, cheap, cost, affordable, economical, minimize cost, tight budget, save money |
| high_throughput | batch, volume, scale, throughput, bulk, many requests, high volume, massive |
| high_quality | accuracy, precise, quality, correct, no errors, meticulous, careful, precision |
| balanced | balance, standard, moderate, general purpose |
| null | No priority mentioned - DEFAULT TO NULL |

## HARDWARE DETECTION:
Extract GPU type EXACTLY as mentioned: H100, H200, A100, A10G, L4, T4, V100, A10
If no GPU mentioned → hardware_preference: null

## OUTPUT FORMAT:
{
  "use_case": "<one of the 9 use cases above>",
  "user_count": <integer>,
  "priority": "<low_latency|cost_saving|high_throughput|high_quality|balanced|null>",
  "hardware_preference": "<GPU type or null>"
}
"""


def build_intent_extraction_prompt(user_message: str, conversation_history: list = None, use_few_shot: bool = True) -> str:
    """
    Build prompt for extracting deployment intent from user conversation.
    """
    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for msg in conversation_history[-3:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context += f"{role}: {content}\n"
        context += "\n"

    few_shot_section = ""
    if use_few_shot:
        few_shot_section = f"""
## EXAMPLES (follow this EXACT format):
{FEW_SHOT_EXAMPLES}
"""

    prompt = f"""You are extracting structured deployment requirements from user requests.

{context}User message: "{user_message}"

CRITICAL RULES:
1. ALWAYS output valid JSON
2. use_case MUST be one of: chatbot_conversational, code_completion, code_generation_detailed, translation, content_generation, summarization_short, document_analysis_rag, long_document_summarization, research_legal_analysis
3. priority MUST be one of: low_latency, cost_saving, high_throughput, high_quality, balanced, or null
4. user_count MUST be an integer (estimate if vague: "thousands"→5000, "small team"→20)
5. hardware_preference: extract GPU type (H100, A100, L4, T4, etc.) or null

USE CASE DISAMBIGUATION (CRITICAL - follow exactly):

CHATBOT vs RAG (common confusion!):
- chatbot_conversational: General chat, NO documents mentioned
  Keywords: "chatbot", "chat bot", "customer support chat", "virtual assistant"
- document_analysis_rag: Has documents, knowledge base, retrieval, search
  Keywords: "RAG", "document Q&A", "knowledge base", "semantic search", "document search", "retrieval"
  
SUMMARIZATION LENGTH (common confusion!):
- summarization_short: Brief summaries, news, articles, short docs
  Keywords: "summary", "summarize", "brief", "tldr", "news", "article"
- long_document_summarization: Long reports, books, chapters, 10+ pages
  Keywords: "long document", "report", "book", "chapter", "lengthy", "50+ pages", "condensation"

OTHER USE CASES:
- code_completion: "autocomplete", "completion", "copilot", "IDE"
- code_generation_detailed: "generate code", "code generation", "write code"
- translation: "translate", "translation", "multilingual"
- content_generation: "content", "marketing", "blog", "copywriting"
- research_legal_analysis: "legal", "research", "academic", "contract", "multi-document analysis"

PRIORITY DETECTION:
- "fast", "quick", "latency", "speed", "instant" → low_latency
- "budget", "cheap", "cost", "affordable" → cost_saving
- "batch", "volume", "throughput", "bulk" → high_throughput
- "accuracy", "quality", "precise", "correct" → high_quality
- "balanced", "standard" → balanced
- Nothing mentioned → null

{few_shot_section}

{INTENT_EXTRACTION_SCHEMA}

Extract from: "{user_message}"
Output ONLY the JSON:"""
    return prompt


# NOTE: Experimental prompts moved to prompts_experimental.py
