"""Prompt templates for LLM interactions."""

# =============================================================================
# FEW-SHOT EXAMPLES FOR IMPROVED ACCURACY
# =============================================================================
# These examples help the LLM understand the exact output format expected.
# Adding 3-5 examples typically improves accuracy by 2-5%.

FEW_SHOT_EXAMPLES = """
### Example 1: Basic chatbot request
Input: "chatbot for 500 users"
Output: {"use_case": "chatbot_conversational", "user_count": 500, "experience_class": "conversational", "latency_requirement": "high", "priority": null, "hardware_preference": null}

### Example 2: Code completion with priority
Input: "code completion for 300 developers, need fast response"
Output: {"use_case": "code_completion", "user_count": 300, "experience_class": "instant", "latency_requirement": "very_high", "priority": "low_latency", "hardware_preference": null}

### Example 3: RAG with hardware preference
Input: "RAG system for 200 users on H100 GPUs, latency is key"
Output: {"use_case": "document_analysis_rag", "user_count": 200, "experience_class": "interactive", "latency_requirement": "very_high", "priority": "low_latency", "hardware_preference": "H100"}

### Example 4: Summarization with cost priority
Input: "summarization for 1000 users, budget is tight"
Output: {"use_case": "summarization_short", "user_count": 1000, "experience_class": "interactive", "latency_requirement": "medium", "priority": "cost_saving", "hardware_preference": null}

### Example 5: Translation with quality focus
Input: "translation service for legal documents, 50 lawyers, accuracy is critical"
Output: {"use_case": "translation", "user_count": 50, "experience_class": "deferred", "latency_requirement": "medium", "priority": "high_quality", "hardware_preference": null}
"""

INTENT_EXTRACTION_SCHEMA = """
Expected JSON schema:
{
  "use_case": "chatbot_conversational|code_completion|code_generation_detailed|translation|content_generation|summarization_short|document_analysis_rag|long_document_summarization|research_legal_analysis",
  "experience_class": "instant|conversational|interactive|deferred|batch",
  "user_count": <integer>,
  "latency_requirement": "very_high|high|medium|low",
  "throughput_priority": "very_high|high|medium|low",
  "budget_constraint": "strict|moderate|flexible|none",
  "priority": "low_latency|cost_saving|high_throughput|high_quality|balanced|null",
  "hardware_preference": "<GPU type if mentioned: H100, H200, A100, A10G, L4, T4, V100, A10, or null>",
  "domain_specialization": ["general"|"code"|"multilingual"|"enterprise"],
  "additional_context": "<any other relevant details mentioned>"
}

Use case descriptions:
- chatbot_conversational: Real-time conversational chatbots (short prompts, short responses)
- code_completion: Fast code completion/autocomplete (short prompts, short completions)
- code_generation_detailed: Detailed code generation with explanations (medium prompts, long responses)
- translation: Document translation (medium prompts, medium responses)
- content_generation: Content creation, marketing copy (medium prompts, medium responses)
- summarization_short: Short document summarization (medium prompts, short summaries)
- document_analysis_rag: RAG-based document Q&A (long prompts with context, medium responses)
- long_document_summarization: Long document summarization (very long prompts, medium summaries)
- research_legal_analysis: Research/legal document analysis (very long prompts, detailed analysis)

Priority detection keywords:
- low_latency: "fast", "quick", "instant", "real-time", "latency is key", "speed matters"
- cost_saving: "budget", "cheap", "cost-effective", "affordable", "minimize cost"
- high_throughput: "batch", "high volume", "scale", "throughput", "many requests"
- high_quality: "accuracy", "precise", "quality matters", "no hallucinations", "accurate"
- balanced: "balance", "standard", "moderate" (or nothing mentioned)

Experience class guidance:
- instant: Extremely low latency required (<200ms TTFT) - code completion, autocomplete
- conversational: Real-time user interaction (chatbots, interactive tools) - low latency needed
- interactive: User waiting but can tolerate slight delay (RAG Q&A, content generation) - balanced
- deferred: User can wait for quality (long summarization, detailed analysis) - quality over speed
- batch: Background/async processing (research, legal analysis) - optimize for quality and cost
"""


def build_intent_extraction_prompt(user_message: str, conversation_history: list = None, use_few_shot: bool = True) -> str:
    """
    Build prompt for extracting deployment intent from user conversation.

    Args:
        user_message: Latest user message
        conversation_history: Optional list of previous messages
        use_few_shot: Whether to include few-shot examples (improves accuracy by ~3%)

    Returns:
        Formatted prompt string
    """
    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for msg in conversation_history[-3:]:  # Last 3 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context += f"{role}: {content}\n"
        context += "\n"

    # Include few-shot examples for improved accuracy
    few_shot_section = ""
    if use_few_shot:
        few_shot_section = f"""
## Examples (follow this exact format):
{FEW_SHOT_EXAMPLES}
"""

    prompt = f"""You are an expert AI assistant for Compass helping users deploy Large Language Models (LLMs) for production use cases.

{context}Current user message: {user_message}

Your task is to extract structured information about their deployment requirements. Analyze what they've said and infer:

1. **Use case**: What type of application (chatbot, customer service, code generation, summarization, etc.)
2. **User count**: How many users or scale mentioned (extract the NUMBER, estimate if not explicit)
3. **Priority**: User's main concern - detect from keywords:
   - "fast", "quick", "latency" → priority: "low_latency"
   - "cheap", "budget", "cost" → priority: "cost_saving"
   - "accuracy", "quality", "precise" → priority: "high_quality"
   - "batch", "throughput", "scale" → priority: "high_throughput"
   - If not mentioned → priority: null
4. **Hardware preference**: Did they mention specific GPU types? (H100, A100, L4, etc.)
5. **Latency requirement**: How important is low latency? (very_high/high/medium/low)
6. **Budget constraint**: How price-sensitive are they?
7. **Domain specialization**: Any specific domains mentioned (code, multilingual, enterprise, etc.)

{few_shot_section}

Be intelligent about inference:
- "thousands of users" → estimate specific number (e.g., 5000)
- "5k users" → user_count: 5000
- "needs to be fast" or "low latency critical" → priority: "low_latency", latency_requirement: "very_high"
- "budget is tight" or "cost is key" → priority: "cost_saving"
- "accuracy is critical" → priority: "high_quality"
- "on H100" or "using A100" → hardware_preference: "H100" or "A100"
- No hardware mentioned → hardware_preference: null

{INTENT_EXTRACTION_SCHEMA}

Now extract from the user message above. Output ONLY the JSON object:"""
    return prompt


# NOTE: Experimental prompts for future conversational features have been
# moved to prompts_experimental.py to keep this file focused on production code.
