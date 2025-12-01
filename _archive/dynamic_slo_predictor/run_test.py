#!/usr/bin/env python3
"""
Dynamic SLO Predictor - Main Test Interface

Pipeline:
1. E5 Embedding Model - Semantic task understanding
2. For KNOWN tasks (>80% similarity): Use lookup table
3. For UNKNOWN tasks (<80% similarity): 
   - Use input as use_case name
   - Calculate weighted average SLOs from similar tasks
   - Cache for future similar queries

Output:
- JSON 1: TASK (use_case, user_count, priority, hardware)
- JSON 2: DESIRED SLO (slo ranges, workload)

Usage:
    python3 -m dynamic_slo_predictor.run_test "your task description"
    python3 -m dynamic_slo_predictor.run_test  # Interactive mode
"""

import json
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from output_schemas import (
    extract_user_count,
    extract_priority,
    extract_hardware,
    extract_task_name,
    build_desired_slo,
    interpolate_slo_ranges,
    calculate_workload_distribution,
    TASK_SLO_RANGES,
    CUSTOM_TASK_CACHE,
    SIMILARITY_THRESHOLD,
    save_custom_task,
    get_cached_custom_task,
    SLORange,
    WorkloadSpec,
    WorkloadDistribution,
    DesiredSLO,
)
from research_data import get_workload_research_source


def normalize_text(text: str) -> str:
    """Normalize text to fix common typos."""
    text = text.lower()
    
    typo_fixes = {
        "ley": "key", "kye": "key", "critica": "critical",
        "latancy": "latency", "througput": "throughput",
        "tranlsation": "translation", "summeryzation": "summarization",
        "summerization": "summarization", "usres": "users", "uers": "users",
    }
    
    for typo, fix in typo_fixes.items():
        text = text.replace(typo, fix)
    
    return text


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        missing.append("scikit-learn")
    
    try:
        import chromadb
    except ImportError:
        missing.append("chromadb")
    
    if missing:
        print("=" * 70)
        print("  MISSING DEPENDENCIES")
        print("=" * 70)
        print(f"\nPlease install: {', '.join(missing)}")
        print(f"\nRun: pip install {' '.join(missing)}")
        print("=" * 70)
        return False
    
    return True


def load_pipeline():
    """Load the E5 embedding model and RAG corpus"""
    import sys
    from pathlib import Path
    
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from dynamic_slo_predictor.task_embedder import TaskEmbedder
    from dynamic_slo_predictor.research_corpus import ResearchCorpus
    
    print("Loading pipeline...")
    
    # Load E5 embedding model for task detection
    print("  → Loading E5 embedding model...")
    embedder = TaskEmbedder()
    
    # Load RAG corpus (ChromaDB with research papers)
    print("  → Loading RAG corpus (ChromaDB)...")
    corpus = ResearchCorpus()
    corpus.initialize()
    
    print("  ✓ Pipeline ready!\n")
    
    return embedder, corpus


def build_custom_slo(task_name: str, similarity_scores: list, user_count: int = None) -> DesiredSLO:
    """
    Build SLO for a custom/unknown task using weighted interpolation.
    
    Args:
        task_name: The custom task name (user's input)
        similarity_scores: List of {"task_type": str, "similarity": float}
        user_count: Optional user count for RPS calculation
        
    Returns:
        DesiredSLO with interpolated values
    """
    # Interpolate SLO ranges from similar predefined tasks
    slo_config = interpolate_slo_ranges(similarity_scores, top_k=3)
    
    # Build SLO ranges
    ttft_range = SLORange(
        min_value=int(slo_config["ttft"][0]),
        max_value=int(slo_config["ttft"][1]),
        unit="ms"
    )
    
    itl_range = SLORange(
        min_value=int(slo_config["itl"][0]),
        max_value=int(slo_config["itl"][1]),
        unit="ms"
    )
    
    e2e_range = SLORange(
        min_value=int(slo_config["e2e"][0]),
        max_value=int(slo_config["e2e"][1]),
        unit="ms"
    )
    
    # Build workload spec
    workload_config = slo_config.get("workload", {})
    requests_in = int((workload_config.get("requests_in", [300, 300])[0] + 
                       workload_config.get("requests_in", [300, 300])[1]) / 2)
    requests_out = int((workload_config.get("requests_out", [175, 175])[0] + 
                        workload_config.get("requests_out", [175, 175])[1]) / 2)
    
    # RPS based on user count
    if user_count:
        rps = round(user_count / 300, 2)  # Estimate: 1 request per user per 5 minutes
    else:
        rps_range = workload_config.get("rps", [10, 10])
        rps = round((rps_range[0] + rps_range[1]) / 2, 2)
    
    workload = WorkloadSpec(
        requests_in=requests_in,
        requests_out=requests_out,
        rps=rps
    )
    
    # Save to cache for future similar queries
    save_custom_task(task_name, slo_config, similarity_scores[:3])
    
    return DesiredSLO(
        task_type=task_name,
        ttft=ttft_range,
        itl=itl_range,
        e2e=e2e_range,
        workload=workload
    )


def process_input(user_input: str, embedder, corpus=None) -> tuple:
    """
    Process user input through the pipeline.
    
    Pipeline:
    1. Normalize text (fix typos)
    2. E5 Embedding → Detect task type semantically
    3. RAG Search → Find relevant research papers
    4. Check similarity:
       - If >= 80%: Use predefined lookup table
       - If < 80%: Custom task → weighted average SLOs, cache it
    
    Returns:
        (task_json, desired_slo_json, info)
    """
    # Step 1: Normalize text
    normalized_input = normalize_text(user_input)
    
    # Step 2: E5 Embedding - Semantic task understanding
    task_analysis = embedder.analyze_task(normalized_input)
    matched_tasks = task_analysis.get("matched_tasks", [])
    
    # Get top match and its similarity
    top_match = matched_tasks[0] if matched_tasks else None
    top_similarity = top_match["similarity"] if top_match else 0
    
    # Get top 3 matches for display
    task_matches = [
        {"name": m["name"], "similarity": f"{m['similarity']:.1%}"}
        for m in matched_tasks[:3]
    ]
    
    # Step 3: RAG Search - Find relevant research papers (with chunk text)
    research_sources = []
    if corpus:
        research_results = corpus.get_research_for_task(normalized_input)
        research_sources = [
            {
                "paper_name": s["paper_name"],
                "chunk_text": s["chunk_text"],
                "similarity": f"{s['similarity']:.1%}"
            }
            for s in research_results.get("sources", [])[:3]
        ]
    
    # Step 4: Extract optional fields
    user_count = extract_user_count(normalized_input)
    priority = extract_priority(normalized_input)
    hardware = extract_hardware(normalized_input)
    
    # Step 5: Determine if known or custom task
    is_custom = top_similarity < SIMILARITY_THRESHOLD
    
    if is_custom:
        # CUSTOM TASK: Extract just the task name, interpolate SLOs
        task_type = extract_task_name(user_input)  # Extract clean task name
        
        # Check cache first
        cached = get_cached_custom_task(task_type)
        if cached:
            # Use cached SLOs
            slo_config = cached["slo_config"]
            desired_slo = build_desired_slo(
                task_type=task_type,
                user_count=user_count,
                similarity_scores=cached.get("similar_to", []),
                is_custom=True
            )
            info_note = "from_cache"
        else:
            # Build similarity scores for interpolation
            similarity_scores = [
                {"task_type": m["name"], "similarity": m["similarity"]}
                for m in matched_tasks
            ]
            
            # Build custom SLO with interpolation
            desired_slo = build_custom_slo(task_type, similarity_scores, user_count)
            info_note = "interpolated"
        
        experience_class = "custom"
    else:
        # KNOWN TASK: Use predefined lookup table
        task_type = top_match["name"]
        experience_class = task_analysis["task"]["experience_class"]
        
        desired_slo = build_desired_slo(
            task_type=task_type,
            user_count=user_count
        )
        info_note = "predefined"
    
    desired_slo_json = desired_slo.to_dict()
    
    # Step 6: Calculate workload distribution if user_count provided
    workload_dist = None
    workload_dist_json = None
    workload_research = None
    if user_count:
        # Get the task type for workload (use closest match for custom tasks)
        workload_task_type = task_type if not is_custom else (matched_tasks[0]["name"] if matched_tasks else "chatbot_conversational")
        
        workload_dist = calculate_workload_distribution(
            workload_task_type,
            user_count
        )
        workload_dist_json = workload_dist.to_dict()
        # Add distribution to desired SLO JSON
        desired_slo_json["workload_distribution"] = workload_dist_json
        
        # Get the research source for this workload pattern
        workload_research = get_workload_research_source(workload_task_type)
    
    # Step 7: Build Task JSON
    task_json = {"use_case": task_type}
    if user_count is not None:
        task_json["user_count"] = user_count
    if priority is not None:
        task_json["priority"] = priority
    if hardware is not None:
        task_json["hardware"] = hardware
    
    # Info for display
    info = {
        "task_matches": task_matches,
        "research_sources": research_sources,
        "workload_distribution": workload_dist_json,
        "workload_research": workload_research,
        "experience_class": experience_class,
        "is_custom": is_custom,
        "source": info_note,
        "similarity_threshold": f"{SIMILARITY_THRESHOLD:.0%}",
    }
    
    return task_json, desired_slo_json, info


def print_results(user_input: str, task_json: dict, slo_json: dict, info: dict):
    """Pretty print the results"""
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    
    print(f"\n📝 USER INPUT:")
    print(f"   \"{user_input}\"")
    
    # Show E5 embedding analysis
    print(f"\n🧠 E5 EMBEDDING (Task Detection):")
    for match in info.get("task_matches", []):
        print(f"   → {match['name']}: {match['similarity']}")
    
    # Show RAG research sources with chunk text
    if info.get("research_sources"):
        print(f"\n📚 RAG RESEARCH (Paper Name + Chunk):")
        for i, src in enumerate(info["research_sources"], 1):
            print(f"\n   [{i}] Paper: {src['paper_name']}")
            print(f"       Similarity: {src['similarity']}")
            # Show chunk preview (first 150 chars)
            chunk = src.get('chunk_text', '')[:150]
            if len(src.get('chunk_text', '')) > 150:
                chunk += "..."
            print(f"       Chunk: \"{chunk}\"")
    
    # Show workload distribution with research source
    if info.get("workload_distribution"):
        wd = info["workload_distribution"]
        wr = info.get("workload_research", {})
        print(f"\n📊 WORKLOAD PATTERN:")
        print(f"   Type: {wd['distribution_type']}")
        print(f"   RPS: {wd['rps']['description']}")
        print(f"   Peak: {wd['peak_info']}")
        if wr:
            print(f"\n   📖 Research Source:")
            print(f"      Paper: {wr.get('paper', 'N/A')}")
            print(f"      Chunk: \"{wr.get('chunk', 'N/A')}\"")
            print(f"      Reason: {wr.get('distribution_reason', 'N/A')}")
    
    # Show if custom or predefined
    if info.get("is_custom"):
        print(f"\n⚠️  CUSTOM TASK (similarity < {info['similarity_threshold']})")
        print(f"   SLOs calculated via weighted interpolation ({info['source']})")
    else:
        print(f"\n✓ PREDEFINED TASK (similarity >= {info['similarity_threshold']})")
    
    print(f"\n{'─' * 70}")
    print("📋 JSON 1: TASK")
    print("─" * 70)
    print(json.dumps(task_json, indent=2))
    
    print(f"\n{'─' * 70}")
    print("🎯 JSON 2: DESIRED SLO")
    print("─" * 70)
    print(json.dumps(slo_json, indent=2))
    
    # Quick summary
    slo = slo_json["slo"]
    workload = slo_json["workload"]
    
    print(f"\n{'─' * 70}")
    print("📊 QUICK SUMMARY")
    print("─" * 70)
    print(f"   Task Type:  {slo_json['task_type']}")
    print(f"   Experience: {info.get('experience_class', 'N/A')}")
    print(f"   TTFT:       {slo['ttft']['range_str']}")
    print(f"   ITL:        {slo['itl']['range_str']}")
    print(f"   E2E:        {slo['e2e']['range_str']}")
    print(f"   Workload:   {workload['requests_in']} tokens in, {workload['requests_out']} tokens out, {workload['rps']} RPS")
    
    print("\n" + "=" * 70)


def interactive_mode(embedder, corpus):
    """Run in interactive mode"""
    print("\n" + "=" * 70)
    print("  DYNAMIC SLO PREDICTOR")
    print("  Using: E5 Embedding + RAG (ChromaDB) + Lookup Table")
    print("=" * 70)
    print(f"\n• Known tasks (>={SIMILARITY_THRESHOLD:.0%} match): Use predefined SLOs")
    print(f"• Unknown tasks (<{SIMILARITY_THRESHOLD:.0%} match): Interpolate + cache")
    print("• RAG shows relevant research papers for all queries")
    print("\nType 'quit' to exit, 'cache' to see cached tasks.\n")
    print("Examples:")
    print("  • code completion for 500 developers")
    print("  • AI for autonomous vehicle navigation")
    print("  • medical diagnosis chatbot")
    print()
    
    while True:
        try:
            user_input = input("📝 Enter task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'cache':
            print(f"\n📦 CACHED CUSTOM TASKS ({len(CUSTOM_TASK_CACHE)}):")
            if CUSTOM_TASK_CACHE:
                for name, data in CUSTOM_TASK_CACHE.items():
                    similar = data.get("similar_to", [])[:2]
                    similar_str = ", ".join([f"{s['task_type']}" for s in similar])
                    print(f"   • {name[:50]}... → similar to: {similar_str}")
            else:
                print("   (empty)")
            print()
            continue
        
        try:
            task_json, slo_json, info = process_input(user_input, embedder, corpus)
            print_results(user_input, task_json, slo_json, info)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
        
        print()


def single_input(user_input: str, embedder, corpus):
    """Process a single input from command line"""
    task_json, slo_json, info = process_input(user_input, embedder, corpus)
    print_results(user_input, task_json, slo_json, info)


def main():
    # Check dependencies
    if not check_dependencies():
        return
    
    # Load pipeline (E5 embedder + RAG corpus)
    embedder, corpus = load_pipeline()
    
    if len(sys.argv) > 1:
        # Command line argument provided
        user_input = " ".join(sys.argv[1:])
        single_input(user_input, embedder, corpus)
    else:
        # Interactive mode
        interactive_mode(embedder, corpus)


if __name__ == "__main__":
    main()
