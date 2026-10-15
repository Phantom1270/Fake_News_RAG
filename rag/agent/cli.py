import sys
import json
import argparse
from dataclasses import asdict

from rag.agent.verifier import ClaimVerifier
from rag.retrieval.retrieval_engine import RetrievalEngine


def init_retrieval_engine():
    """Initialize the RAG retrieval engine."""
    return RetrievalEngine()


def print_result(result, format_type):
    if format_type == "json":
        # Convert enum to string before JSON serialization
        res_dict = asdict(result)
        res_dict["verdict"] = result.verdict.value
        print(json.dumps(res_dict, indent=2))
        return

    print("\n" + "="*50)
    print(f"CLAIM: {result.claim}")
    print("="*50)
    
    # Color formatting if possible (simple ANSI)
    verdict_colors = {
        "true": "\033[92m",        # Green
        "false": "\033[91m",       # Red
        "misleading": "\033[93m",  # Yellow
        "partially_true": "\033[93m",
        "unverifiable": "\033[90m" # Gray
    }
    reset = "\033[0m"
    
    v_str = result.verdict.value
    color = verdict_colors.get(v_str, "")
    
    print(f"\nVERDICT: {color}{v_str.upper()}{reset}")
    print(f"CONFIDENCE: {result.confidence:.2f}")
    print(f"BACKEND: {result.llm_backend.upper()}")
    
    print("\n--- EXPLANATION ---")
    print(result.explanation)
    
    if result.reasoning:
        print("\n--- REASONING ---")
        print(result.reasoning)
        
    print(f"\n--- EVIDENCE ({result.matched_fact_checks} sources) ---")
    if not result.evidence:
        print("No evidence found.")
    else:
        for i, ev in enumerate(result.evidence):
            print(f"\n[{i+1}] {ev.source_title}")
            print(f"    Source: {ev.source_name}")
            print(f"    URL: {ev.source_url}")
            print(f"    Original Verdict: {ev.original_verdict}")
            print(f"    Similarity: {ev.similarity_score:.3f}")


def run_interactive(verifier, format_type):
    print("Fake News Detector - Interactive Mode")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            claim = input("> Enter claim: ").strip()
            if not claim:
                continue
            if claim.lower() in ["exit", "quit"]:
                break
                
            result = verifier.verify(claim)
            print_result(result, format_type)
            print("\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fake News Detector — Claim Verification")
    parser.add_argument("claim", nargs="?", help="Claim to verify")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--backend", "-b", choices=["groq", "gemini"], default=None,
                        help="LLM backend (default: from config)")
    parser.add_argument("--api-key", "-k", default=None,
                        help="Your API key (uses trial key if not provided)")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text",
                        help="Output format")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of chunks to retrieve")
    args = parser.parse_args()

    # Disable httpx info logs
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not args.claim and not args.interactive:
        parser.print_help()
        return

    print("Initializing RAG Engine...")
    retrieval_engine = init_retrieval_engine()
    
    try:
        verifier = ClaimVerifier(
            retrieval_engine=retrieval_engine,
            llm_backend=args.backend,
            user_api_key=args.api_key
        )
    except Exception as e:
        print(f"Failed to initialize Verifier: {e}")
        sys.exit(1)

    if args.interactive:
        run_interactive(verifier, args.format)
    else:
        result = verifier.verify(args.claim, top_k=args.top_k)
        print_result(result, args.format)


if __name__ == "__main__":
    main()
