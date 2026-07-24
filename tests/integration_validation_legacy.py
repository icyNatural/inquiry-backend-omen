import sys
import time
import json
from pathlib import Path

# Add scripts folder to path
sys.path.append(str(Path(__file__).resolve().parent))

from cognition_layer import CognitionLayer
from scope_processor import ScopeProcessor
import memory_manager

# Define the 30 representative test queries
TEST_SUITE = [
    # Philosophy
    {
        "query": "plato thinking structures",
        "mode": "recall",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Philosophy"
    },
    {
        "query": "socrates and dialogue methods",
        "mode": "recall",
        "scope": {"namespaces": ["chatgpt_export"]},
        "category": "Philosophy"
    },
    {
        "query": "why seeking creates lack",
        "mode": "framework",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Philosophy"
    },
    {
        "query": "ambiguity as a cognitive tool",
        "mode": "synthesis",
        "scope": {"namespaces": ["refined_notes"]},
        "category": "Philosophy"
    },
    {
        "query": "structural intelligence models",
        "mode": "framework",
        "scope": {"namespaces": ["refined_notes"]},
        "category": "Philosophy"
    },
    # Language Systems
    {
        "query": "poetic diction and word atmosphere",
        "mode": "raw",
        "scope": {"namespaces": ["refined_notes"], "domains": ["language_system"]},
        "category": "Language Systems"
    },
    {
        "query": "phonaesthemes sound clusters",
        "mode": "raw",
        "scope": {"namespaces": ["refined_notes"], "domains": ["language_system"]},
        "category": "Language Systems"
    },
    {
        "query": "sound symbolism and meaning suggestion",
        "mode": "recall",
        "scope": {"namespaces": ["refined_notes"], "domains": ["language_system"]},
        "category": "Language Systems"
    },
    {
        "query": "verbal aura word associations",
        "mode": "recall",
        "scope": {"namespaces": ["refined_notes"], "domains": ["language_system"]},
        "category": "Language Systems"
    },
    {
        "query": "language experience stack felt experience",
        "mode": "framework",
        "scope": {"namespaces": ["refined_notes"], "domains": ["language_system"]},
        "category": "Language Systems"
    },
    # Frameworks
    {
        "query": "elite beta thinking subtraction",
        "mode": "framework",
        "scope": {"namespaces": ["refined_notes"]},
        "category": "Frameworks"
    },
    {
        "query": "zero-state thinking motion before narration",
        "mode": "recall",
        "scope": {"namespaces": ["refined_notes"]},
        "category": "Frameworks"
    },
    {
        "query": "stable node calm reference point",
        "mode": "framework",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Frameworks"
    },
    {
        "query": "energy engine physiology baseline",
        "mode": "framework",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Frameworks"
    },
    {
        "query": "language execution framework outcomes",
        "mode": "framework",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Frameworks"
    },
    # Ambiguous Concepts
    {
        "query": "what does flow mean",
        "mode": "grounding",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"], "exclude_terms": ["music", "flute"]},
        "category": "Ambiguous Concepts"
    },
    {
        "query": "what is an engine in my notes",
        "mode": "grounding",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Ambiguous Concepts"
    },
    {
        "query": "mental model vs software model",
        "mode": "grounding",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Ambiguous Concepts"
    },
    {
        "query": "state and motion dynamics",
        "mode": "synthesis",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Ambiguous Concepts"
    },
    {
        "query": "clamping biological speed",
        "mode": "recall",
        "scope": {"namespaces": ["chatgpt_export"]},
        "category": "Ambiguous Concepts"
    },
    # Cross-Domain Synthesis
    {
        "query": "rap rhythm and thought flow",
        "mode": "synthesis",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Cross-Domain Synthesis"
    },
    {
        "query": "music and memory compression",
        "mode": "synthesis",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Cross-Domain Synthesis"
    },
    {
        "query": "physical motion and thinking speed",
        "mode": "synthesis",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Cross-Domain Synthesis"
    },
    {
        "query": "how does music state connect to stable node",
        "mode": "synthesis",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Cross-Domain Synthesis"
    },
    {
        "query": "integrating learning systems and neuroscience",
        "mode": "synthesis",
        "scope": {"namespaces": ["chatgpt_export", "refined_notes"]},
        "category": "Cross-Domain Synthesis"
    },
    # Project Questions
    {
        "query": "Android learning integration",
        "mode": "recall",
        "scope": {"namespaces": ["chatgpt_export"]},
        "category": "Project Questions"
    },
    {
        "query": "Astro framework site curriculum",
        "mode": "recall",
        "scope": {"namespaces": ["chatgpt_export"]},
        "category": "Project Questions"
    },
    {
        "query": "Icy-Clear-Research repository structure",
        "mode": "recall",
        "scope": {"namespaces": ["chatgpt_export"]},
        "category": "Project Questions"
    },
    {
        "query": "Spotify playlist integration details",
        "mode": "recall",
        "scope": {"namespaces": ["chatgpt_export"]},
        "category": "Project Questions"
    },
    {
        "query": "concept map of thinking systems",
        "mode": "mapping",
        "scope": {"namespaces": ["refined_notes"]},
        "category": "Project Questions"
    }
]

def main():
    print(f"=== STARTING INTEGRATION VALIDATION ({len(TEST_SUITE)} queries) ===")
    
    report_path = Path("evaluation_logs") / "checkpoint_integration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Integration Validation Report\n\n")
        f.write("This report validates the end-to-end performance of the Memory, Scope, and Cognition layers.\n")
        f.write("It contains evaluations of 30 representative queries from historical notes and exports.\n\n")
        f.write("## Validation Metrics Summary Table\n\n")
        f.write("| Index | Query | Category | Mode | Status | Top Match (Dist) | Output Length |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        
    for idx, t in enumerate(TEST_SUITE, start=1):
        print(f"\n[{idx}/{len(TEST_SUITE)}] Query: '{t['query']}' | Mode: '{t['mode']}'...")
        start_time = time.time()
        
        status = "Success"
        top_dist = "N/A"
        out_len = 0
        res = None
        error_msg = ""
        
        try:
            res = CognitionLayer.execute_cognition_pipeline(
                query=t["query"],
                mode=t["mode"],
                scope=t["scope"],
                model_name="gemma3:12b"
            )
            out_len = len(res["answer"])
            if res["sources"]:
                top_dist = f"{res['sources'][0]['distance']:.3f}"
        except Exception as e:
            status = f"Failed ({e})"
            error_msg = str(e)
            
        elapsed = time.time() - start_time
        print(f"-> Result: {status} in {elapsed:.2f}s")
        
        # Append row to the summary table
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"| {idx} | **{t['query']}** | {t['category']} | `{t['mode']}` | {status} | {top_dist} | {out_len} chars |\n")
            
        # Append detailed block at the bottom of the file
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n### Query {idx}: {t['query']}\n\n")
            f.write(f"- **Category**: {t['category']}\n")
            f.write(f"- **Mode**: `{t['mode']}`\n")
            f.write(f"- **Scope**: `{t['scope']}`\n")
            f.write(f"- **Status**: {status}\n")
            f.write(f"- **Execution Time**: {elapsed:.2f}s\n\n")
            
            if res:
                f.write("#### 1. Retrieved & Scoped Chunks\n")
                if res["sources"]:
                    for s in res["sources"]:
                        f.write(f"- **{s['title']}** (Source: `{s['source']}`, Path: `{s['path']}`, Distance: `{s['distance']:.3f}`)\n")
                else:
                    f.write("*No sources retrieved.*\n")
                
                f.write("\n#### 2. Excluded Chunks Diagnostics\n")
                if res["excluded_sources"]:
                    for ex in res["excluded_sources"][:3]: # top 3 exclusions
                        f.write(f"- Excluded: **{ex['title']}** (Source: `{ex['source']}`) -> Reason: `{ex['reason']}`\n")
                else:
                    f.write("*No exclusions.*\n")
                    
                f.write("\n#### 3. Intermediate Interpreted Signals\n")
                f.write(f"```text\n{res['interpreted_signals'].strip()}\n```\n")
                
                f.write("\n#### 4. Final Answer\n")
                f.write(f"{res['answer'].strip()}\n\n")
            else:
                f.write(f"**Error Details**: `{error_msg}`\n\n")
                
            f.write("---\n")
            
    print(f"\nValidation completed. Incremental report finished at {report_path}")

if __name__ == "__main__":
    main()
