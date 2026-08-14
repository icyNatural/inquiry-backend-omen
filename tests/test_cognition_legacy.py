from app.legacy.cognition_layer_original import CognitionLayer

TESTS = [
    {
        "query": "how can I refine my thinking and reduce mental loops?",
        "mode": "recall",
        "scope": {"namespaces": ["refined_notes"]},
        "description": "Recall mode on thinking refinement"
    },
    {
        "query": "how do attention and focus practices connect?",
        "mode": "synthesis",
        "scope": {"namespaces": ["refined_notes", "chatgpt_export"]},
        "description": "Synthesis mode on focus systems"
    },
    {
        "query": "what is elite beta thinking and how does it work?",
        "mode": "framework",
        "scope": {"namespaces": ["refined_notes"]},
        "description": "Framework mode on elite beta thinking"
    },
    {
        "query": "give me the raw notes regarding language behavior",
        "mode": "raw",
        "scope": {"namespaces": ["refined_notes"], "domains": ["language_system"]},
        "description": "Raw mode on language system notes"
    },
    {
        "query": "what does flow mean in my notes?",
        "mode": "grounding",
        "scope": {"namespaces": ["refined_notes", "chatgpt_export"]},
        "description": "Grounding mode to resolve ambiguity of 'flow'"
    },
    {
        "query": "how does zero-state thinking relate to attention shaping?",
        "mode": "mapping",
        "scope": {"namespaces": ["refined_notes"]},
        "description": "Mapping mode to draw concept relationships"
    }
]

def main():
    print("=== STARTING COGNITION LAYER AND 6 MODES VALIDATION ===")
    
    report_path = Path("evaluation_logs") / "cognition_modes_validation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Cognition Layer Validation Report (6 Modes)\n\n")
        f.write("This report validates the 6 cognition modes: Recall, Synthesis, Framework, Raw, Grounding, and Mapping.\n\n")
        
        for i, t in enumerate(TESTS, start=1):
            print(f"\n[{i}/6] Running mode '{t['mode']}' for query: '{t['query']}'...")
            
            try:
                res = CognitionLayer.execute_cognition_pipeline(
                    query=t["query"],
                    mode=t["mode"],
                    scope=t["scope"],
                    model_name="gemma3:12b"
                )
                
                print(f"[{i}/6] Success! Answer generated using {res['model']}")
                
                f.write(f"## Test {i}: {t['description']}\n\n")
                f.write(f"- **Query**: `{t['query']}`\n")
                f.write(f"- **Mode**: `{res['mode']}`\n")
                f.write(f"- **Model Used**: `{res['model']}`\n")
                f.write(f"- **Scope**: `{res['scope']}`\n\n")
                
                f.write("### 1. Selected Sources\n")
                for s in res["sources"]:
                    f.write(f"- **{s['title']}** (Namespace: `{s['source']}`, Path: `{s['path']}`, Dist: `{s['distance']:.3f}`)\n")
                f.write("\n")
                
                f.write("### 2. Intermediate Interpreted Signals\n")
                f.write(f"```text\n{res['interpreted_signals'].strip()}\n```\n\n")
                
                f.write("### 3. Final Answer Output\n")
                f.write(f"{res['answer'].strip()}\n\n")
                f.write("---\n\n")
                
            except Exception as e:
                print(f"[{i}/6] Error: {e}")
                f.write(f"## Test {i}: {t['description']} (FAILED)\n\n")
                f.write(f"- **Query**: `{t['query']}`\n")
                f.write(f"- **Error**: `{e}`\n\n")
                f.write("---\n\n")
                
    print(f"\nAll tests completed. Report written to {report_path}")

if __name__ == "__main__":
    main()
