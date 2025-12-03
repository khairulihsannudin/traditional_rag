# src/run.py
import argparse
import asyncio
from datetime import datetime
import json

from pathlib import Path
from workflow import app
import logging

def make_serializable(obj):
    """Convert non-serializable objects to serializable forms."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, 'content'):  # For LangChain messages (e.g., HumanMessage, AIMessage)
        return {"type": obj.__class__.__name__, "content": obj.content}
    elif hasattr(obj, '__dict__'):  # For other custom objects
        return {"type": obj.__class__.__name__, **obj.__dict__}
    else:
        try:
            json.dumps(obj)  # Test if serializable
            return obj
        except TypeError:
            return str(obj)  # Fallback to string


async def main():
    """The main function is to run the agent."""
    logging.getLogger('mcp_use').propagate = False
    
    from workflow import enable_instrumentation
    enable_instrumentation(True)
    
    parser = argparse.ArgumentParser(description="Run Multi-Agent with questions.")
    parser.add_argument("question", type=str, help="Questions to ask agents.")
    parser.add_argument("--ground-truth", type=str, default=None, help="Expected answer for comparison (optional).")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save run data.")
    args = parser.parse_args()

    initial_state = {
        "question": args.question,
        "original_question": args.question,
        "messages": [("human", args.question)],
        "_timing_data": {},
    }

    config = {"recursion_limit": 30}

    final_result = await app.ainvoke(initial_state, config=config)
    serializable_state = make_serializable({k: v for k, v in final_result.items() if k not in ['_timing_data']})
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "question": args.question,
        "answer": final_result.get('answer'),
        "ground_truth": args.ground_truth,
        "timing_data": final_result.get('_timing_data', {}),
        "contexts": {
            "log_vector_context": final_result.get('log_vector_context'),
        },
        "state": serializable_state,
    }
    
    
    if args.ground_truth:
        # Basic similarity check (you can enhance with metrics from run_evaluation.py)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, run_data["answer"], args.ground_truth).ratio()
        run_data["similarity_to_ground_truth"] = similarity
    
    # Save to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    filename = f"output.json"
    file_path = output_dir / filename
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []  
            except json.JSONDecodeError:
                existing_data = []  
    else:
        existing_data = []
    existing_data.append(run_data)
    
    next_id = max([item.get('id', 0) for item in existing_data], default=0) + 1
    run_data['id'] = next_id
    
    # Append new run data
    existing_data.append(run_data)
    
    # Save back to file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
            
    print(f"\n--- Run Data Saved to: {output_dir / filename} ---")
    print("\n--- Final Answer ---")
    print(final_result.get('answer'))
    
    if args.ground_truth:
        print(f"\n--- Similarity to Ground Truth: {run_data['similarity_to_ground_truth']:.2f} ---")

if __name__ == "__main__":
    asyncio.run(main())