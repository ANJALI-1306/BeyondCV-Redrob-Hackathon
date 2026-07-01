import gradio as gr
import subprocess
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import sys
from typing import Tuple, Optional

PROJECT_ROOT = Path(__file__).parent
DEMO_DATASET = PROJECT_ROOT / "dataset" / "demo_candidates.jsonl"


def run_ranking(uploaded_file: Optional[str] = None, use_demo: bool = False) -> Tuple[str, Optional[str], Optional[pd.DataFrame], Optional[str]]:
    """
    Run the ranking pipeline using rank.py as a subprocess.

    Args:
        uploaded_file: Path to uploaded JSONL file (if provided)
        use_demo: Whether to use the demo dataset

    Returns:
        Tuple of (status_message, csv_path, dataframe, error_message)
    """
    temp_input_file = None

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False
    ) as tmp_output:
        output_csv = tmp_output.name

    try:
        if use_demo:
            candidates_path = DEMO_DATASET
            if not candidates_path.exists():
                return "Error: Demo dataset not found at dataset/demo_candidates.jsonl", None, None, "Demo dataset file missing"
        elif uploaded_file is not None:
            temp_input_file = tempfile.NamedTemporaryFile(
                suffix=".jsonl",
                delete=False
            )
            temp_input_file.close()
            source_path = (
                uploaded_file.name
                if hasattr(uploaded_file, "name")
                else str(uploaded_file)
            )

            shutil.copy(source_path, temp_input_file.name)
            candidates_path = Path(temp_input_file.name)
        else:
            return "Error: Please either upload a JSONL file or select Run Demo Dataset", None, None, "No input provided"

        if not candidates_path.exists():
            return f"Error: Candidates file not found at {candidates_path}", None, None, f"File not found: {candidates_path}"

        cmd = [
            sys.executable,
            "rank.py",
            "--sandbox",
            "--candidates", str(candidates_path),
            "--out", output_csv
        ]

        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )

        if not Path(output_csv).exists():
            return "Error: Output CSV was not generated", None, None, "Output file missing"

        df = pd.read_csv(output_csv)
        df_display = df[[
            "candidate_id",
            "rank",
            "score",
            "reasoning",
        ]]

        if temp_input_file is not None:
            Path(temp_input_file.name).unlink(missing_ok=True)

        return "Ranking completed successfully.", output_csv, df_display, None

    except subprocess.CalledProcessError as e:
        error_msg = f"Ranking failed with return code {e.returncode}\n\nSTDERR:\n{e.stderr}"
        return f"Error: Ranking failed", None, None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return f"Error: {str(e)}", None, None, error_msg
    finally:
        if temp_input_file is not None and Path(temp_input_file.name).exists():
            Path(temp_input_file.name).unlink(missing_ok=True)


def create_interface() -> gr.Blocks:
    """Create and return the Gradio interface."""

    with gr.Blocks(title="BeyondCV AI Candidate Ranking") as demo:
        gr.Markdown(
            """
            # BeyondCV AI Candidate Ranking System

            AI-powered Learning-to-Rank candidate ranking engine.

            Upload a JSONL file or run the built-in demo dataset.
            """
        )

        with gr.Row():
            with gr.Column():
                use_demo = gr.Checkbox(
                    label="Run Demo Dataset",
                    value=False,
                    info="Use built-in dataset/demo_candidates.jsonl"
                )

                uploaded_file = gr.File(
                    label="Upload JSONL File",
                    file_types=[".jsonl"],
                    visible=True
                )

                run_button = gr.Button("Run Ranking", variant="primary", size="lg")

            with gr.Column():
                status = gr.Textbox(
                    label="Status",
                    value="Waiting for input...",
                    interactive=False,
                    lines=3
                )

                error_box = gr.Textbox(
                    label="Logs / Errors",
                    lines=10,
                    interactive=False,
                    visible=False
                )

        results = gr.Dataframe(
            label="Ranking Results",
            interactive=False,
            wrap=True
        )

        output_csv = gr.File(
            label="Download Ranked CSV"
        )

        def handle_ranking(use_demo, uploaded_file):
            if not use_demo and uploaded_file is None:
                return (
                    "Please upload a JSONL file or enable Demo mode.",
                    None,
                    None,
                    gr.update(value="", visible=False),
                )

            status_message, csv_path, dataframe, logs = run_ranking(
                uploaded_file,
                use_demo
            )

            return (
                status_message,
                csv_path,
                dataframe,
                gr.update(
                    value=logs if logs else "",
                    visible=logs is not None
                )
            )

        run_button.click(
            fn=handle_ranking,
            inputs=[use_demo, uploaded_file],
            outputs=[
                status,
                output_csv,
                results,
                error_box
            ]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()