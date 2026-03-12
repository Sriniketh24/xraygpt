"""
Gradio demo for XRayGPT.

Provides a simple web interface for uploading chest X-ray images
and viewing generated radiology reports.

Usage:
    python demo/app.py --checkpoint checkpoints/best_model.pt
"""

import argparse
from pathlib import Path

import gradio as gr

from src.inference.generate import ReportGenerator

DISCLAIMER = (
    "**DISCLAIMER**: This is a research/educational tool and is NOT intended "
    "for clinical diagnosis or medical decision-making. The generated reports "
    "are AI-produced and may contain errors."
)

DESCRIPTION = """
# XRayGPT: Chest X-Ray Report Generator

Upload a chest X-ray image to generate an AI-produced radiology-style report.

This system uses a Vision Transformer (ViT) to encode the image and GPT-2
to generate the report text.
"""


def create_demo(generator: ReportGenerator) -> gr.Blocks:
    """Build the Gradio interface."""

    def generate_report(
        image,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if image is None:
            return "Please upload a chest X-ray image."

        # Save temp image and run inference
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            report = generator.predict(
                f.name,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )

        return report

    with gr.Blocks(
        title="XRayGPT Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(DESCRIPTION)
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Chest X-Ray",
                    height=400,
                )
                with gr.Accordion("Generation Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more conservative",
                    )
                    max_tokens = gr.Slider(
                        minimum=32, maximum=256, value=128, step=16,
                        label="Max Tokens",
                        info="Maximum length of generated report",
                    )
                generate_btn = gr.Button("Generate Report", variant="primary")

            with gr.Column(scale=1):
                report_output = gr.Textbox(
                    label="Generated Radiology Report",
                    lines=12,
                    interactive=False,
                )

        # Example images (if available)
        example_dir = Path("outputs/samples")
        if example_dir.exists():
            examples = [
                str(p) for p in sorted(example_dir.glob("*.png"))[:3]
            ]
            if examples:
                gr.Examples(
                    examples=[[ex] for ex in examples],
                    inputs=image_input,
                    label="Example X-Rays",
                )

        generate_btn.click(
            fn=generate_report,
            inputs=[image_input, temperature, max_tokens],
            outputs=report_output,
        )

        gr.Markdown("---")
        gr.Markdown(
            "Built with ViT + GPT-2 | "
            "[GitHub](https://github.com/yourusername/xraygpt) | "
            "Research use only"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="XRayGPT Demo")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config file override",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port for Gradio server",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public share link",
    )
    args = parser.parse_args()

    print("Loading model...")
    generator = ReportGenerator.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )
    print("Model loaded!")

    demo = create_demo(generator)
    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
