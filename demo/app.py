"""Gradio demo for the Hugging Face Space. Owner: member 5."""
import gradio as gr


def run(before, after):
    return "TODO: call pipeline.run"


gr.Interface(run, [gr.Image(), gr.Image()], gr.Textbox()).launch()
