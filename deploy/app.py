import gradio as gr
from utils import generate_image, generate_video

with gr.Blocks() as demo:
    gr.Markdown("Generate single image or video with stylization")
    framework = gr.Radio(
        ["UPST-NeRF", "StyleRF"], label="Choose a framework",
        value="UPST-NeRF"
    )
    scene = gr.Radio(
        ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], label="Choose a scene",
        value="lego"
    )
    style_image = gr.Image(label="Upload a style image", source='upload', image_mode="RGB", type='pil')
    
    with gr.Tab("Generate single image"):
        phi = gr.Slider(label="Value of phi", value=-30, minimum=-90, maximum=0, step=1)
        theta = gr.Slider(label="Value of theta", value=0, minimum=-180, maximum=180, step=1)
        
        image_button = gr.Button("Generate")
        output_image = gr.Image(height=800, width=800)
    with gr.Tab("Generate video"):
        video_button = gr.Button("Generate")
        output_video = gr.Video()

    image_button.click(generate_image, inputs=[framework, scene, style_image, phi, theta], outputs=output_image)
    video_button.click(generate_video, inputs=[framework, scene, style_image], outputs=output_video)

demo.queue().launch(share=True)
