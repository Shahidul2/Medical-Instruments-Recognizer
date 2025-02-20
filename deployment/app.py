from fastai.vision.all import *
import gradio as gr

# Load trained model
model_path = "models/medins-recognizer-v2.pkl"
learn = load_learner(model_path)

def recognize_image(image):
    pred, idx, probs = learn.predict(image)
    labels = learn.dls.vocab  # Get correct labels from trained model since manual label was being mismatched
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Define Gradio interface
image = gr.Image(type="pil") 
label = gr.Label(num_top_classes=3)  #  Reduced to 3 top predictions for clarity

examples = [
    "test_images/01.jpg",
    "test_images/02.jpg",
    "test_images/03.jpg",
    "test_images/04.jpg"
]

iface = gr.Interface(
    fn=recognize_image, 
    inputs=image, 
    outputs=label, 
    examples=examples,
    title="Medical Instrument Classifier"
)

# Launch Gradio app
iface.launch()
