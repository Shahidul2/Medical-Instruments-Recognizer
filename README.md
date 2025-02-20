# Medical Instrument Classifier
A TL-based vision model that recognize **10 specific medical instruments** from images.  

---

## What It Can Recognize
This model can classify **10 types of medical instruments**:
- Scalpel, Thermometer, Forceps, Bone Saw, Reflex Hammer  
- Ultrasound Probe, Defibrillator, Otoscope, ECG Machine, Glucometer  

---

## How It Works
 **Dataset:** Collected using DuckDuckGo, cleaned using FastAI’s `ImageClassifierCleaner`.  
 **Training:** Fine-tuned a **DenseNet121** model, achieving **~90% accuracy**.  
 **Deployment:** Hosted on **Hugging Face Spaces** with **Gradio** for easy interaction.  
 **Web API:** Integrated with **GitHub Pages** for real-time classification.  

---

## Try It Out
- **Gradio App** → [Test the Model](https://huggingface.co/spaces/Shahidul279/medins-recognizer)  
- **GitHub Pages API Integration** → [Live Web App](https://shahidul2.github.io/Medical-Instruments-Recognizer/)  


## **Tech Stack**
- **FastAI + PyTorch** for training  
- **Gradio + Hugging Face Spaces** for deployment  
- **GitHub Pages** for API integration  
- **Google Colab** for development  

