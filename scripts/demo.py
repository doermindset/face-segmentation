import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from utils.utils import get_mask
def show_mask(input_image):

    model = torch.jit.load(rf"C:\work\an 3\dl\face-segmentation\checkpoints\unet_scripted_model.pt")

    preprocess = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.LANCZOS),
        transforms.ToTensor()])

    input_image = preprocess(input_image)
    batch_image = input_image.unsqueeze(0)
    with torch.no_grad():
        output = model(batch_image)

    mask = get_mask(output.squeeze(), 3)
    pil_image = Image.fromarray(mask)

    return pil_image

def image_mod(image):
    return image.rotate(45)

demo = gr.Interface(
    show_mask,
    gr.Image(type="pil"),
    gr.Image(height=255, width=255),
    examples=[
        rf"C:\work\an 3\dl\face-segmentation\data\lfw_dataset\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg",
        rf"C:\work\an 3\dl\face-segmentation\data\lfw_dataset\lfw_funneled\Aaron_Peirsol\Aaron_Peirsol_0002.jpg"
    ],
)

if __name__ == "__main__":
    demo.launch()
