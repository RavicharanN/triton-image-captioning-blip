import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Define a wrapper to expose the forward pass (which returns logits)
class BLIPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.model(pixel_values=pixel_values,
                             input_ids=input_ids,
                             attention_mask=attention_mask)
        return outputs.logits

if __name__ == "__main__":
    # Load the processor and BLIP model.
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.eval()

    # Download an example image and prepare it.
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    text_prompt = "a photography of"

    # Use the processor to create dummy inputs for export.
    dummy_inputs = processor(raw_image, text_prompt, return_tensors="pt")
    dummy_pixel_values = dummy_inputs["pixel_values"]
    dummy_input_ids = dummy_inputs["input_ids"]
    dummy_attention_mask = dummy_inputs["attention_mask"]

    # Wrap the model.
    wrapper = BLIPWrapper(model)

    # Define dynamic axes so that batch size and sequence length are flexible.
    dynamic_axes = {
        "pixel_values": {0: "batch_size"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"}
    }

    # Export to ONNX.
    onnx_filename = "model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_pixel_values, dummy_input_ids, dummy_attention_mask),
        onnx_filename,
        input_names=["pixel_values", "input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True
    )
    print(f"Exported model to {onnx_filename}")

