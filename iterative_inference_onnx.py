import numpy as np
import tritonclient.http as httpclient
from transformers import BlipProcessor
from PIL import Image
import requests

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
eos_token_id = processor.tokenizer.eos_token_id
max_length = 20

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
text_prompt = "a photograph of"

inputs = processor(raw_image, text_prompt, return_tensors="pt")
pixel_values = inputs["pixel_values"].numpy()      # (1, 3, 384, 384)
input_ids = inputs["input_ids"].numpy()              # (1, seq_len)
attention_mask = inputs["attention_mask"].numpy()    # (1, seq_len)

# Triton HTTP client
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

# Iterative decoding loop
# Feeds the outputs of one pass of inference back to the model as input tokens
for _ in range(max_length - input_ids.shape[1]):
    infer_inputs = [
        httpclient.InferInput("pixel_values", pixel_values.shape, "FP32"),
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
    ]
    infer_inputs[0].set_data_from_numpy(pixel_values)
    infer_inputs[1].set_data_from_numpy(input_ids)
    infer_inputs[2].set_data_from_numpy(attention_mask)

    # logit outputs from onnx 
    infer_outputs = [httpclient.InferRequestedOutput("logits")]

    # Run the inference request.
    results = triton_client.infer("base_model_cpu", inputs=infer_inputs, outputs=infer_outputs)
    logits = results.as_numpy("logits")  # Shape: (1, seq_len, vocab_size)

    # Extract the logits for the last token.
    next_token_logits = logits[:, -1, :]  # (1, vocab_size)
    next_token = np.argmax(next_token_logits, axis=-1)  # (1,)

    # Append the predicted token to the sequence.
    next_token = next_token[:, None]  # Reshape to (1, 1)
    input_ids = np.concatenate([input_ids, next_token], axis=1)
    # Also update the attention mask.
    new_mask = np.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
    attention_mask = np.concatenate([attention_mask, new_mask], axis=1)

    # Stop if the EOS token is generated.
    if next_token[0, 0] == eos_token_id:
        break

# Decode the generated token sequence.
caption = processor.decode(input_ids[0], skip_special_tokens=True)
print("Triton iterative decoding caption:")
print(caption)



