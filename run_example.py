import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

sys.path.insert(0, "/mnt/nvme2/zys/qwevl_algo")
from qwen3vl_improved import Qwen3VLForConditionalGeneration
from qwen3vl_improved import Qwen3VLProcessor
from qwen3vl_improved import adapt_weights_for_global_token

MODEL_PATH = "/mnt/nvme2/zys/models/Qwen3-VL-4B-Instruct"

# ignore_mismatched_sizes=True: skip layers whose shapes changed (merger, global_proj).
# adapt_weights_for_global_token then fills those layers with principled init values.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto", ignore_mismatched_sizes=True
)
adapt_weights_for_global_token(model, MODEL_PATH)

processor = Qwen3VLProcessor.from_pretrained(MODEL_PATH)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
