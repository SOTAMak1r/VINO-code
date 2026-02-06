import json
import torch
import fire
from transformers import AutoModelForImageTextToText, AutoProcessor

def main(
    json_path='',    
    output_path='',  
    ckpt_path='./checkpoints/Qwen/Qwen3-VL-4B-Instruct',
):
    # 加载模型
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt_path, dtype=torch.bfloat16, device_map="auto" 
    )

    processor = AutoProcessor.from_pretrained(ckpt_path)
    
    chat_template_new = "{% set image_counter = namespace(value=0) %}{% set video_counter = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant aimed at generating an output video caption. Based on the user's input command, provide a caption describing the video and output it directly. <|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_counter.value = image_counter.value + 1 %}Image {{ image_counter.value }}: <|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_counter.value = video_counter.value + 1 %}Video {{ video_counter.value }}: <|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    processor.chat_template = chat_template_new

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        prompt = item.get('caption', "")
        item['caption_raw'] = prompt 

        ref_image_paths = item.get('ref_image_paths', [])
        ref_video_path  = item.get('ref_video_path', None)

        content = []
        
        if isinstance(ref_image_paths, str):
            ref_image_paths = eval(ref_image_paths)
            
        for image_path in ref_image_paths:
            content.append({
                "type": "image",
                "image": image_path,
            })

        if ref_video_path:
            content.append({
                "type": 'video',
                "video": ref_video_path,
                "min_pixels": 256*32*32, 
                "max_pixels": 16384*32*32,
            })

        if prompt:
            content.append({
                "type": "text",
                "text": prompt
            })

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print(f"Generated: {output_text[0]}")
        item['caption'] = output_text[0]

    # 保存为 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    fire.Fire(main)