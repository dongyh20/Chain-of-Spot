import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
# import shortuuid
import uuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path, expand2square

from PIL import Image
import math
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if image_processor is None:
    # qwen
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        image_transform = transforms.Compose([
            transforms.Resize(
                (448, 448),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        
        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            
            qs = DEFAULT_IMAGE_TOKEN + '\n' + 'To answer the question: "' + qs + '", what is the region of interest in the image?'
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            if image_processor is not None:
                image_tensor = process_images([image], image_processor, model.config)[0]
            else:
                image_tensor = image_transform(image)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            del output_ids
            torch.cuda.empty_cache()

            qs1 = DEFAULT_IMAGE_TOKEN + '\n' + 'To answer the question: "' + cur_prompt + '", what is the region of interest in the image?'
            qs2 = f'The region of interest in the image is <image>\n. Answer the question: "{cur_prompt}"'
            if args.single_pred_prompt:
                qs2 = qs2 + '\n' + "Answer with the option's letter from the given choices directly."
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs1)
            conv.append_message(conv.roles[1], outputs)
            conv.append_message(conv.roles[0], qs2)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            
            image = Image.fromarray(np.array(image)).convert('RGB')
            try:
                outputs_num = outputs[1:-1].split(', ')
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                x, y = image.size
                image = image.crop((x * float(outputs_num[0]), y * float(outputs_num[1]), x * float(outputs_num[2]), y * float(outputs_num[3])))
            except:
                print('Error in cropping the image')
            image_tensor_crop = process_images([image], image_processor, model.config)[0]

            image_tensor_crop = image_tensor_crop.to(dtype=torch.float16, device='cuda', non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            image_tensor = torch.cat((image_tensor.unsqueeze(0), image_tensor_crop.unsqueeze(0)), dim=1)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            
            # ans_id = shortuuid.uuid()
            ans_id = str(uuid.uuid4())
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
