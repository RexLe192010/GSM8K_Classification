# import numpy as np
# import torch
# import gc
# from fastchat.serve.inference import prepare_logits_processor, partial_stop
# from collections.abc import Iterable
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )
# from peft import PeftModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
# from src.utils import find_marker_content, ensure_end_with_marker
# import logging
# LOGGER = logging.getLogger(__name__)


# def load_model(model_path, device, load_in_8bit=False, low_cpu_mem_usage=True, resume_lora_from_path=None):
#     if device == "cpu":
#         kwargs = {"torch_dtype": torch.float32}
#     elif device == "cuda":
#         kwargs = {"torch_dtype": torch.float16}
#     else:
#         raise ValueError(f"Invalid device: {device}")
#     kwargs['load_in_8bit'] = load_in_8bit
#     kwargs['low_cpu_mem_usage'] = low_cpu_mem_usage
#     kwargs['device_map'] = 'auto'
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         **kwargs,
#     )
#     if resume_lora_from_path:
#         print("Will load lora from {}".format(resume_lora_from_path))
#         model = PeftModel.from_pretrained(
#             model,
#             resume_lora_from_path,
#             torch_dtype=torch.float16,
#         )
#     # if device == "cuda":
#     #     model.to(device)
#     return model, tokenizer

# @torch.inference_mode()
# def generate_stream(
#     model, tokenizer, params, device, context_len=2048, stream_interval=2, hide_stop_str=False,
# ):
#     prompt = params["prompt"]
#     len_prompt = len(prompt)
#     temperature = float(params.get("temperature", 1.0))
#     repetition_penalty = float(params.get("repetition_penalty", 1.0))
#     top_p = float(params.get("top_p", 1.0))
#     top_k = int(params.get("top_k", -1))  # -1 means disable
#     max_new_tokens = int(params.get("max_new_tokens", 256))
#     stop_str = params.get("stop", None)
#     echo = bool(params.get("echo", True))
#     stop_token_ids = params.get("stop_token_ids", None) or []
#     stop_token_ids.append(tokenizer.eos_token_id)

#     logits_processor = prepare_logits_processor(
#         temperature, repetition_penalty, top_p, top_k
#     )

#     input_ids = tokenizer(prompt).input_ids
#     input_echo_len = len(input_ids)
#     output_ids = list(input_ids)

#     if model.config.is_encoder_decoder:
#         max_src_len = context_len
#     else:
#         max_src_len = context_len - max_new_tokens - 8

#     input_ids = input_ids[-max_src_len:]

#     if model.config.is_encoder_decoder:
#         encoder_output = model.encoder(
#             input_ids=torch.as_tensor([input_ids], device=device)
#         )[0]
#         start_ids = torch.as_tensor(
#             [[model.generation_config.decoder_start_token_id]],
#             dtype=torch.int64,
#             device=device,
#         )

#     past_key_values = out = None
#     for i in range(max_new_tokens):
#         if i == 0:
#             if model.config.is_encoder_decoder:
#                 out = model.decoder(
#                     input_ids=start_ids,
#                     encoder_hidden_states=encoder_output,
#                     use_cache=True,
#                 )
#                 logits = model.lm_head(out[0])
#             else:
#                 out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
#                 logits = out.logits
#             past_key_values = out.past_key_values
#         else:
#             if model.config.is_encoder_decoder:
#                 out = model.decoder(
#                     input_ids=torch.as_tensor([[token]], device=device),
#                     encoder_hidden_states=encoder_output,
#                     use_cache=True,
#                     past_key_values=past_key_values,
#                 )

#                 logits = model.lm_head(out[0])
#             else:
#                 out = model(
#                     input_ids=torch.as_tensor([[token]], device=device),
#                     use_cache=True,
#                     past_key_values=past_key_values,
#                 )
#                 logits = out.logits
#             past_key_values = out.past_key_values

#         if logits_processor:
#             if repetition_penalty > 1.0:
#                 tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
#             else:
#                 tmp_output_ids = None
#             last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
#         else:
#             last_token_logits = logits[0, -1, :]

#         if device == "mps":
#             # Switch to CPU by avoiding some bugs in mps backend.
#             last_token_logits = last_token_logits.float().to("cpu")

#         if temperature < 1e-5 or top_p < 1e-8:  # greedy
#             token = int(torch.argmax(last_token_logits))
#         else:
#             probs = torch.softmax(last_token_logits, dim=-1)
#             token = int(torch.multinomial(probs, num_samples=1))

#         output_ids.append(token)

#         if token in stop_token_ids:
#             stopped = True
#         else:
#             stopped = False

#         if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
#             if echo:
#                 tmp_output_ids = output_ids
#                 rfind_start = len_prompt
#             else:
#                 tmp_output_ids = output_ids[input_echo_len:]
#                 rfind_start = 0

#             output = tokenizer.decode(
#                 tmp_output_ids,
#                 skip_special_tokens=True,
#                 spaces_between_special_tokens=False,
#             )

#             partially_stopped = False
#             if stop_str:
#                 if isinstance(stop_str, str):
#                     pos = output.rfind(stop_str, rfind_start)
#                     if pos != -1:
#                         if hide_stop_str:
#                             output = output[:pos]
#                         stopped = True
#                     else:
#                         partially_stopped = partial_stop(output, stop_str)
#                 elif isinstance(stop_str, Iterable):
#                     for each_stop in stop_str:
#                         pos = output.rfind(each_stop, rfind_start)
#                         if pos != -1:
#                             if hide_stop_str:
#                                 output = output[:pos]
#                             stopped = True
#                             break
#                         else:
#                             partially_stopped = partial_stop(output, each_stop)
#                             if partially_stopped:
#                                 break
#                 else:
#                     raise ValueError("Invalid stop field type.")

#             # prevent yielding partial stop sequence
#             if not partially_stopped:
#                 yield {
#                     "text": output,
#                     "usage": {
#                         "prompt_tokens": input_echo_len,
#                         "completion_tokens": i,
#                         "total_tokens": input_echo_len + i,
#                     },
#                     "finish_reason": None,
#                 }

#         if stopped:
#             break

#     # finish stream event, which contains finish reason
#     if i == max_new_tokens - 1:
#         finish_reason = "length"
#     elif stopped:
#         finish_reason = "stop"
#     else:
#         finish_reason = None

#     yield {
#         "text": output,
#         "usage": {
#             "prompt_tokens": input_echo_len,
#             "completion_tokens": i,
#             "total_tokens": input_echo_len + i,
#         },
#         "finish_reason": finish_reason,
#     }

#     # clean
#     del past_key_values, out
#     gc.collect()
#     torch.cuda.empty_cache()



class ChatGPT(object):

    def __init__(self, temp=0.7, request_timeout=60):
        self.chat = ChatOpenAI(
            temperature=temp,
            request_timeout=request_timeout
        )

    def __call__(self, prompt):
        messages = [
            SystemMessage(content="You are an helpful assistant."),
            HumanMessage(content=prompt)
        ]
        response = self.chat(messages)
        return response.content


# def get_answer(prompt, tokenizer, model, model_path, executor, max_tool_cnt=20):
#     tool_cnt = 0
#     suffix = ""
#     answer = None
#     while tool_cnt < max_tool_cnt:
#         params = {
#             "model": model_path,
#             "prompt": prompt + suffix,
#             "temperature": 0.5,
#             "repetition_penalty": 1.0,
#             "max_new_tokens": 1024,
#             "stop": ["</Thinking>", '</Tool>'],
#             "stop_token_ids": [],
#             "echo": False,
#         }
#         stream = generate_stream(
#             model,
#             tokenizer,
#             params,
#             'cuda',
#         )
#         for entry in stream:
#             LOGGER.debug(entry['text'])
#         assert entry["finish_reason"] == "stop"
#         text = ensure_end_with_marker(entry['text'], max_truncation=20, marker_names=['Thinking', 'Tool'])
#         suffix += text
#         if text.endswith("</Thinking>"):
#             answer = find_marker_content(text, marker_name="Answer", include_marker=False, reverse=True)
#             break
#         elif text.endswith("</Tool>"):
#             call_str = find_marker_content(text, marker_name="Tool", include_marker=False, reverse=True)
#             tool_is_error, tool_result = executor.run_with_str(call_str)
#             suffix += '\n<Result>{}</Result>\n'.format(tool_result)
#         else:
#             raise NotImplementedError
#         tool_cnt += 1
#     return {
#         'answer': answer,
#         'process_text': suffix,
#     }