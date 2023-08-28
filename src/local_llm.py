import numpy as np
import torch
import gc
from fastchat.serve.inference import prepare_logits_processor, partial_stop
from collections.abc import Iterable
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import contextlib
from datasets import Dataset
import random
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import os

os.environ["WANDB_DISABLED"] = "true"

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

class LocalLLM(object):
    
    def __init__(
        self, 
        model_path, 
        device='cuda', 
        load_in_8bit=True, 
        low_cpu_mem_usage=True, 
        context_window_size=2048,
    ):
        if device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
        else:
            raise ValueError(f"Invalid device: {device}")
        kwargs['load_in_8bit'] = load_in_8bit
        kwargs['low_cpu_mem_usage'] = low_cpu_mem_usage
        kwargs['device_map'] = 'auto'
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )
        tokenizer.pad_token_id = (
            0  # TBD: check this
        )
        tokenizer.padding_side = "left"  # Allow batched inference
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_peft_model = False
        self.is_lora_disabled = False
        self.context_window_size = context_window_size
    
    def add_and_enable_lora(self, lora_path, lora_name='default'):
        # this is only for inferencing, cannot add_lora then training
        # for training, use the resume from lora parameter!
        if self.is_peft_model is False:
            # first time load
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                torch_dtype=torch.float16,
                adapter_name=lora_name,
            )
            self.is_peft_model = True
        else:
            self.model.load_adapter(
                lora_path,
                lora_name,
            )
        self.is_lora_disabled = False
    
    def disable_lora(self):
        self.is_lora_disabled = True
        
    def inference_ctx(self):
        if self.is_peft_model and self.is_lora_disabled:
            ctx = self.model.disable_adapter()
        else:
            ctx = contextlib.nullcontext()
        return ctx
    
    def set_and_enable_lora(self, lora_name):
        self.model.set_adapter(lora_name)
        self.is_lora_disabled = False
    
    def train_and_save(
        self,
        text_sample_list,
        batch_size: int = 4,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        learning_rate: float = 3e-4,
        lora_r: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        resume_from_checkpoint: str = None, # could be lora or normal ckp
        output_dir = './ret/',
        max_sample_num=None,
    ):
        assert self.is_peft_model is False
        assert batch_size % micro_batch_size == 0
        gradient_accumulation_steps = batch_size // micro_batch_size

        # prepare model
        model = prepare_model_for_int8_training(self.model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        if resume_from_checkpoint:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
        
        # create dataset
        dataset_lines = [
            text_sample.get_train_sample(self.tokenizer, self.context_window_size) 
            for text_sample in text_sample_list
        ]
        tokenized_dataset = Dataset.from_list(dataset_lines)
        print(f"Loaded {len(tokenized_dataset)} samples; Randomly selected 10 samples from the dataset:")
        random_ids = random.sample(list(range(len(tokenized_dataset))), 10)
        for index in random_ids:
            sample = tokenized_dataset[index]
            print("-----sample {} starts-----".format(index))
            print(self.tokenizer.decode(sample['input_ids']))
            print("------sample {} ends------".format(index))
            print("")
        
        train_data = tokenized_dataset.shuffle()
        if max_sample_num is not None:
            print(f"Only select {max_sample_num} samples!")
            selected_ids = random.sample(list(range(len(tokenized_dataset))), max_sample_num)
            train_data = train_data.select(selected_ids)
        val_data = None
        val_set_size = 0

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=100 if val_set_size > 0 else None,
                save_steps=200,
                output_dir=output_dir,
                save_total_limit=20,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=None,
                group_by_length=False,
                report_to=None,
                run_name=None,
            ),
            callbacks=[SavePeftModelCallback],
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        model.config.use_cache = False
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        model.save_pretrained(output_dir)
        
    @torch.inference_mode()
    def logit_inference(self, prompt, target_tokens=['A', 'B'], verbose=False):
        # control whether disable lora!
        with self.inference_ctx():
            target_ids = []
            for c in target_tokens:
                token_ids = self.tokenizer.encode(c, add_special_tokens=False)
                assert len(token_ids) == 1
                target_ids.append(token_ids[0])
            input_ids = self.tokenizer(prompt).input_ids
            out = self.model(torch.as_tensor([input_ids], device=self.device), use_cache=False)
            logits = out.logits
            last_token_logits = logits[0, -1, :]
            token_prob = last_token_logits.cpu().detach().numpy()
            if verbose:
                for top_id in np.argsort(token_prob)[::-1][:10]:
                    print(self.tokenizer.decode(top_id), token_prob[top_id])
            target_logits = [token_prob[id] for id in target_ids]
            return target_logits
    
    def inference(self, *args, **kwargs):
        for ret in self.streaming_inference(*args, **kwargs):
            pass
        return ret['text']
    
    @torch.inference_mode()
    def streaming_inference(
        self, prompt, temperature=0.5, repetition_penalty=1.0, top_p=1.0, top_k=-1, 
        max_new_tokens=256, stop_str=None, include_prompt_in_response=False, stop_token_ids=[],
        stream_interval=2, hide_stop_str=False, free_mem=False,
    ):
        # control whether disable lora!
        with self.inference_ctx():
            len_prompt = len(prompt)
            stop_token_ids.append(self.tokenizer.eos_token_id)
            logits_processor = prepare_logits_processor(
                temperature, repetition_penalty, top_p, top_k
            )

            input_ids = self.tokenizer(prompt).input_ids
            input_echo_len = len(input_ids)
            output_ids = list(input_ids)

            if self.model.config.is_encoder_decoder:
                max_src_len = self.context_window_size
            else:
                max_src_len = self.context_window_size - max_new_tokens - 8

            input_ids = input_ids[-max_src_len:]

            if self.model.config.is_encoder_decoder:
                encoder_output = self.model.encoder(
                    input_ids=torch.as_tensor([input_ids], device=self.device)
                )[0]
                start_ids = torch.as_tensor(
                    [[self.model.generation_config.decoder_start_token_id]],
                    dtype=torch.int64,
                    device=self.device,
                )

            past_key_values = out = None
            for i in range(max_new_tokens):
                if i == 0:
                    if self.model.config.is_encoder_decoder:
                        out = self.model.decoder(
                            input_ids=start_ids,
                            encoder_hidden_states=encoder_output,
                            use_cache=True,
                        )
                        logits = self.model.lm_head(out[0])
                    else:
                        out = self.model(torch.as_tensor([input_ids], device=self.device), use_cache=True)
                        logits = out.logits
                    past_key_values = out.past_key_values
                else:
                    if self.model.config.is_encoder_decoder:
                        out = self.model.decoder(
                            input_ids=torch.as_tensor([[token]], device=self.device),
                            encoder_hidden_states=encoder_output,
                            use_cache=True,
                            past_key_values=past_key_values,
                        )

                        logits = self.model.lm_head(out[0])
                    else:
                        out = self.model(
                            input_ids=torch.as_tensor([[token]], device=self.device),
                            use_cache=True,
                            past_key_values=past_key_values,
                        )
                        logits = out.logits
                    past_key_values = out.past_key_values

                if logits_processor:
                    if repetition_penalty > 1.0:
                        tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                    else:
                        tmp_output_ids = None
                    last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
                else:
                    last_token_logits = logits[0, -1, :]

                if self.device == "mps":
                    # Switch to CPU by avoiding some bugs in mps backend.
                    last_token_logits = last_token_logits.float().to("cpu")

                if temperature < 1e-5 or top_p < 1e-8:  # greedy
                    token = int(torch.argmax(last_token_logits))
                else:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    token = int(torch.multinomial(probs, num_samples=1))

                output_ids.append(token)

                if token in stop_token_ids:
                    stopped = True
                else:
                    stopped = False

                if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                    if include_prompt_in_response:
                        tmp_output_ids = output_ids
                        rfind_start = len_prompt
                    else:
                        tmp_output_ids = output_ids[input_echo_len:]
                        rfind_start = 0

                    output = self.tokenizer.decode(
                        tmp_output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                    )

                    partially_stopped = False
                    if stop_str:
                        if isinstance(stop_str, str):
                            pos = output.rfind(stop_str, rfind_start)
                            if pos != -1:
                                if hide_stop_str:
                                    output = output[:pos]
                                stopped = True
                            else:
                                partially_stopped = partial_stop(output, stop_str)
                        elif isinstance(stop_str, Iterable):
                            for each_stop in stop_str:
                                pos = output.rfind(each_stop, rfind_start)
                                if pos != -1:
                                    if hide_stop_str:
                                        output = output[:pos]
                                    stopped = True
                                    break
                                else:
                                    partially_stopped = partial_stop(output, each_stop)
                                    if partially_stopped:
                                        break
                        else:
                            raise ValueError("Invalid stop field type.")

                    # prevent yielding partial stop sequence
                    if not partially_stopped:
                        yield {
                            "text": output,
                            "usage": {
                                "prompt_tokens": input_echo_len,
                                "completion_tokens": i,
                                "total_tokens": input_echo_len + i,
                            },
                            "finish_reason": None,
                        }

                if stopped:
                    break

            # finish stream event, which contains finish reason
            if i == max_new_tokens - 1:
                finish_reason = "length"
            elif stopped:
                finish_reason = "stop"
            else:
                finish_reason = None

            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": finish_reason,
            }

            # clean
            if free_mem:
                del past_key_values, out
                gc.collect()
                torch.cuda.empty_cache()

class Segment(object):
    
    def __init__(self, text, is_skip_training=False):
        self.text = text
        self.is_skip_training = is_skip_training

class TextSample(object):
    
    def __init__(self, segment_list):
        # segment and segment are separated by space
        self.segment_list = segment_list
        
    def get_text(self):
        return ' '.join([segment.text for segment in self.segment_list])
    
    def get_train_sample(self, tokenizer, cutoff_len):
        token_list = []
        label_list = []
        if tokenizer.add_bos_token:
            token_list.append(tokenizer.bos_token_id)
            label_list.append(tokenizer.bos_token_id)
        for segment in self.segment_list:
            segment_tokens = tokenizer(segment.text, add_special_tokens=False)['input_ids']
            token_list.extend(segment_tokens)
            if segment.is_skip_training:
                label_list.extend([-100] * len(segment_tokens))
            else:
                label_list.extend(segment_tokens)
        if tokenizer.add_eos_token:
            token_list.append(tokenizer.eos_token_id)
            label_list.append(tokenizer.eos_token_id)
        if len(token_list) > cutoff_len:
            token_list = token_list[:cutoff_len]
            label_list = label_list[:cutoff_len]
        attention_mask_list = [1] * len(label_list)
        return {
            'labels': label_list,
            'input_ids': token_list,
            'attention_mask': attention_mask_list,
        }