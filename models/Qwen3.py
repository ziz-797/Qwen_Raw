import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import Qwen3ForCausalLM, AutoTokenizer, Qwen3Config
from transformers import SiglipVisionConfig, SiglipVisionModel
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import evaluate


class Qwen3(pl.LightningModule):
    """
    Qwen3 model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        print(f'Loading vision encoder:{args.vision_model}')
        vis_config = SiglipVisionConfig.from_pretrained(args.vision_model)
        self.visual_encoder = SiglipVisionModel.from_pretrained(args.vision_model, config=vis_config)
        print("Freezing unused vision parameters (head/pooler)...")
        for name, param in self.visual_encoder.named_parameters():
            if "head" in name or "pooler" in name:
                param.requires_grad = False
                print(f"  -> Frozen: {name}")
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["q_proj", "v_proj"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading Qwen3')
        self.qwen3_tokenizer = AutoTokenizer.from_pretrained(args.qwen3_model, use_fast=False)
        # if self.qwen3_tokenizer.bos_token_id is None:
        #     im_start_id = self.qwen3_tokenizer.convert_tokens_to_ids("<|im_start|>")
        #     if im_start_id is None or im_start_id == self.qwen3_tokenizer.unk_token_id:
        #         im_start_id = self.qwen3_tokenizer.eos_token_id
        #     self.qwen3_tokenizer.bos_token_id = im_start_id
        qwen3_config = Qwen3Config.from_pretrained(args.qwen3_model)
        if args.low_resource:
            self.qwen3_model = Qwen3ForCausalLM.from_pretrained(
                args.qwen3_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                config=qwen3_config,
                device_map="auto"
            )
        else:
            self.qwen3_model = Qwen3ForCausalLM.from_pretrained(
                args.qwen3_model,
                torch_dtype=torch.bfloat16,
            )
        # if self.qwen3_tokenizer.pad_token_id is None:
        #     self.qwen3_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        #     self.qwen3_model.resize_token_embeddings(len(self.qwen3_tokenizer))
        if args.llm_use_lora:
            self.embed_tokens = self.qwen3_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.qwen3_model = get_peft_model(self.qwen3_model, peft_config)
            self.qwen3_model.print_trainable_parameters()
            print('Loading Qwen3 LoRA Done')         
        elif args.freeze_llm:
            self.embed_tokens = self.qwen3_model.get_input_embeddings()
            for name, param in self.qwen3_model.named_parameters():
                param.requires_grad = False
            print('Loading Frozen Qwen3 Done')
        else:
            self.embed_tokens = self.qwen3_model.get_input_embeddings()
            self.qwen3_model.gradient_checkpointing_enable() 
            print('Loading Trainable Qwen3 (Full Fine-tuning) Done')

        self.qwen3_proj = nn.Linear(vis_config.hidden_size, qwen3_config.hidden_size)
        self.layer_norm = nn.LayerNorm(qwen3_config.hidden_size)
        self.end_sym = self.qwen3_tokenizer.eos_token
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        self.rouge_metric = evaluate.load("rouge")
        self.bleu_metric = evaluate.load("bleu")
        self.meteor_metric = evaluate.load("meteor")
        if args.delta_file is not None:
            ckpt = torch.load(
                args.delta_file,
                map_location=torch.device(f'cuda:{torch.cuda.current_device()}'),
                weights_only=False,   # 关键：关闭安全模式
            )
            state_dict = ckpt['model']
            # state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        predictions = []
        references = []
        
        sorted_ids = sorted(ref.keys())
        for id in sorted_ids:
            if id in hypo:
                predictions.append(hypo[id][0])
                references.append(ref[id])
        
        final_scores = {}
        
        bleu1_results = self.bleu_metric.compute(predictions=predictions, references=references, max_order=1)
        final_scores["Bleu_1"] = bleu1_results["bleu"]
        
        bleu2_results = self.bleu_metric.compute(predictions=predictions, references=references, max_order=2)
        final_scores["Bleu_2"] = bleu2_results["bleu"]
        
        bleu3_results = self.bleu_metric.compute(predictions=predictions, references=references, max_order=3)
        final_scores["Bleu_3"] = bleu3_results["bleu"]
        
        bleu4_results = self.bleu_metric.compute(predictions=predictions, references=references, max_order=4)
        final_scores["Bleu_4"] = bleu4_results["bleu"]
        
        rouge_results = self.rouge_metric.compute(predictions=predictions, references=references, rouge_types=["rougeL"])
        final_scores["ROUGE_L"] = rouge_results["rougeL"]
        
        meteor_results = self.meteor_metric.compute(predictions=predictions, references=references)
        final_scores["METEOR"] = meteor_results["meteor"]
        
        return final_scores

    # def encode_img(self, images):
        # image_embeds = []
        # for image in images:
        #     device = image.device
        #     if self.hparams.global_only:
        #         image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
        #     else:
        #         image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
        #     image_embeds.append(image_embed)
            
        # image_embeds = torch.stack(image_embeds).mean(0)
        # inputs_qwen3 = self.qwen3_proj(image_embeds)
        # atts_qwen3 = torch.ones(inputs_qwen3.size()[:-1], dtype=torch.long).to(image.device)
        # return inputs_qwen3, atts_qwen3
    def encode_img(self, images):
        device = images.device

        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() == 5:
            if images.size(1) == 1:
                images = images.squeeze(1)
            else:
                images = images[:, 0]

        out = self.visual_encoder(pixel_values=images)

        if self.hparams.global_only:
            image_embeds = out.pooler_output.unsqueeze(1)
        else:
            image_embeds = out.last_hidden_state

        inputs_qwen3 = self.qwen3_proj(image_embeds)
        atts_qwen3 = torch.ones(inputs_qwen3.size()[:-1], dtype=torch.long, device=device)
        return inputs_qwen3, atts_qwen3


    # def prompt_wrap(self, img_embeds, atts_img):
    #     prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
    #     batch_size = img_embeds.shape[0]
    #     p_before, p_after = prompt.split('<ImageHere>')
    #     p_before_tokens = self.qwen3_tokenizer(
    #         p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
    #     p_after_tokens = self.qwen3_tokenizer(
    #         p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
    #     p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
    #     p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
    #     wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
    #     wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
    #     return wrapped_img_embeds, wrapped_atts_img


    def forward(self, samples):
        image = samples["pixel_values"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        prompt_text = samples["prompt"]
        label_text = samples["findings"]
        # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.qwen3_tokenizer.padding_side = "right"
        label_text  = [f + self.end_sym for f in label_text]
        label_tokens = self.qwen3_tokenizer(
            label_text,
            return_tensors=None,
            padding= False,
            truncation=True,
            add_special_tokens=False
        ).to(image[0].device)
        label_lens = [len(x) for x in label_tokens["input_ids"]]
        prompt_tokens = self.qwen3_tokenizer(
            prompt_text,
            return_tensors=None,
            padding= False,
            truncation=True,
            add_special_tokens=False
        ).to(image[0].device)
        prompt_lens = [len(x) for x in prompt_tokens["input_ids"]]
        # prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)  # (B, S_p, D)
        # atts_prompt = prompt_tokens.attention_mask 
        input_ids_list = []
        labels_list = []
        for p_ids, l_ids in zip(prompt_tokens["input_ids"], label_tokens["input_ids"]):
            ids = p_ids + l_ids
            labs = [-100] * len(p_ids) + l_ids
            input_ids_list.append(ids)
            labels_list.append(labs)
        # to_regress_tokens = self.qwen3_tokenizer(
        #     text,
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     add_special_tokens=False
        # ).to(image[0].device)
        # full_lens = [len(x) for x in to_regress_tokens["input_ids"]]

        # pad_id = self.qwen3_tokenizer.pad_token_id
        # full_text = to_regress_tokens.input_ids.masked_fill(
        #     to_regress_tokens.input_ids == pad_id, -100
        # )
        # for i, Lp in enumerate(prompt_lens):
        #     full_text[i, :Lp] = -100
        attn_list = [[1] * len(ids) for ids in input_ids_list]
        padded = self.qwen3_tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attn_list},
            padding="longest",
            return_tensors="pt",
        ).to(image[0].device)

        input_ids = padded["input_ids"]          # (B, S_txt)
        atts_txt  = padded["attention_mask"]     # (B, S_txt)
        B, S_txt = input_ids.shape

    
        labels_txt = torch.full((B, S_txt), -100, dtype=torch.long, device=image[0].device)
        for i, labs in enumerate(labels_list):
            n = min(len(labs), S_txt)
            labels_txt[i, :n] = torch.tensor(labs[:n], dtype=torch.long, device=image[0].device)
        to_regress_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_img, atts_txt], dim=1)

        prefix_len = img_embeds.size(1)

        labels = torch.full(
            (inputs_embeds.size(0), inputs_embeds.size(1)),
            -100,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        labels[:, prefix_len:] = labels_txt

        outputs = self.qwen3_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )
        loss = outputs.loss
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        self.qwen3_tokenizer.padding_side = "left"
        image = samples["pixel_values"]
        to_regress_tokens = self.qwen3_tokenizer(
            samples['findings'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)
        prompt_tokens = self.qwen3_tokenizer(
            samples['prompt'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False
        ).to(image[0].device)
        prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)  # (B, S_p, D)
        atts_prompt = prompt_tokens.attention_mask 

        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        batch_size = img_embeds.shape[0]
        # bos = torch.ones([batch_size, 1],
        #                  dtype=atts_img.dtype,
        #                  device=atts_img.device) * self.qwen3_tokenizer.bos_token_id
        # bos_embeds = self.embed_tokens(bos)
        # atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([img_embeds, prompt_embeds], dim=1)
        attention_mask = torch.cat([atts_img, atts_prompt], dim=1)

        outputs = self.qwen3_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            pad_token_id=self.qwen3_tokenizer.pad_token_id,
            eos_token_id=self.qwen3_tokenizer.eos_token_id,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = []
        for ids, m in zip(to_regress_tokens["input_ids"], to_regress_tokens["attention_mask"]):
            ids = ids[m.bool()]             
            ref.append(self.decode(ids))
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.qwen3_tokenizer.decode(output_token, skip_special_tokens=True)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            self.val_step_outputs.clear()
            return
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=False, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.qwen3_tokenizer.padding_side = "left"
        image = samples["pixel_values"]
        to_regress_tokens = self.qwen3_tokenizer(
            samples['findings'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)
        prompt_tokens = self.qwen3_tokenizer(
            samples['prompt'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False
        ).to(image[0].device)
        prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)  # (B, S_p, D)
        atts_prompt = prompt_tokens.attention_mask 

        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        batch_size = img_embeds.shape[0]
        # bos = torch.ones([batch_size, 1],
        #                  dtype=atts_img.dtype,
        #                  device=atts_img.device) * self.qwen3_tokenizer.bos_token_id
        # bos_embeds = self.embed_tokens(bos)
        # atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([img_embeds, prompt_embeds], dim=1)
        attention_mask = torch.cat([atts_img, atts_prompt], dim=1)

        outputs = self.qwen3_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            pad_token_id=self.qwen3_tokenizer.pad_token_id,
            eos_token_id=self.qwen3_tokenizer.eos_token_id,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = []
        for ids, m in zip(to_regress_tokens["input_ids"], to_regress_tokens["attention_mask"]):
            ids = ids[m.bool()]             
            ref.append(self.decode(ids))
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()