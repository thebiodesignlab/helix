from .utils import MutationDataset, sample_data, split_train, compute_score, BT_loss, KLloss, evaluate
from .common import stub, image, RESULTS_PATH, CHECKPOINT_PATH
import os
import yaml
import torch
import numpy as np
import pandas as pd
import modal
from torch.utils.data import DataLoader
from transformers import EsmForMaskedLM, EsmTokenizer
from accelerate import Accelerator

N_GPUS = int(os.environ.get("N_GPUS", 2))
GPU_CONFIG = os.environ.get("GPU_CONFIG", modal.gpu.H100(count=N_GPUS))


@stub.cls(
    gpu=GPU_CONFIG,
    image=image,
    # 12hrs
    timeout=12 * 60 * 60,
    # For occasional connection error to 'cdn-lfs.huggingface.co'
    retries=1,
    # volumes=VOLUME_CONFIG,
)
class ConFitTrainer:
    def __init__(self, raw_config: str):
        self.raw_config = raw_config

    @modal.enter()
    def init(self):
        self.config = self.load_config(self.raw_config)
        self.accelerator = Accelerator()
        self.model, self.model_reg, self.tokenizer, basemodel = self.create_model()
        self.optimizer, self.scheduler = self.create_optimizer_scheduler()
        self.trainset, self.testset, self.valset = self.create_datasets()
        self.trainloader, self.testloader, self.valloader = self.create_dataloaders()

    def load_config(self, raw_config):
        config = yaml.safe_load(raw_config)
        # Ensure all values are of the correct type
        typed_config = {k: self._convert_type(v) for k, v in config.items()}
        return typed_config

    def _convert_type(self, value):
        if isinstance(value, str):
            # Attempt to interpret strings as integers, floats, or booleans
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except ValueError:
                if value.lower() in ['true', 'false']:
                    return value.lower() == 'true'
        return value

    def create_model(self):
        from peft import LoraConfig, get_peft_model
        model_name = self.config['model']
        if model_name == 'ESM-1v':
            basemodel = EsmForMaskedLM.from_pretrained(
                f'facebook/esm1v_t33_650M_UR90S_{self.config["model_seed"]}')
            model_reg = EsmForMaskedLM.from_pretrained(
                f'facebook/esm1v_t33_650M_UR90S_{self.config["model_seed"]}')
            tokenizer = EsmTokenizer.from_pretrained(
                f'facebook/esm1v_t33_650M_UR90S_{self.config["model_seed"]}')
        elif model_name == 'ESM-2':
            basemodel = EsmForMaskedLM.from_pretrained(
                'facebook/esm2_t48_15B_UR50D')
            model_reg = EsmForMaskedLM.from_pretrained(
                'facebook/esm2_t48_15B_UR50D')
            tokenizer = EsmTokenizer.from_pretrained(
                'facebook/esm2_t48_15B_UR50D')
        elif model_name == 'ESM-1b':
            basemodel = EsmForMaskedLM.from_pretrained(
                'facebook/esm1b_t33_650M_UR50S')
            model_reg = EsmForMaskedLM.from_pretrained(
                'facebook/esm1b_t33_650M_UR50S')
            tokenizer = EsmTokenizer.from_pretrained(
                'facebook/esm1b_t33_650M_UR50S')
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        for param in model_reg.parameters():
            param.requires_grad = False
        model_reg.eval()

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            target_modules=["query", "value"]
        )
        model = get_peft_model(basemodel, peft_config)

        return model, model_reg, tokenizer, basemodel

    def create_optimizer_scheduler(self):
        from peft.utils.other import fsdp_auto_wrap_policy
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config['ini_lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2*self.config['max_epochs'], eta_min=self.config['min_lr'])
        if os.environ.get("ACCELERATE_USE_FSDP", None) is not None:
            self.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(
                self.model)
        self.model, optimizer, scheduler = self.accelerator.prepare(
            self.model, optimizer, scheduler)
        self.model_reg = self.accelerator.prepare(self.model_reg)
        return optimizer, scheduler

    def create_datasets(self):
        if self.accelerator.is_main_process:
            sample_data(
                self.config['dataset'], self.config['sample_seed'], self.config['shot'])
            split_train(self.config['dataset'])

        with self.accelerator.main_process_first():
            train_csv = pd.DataFrame(None)
            test_csv = pd.read_csv(
                f'/confit/data/{self.config["dataset"]}/test.csv')
            for i in range(1, 6):
                if i == self.config['model_seed']:
                    val_csv = pd.read_csv(
                        f'/confit/data/{self.config["dataset"]}/train_{i}.csv')
                temp_csv = pd.read_csv(
                    f'/confit/data/{self.config["dataset"]}/train_{i}.csv')
                train_csv = pd.concat([train_csv, temp_csv], axis=0)

        trainset = MutationDataset(
            data=train_csv, fname=self.config['dataset'], tokenizer=self.tokenizer)
        testset = MutationDataset(
            data=test_csv, fname=self.config['dataset'], tokenizer=self.tokenizer)
        valset = MutationDataset(
            data=val_csv, fname=self.config['dataset'], tokenizer=self.tokenizer)
        return trainset, testset, valset

    def create_dataloaders(self):
        with self.accelerator.main_process_first():
            trainloader = DataLoader(
                self.trainset, batch_size=self.config['batch_size'], collate_fn=self.trainset.collate_fn, shuffle=True)
            testloader = DataLoader(
                self.testset, batch_size=2, collate_fn=self.testset.collate_fn)
            valloader = DataLoader(
                self.valset, batch_size=2, collate_fn=self.testset.collate_fn)

        trainloader = self.accelerator.prepare(trainloader)
        testloader = self.accelerator.prepare(testloader)
        valloader = self.accelerator.prepare(valloader)
        return trainloader, testloader, valloader

    @modal.method()
    def train(self):
        best_sr = -np.inf
        endure = 0

        for epoch in range(self.config['max_epochs']):
            loss = self.train_step(self.model, self.model_reg, self.trainloader,
                                   self.optimizer, self.tokenizer, self.config['lambda_reg'])
            self.accelerator.print(
                f'========epoch{epoch}; training loss :{loss}=================')
            sr = evaluate(self.model, self.valloader,
                          self.tokenizer, self.accelerator)
            self.accelerator.print(
                f'========epoch{epoch}; val spearman correlation :{sr}=================')
            self.scheduler.step()
            if best_sr > sr:
                endure += 1
            else:
                endure = 0
                best_sr = sr
                self.save_checkpoint(epoch)
            if sr == 1.0:
                self.accelerator.print(
                    f'========early stop at epoch{epoch}!============')
                break
            if endure > self.config['endure_time']:
                self.accelerator.print(
                    f'========early stop at epoch{epoch}!============')
                break

        self.accelerator.print(
            '=======training done!, test the performance!========')
        self.evaluate_and_save_test_results()

    def train_step(self, model, model_reg, trainloader, optimizer, tokenizer, lambda_reg):
        model.train()
        total_loss = 0.
        for step, data in enumerate(trainloader):
            seq, mask, wt, wt_mask, pos, golden_score, _ = data

            score, logits = compute_score(model, seq, mask, wt, pos, tokenizer)
            score = score.cuda()
            l_BT = BT_loss(score, golden_score)
            out_reg = model_reg(wt, wt_mask)
            logits_reg = out_reg.logits
            l_reg = KLloss(logits, logits_reg, seq, mask)
            loss = l_BT + lambda_reg*l_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss

    def save_checkpoint(self, epoch):
        if not os.path.isdir(f'{CHECKPOINT_PATH}/{self.config["dataset"]}'):
            if self.accelerator.is_main_process:
                os.makedirs(f'{CHECKPOINT_PATH}/{self.config["dataset"]}')
        save_path = os.path.join(
            CHECKPOINT_PATH, f'{self.config["dataset"]}', f'seed{self.config["model_seed"]}')
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_path)

    def evaluate_and_save_test_results(self):
        from peft import PeftModel
        save_path = os.path.join(
            CHECKPOINT_PATH, f'{self.config["dataset"]}', f'seed{self.config["model_seed"]}')
        del self.model
        self.accelerator.free_memory()
        self.model = PeftModel.from_pretrained(
            self.create_model()[3], save_path)
        self.model = self.accelerator.prepare(self.model)
        sr, score, pid = evaluate(
            self.model, self.testloader, self.tokenizer, self.accelerator, istest=True)
        pred_csv = pd.DataFrame(
            {f'{self.config["model_seed"]}': score, 'PID': pid})
        if self.accelerator.is_main_process:
            if not os.path.isdir(f'{RESULTS_PATH}/{self.config["dataset"]}'):
                os.makedirs(f'{RESULTS_PATH}/{self.config["dataset"]}')
            if os.path.exists(f'{RESULTS_PATH}/{self.config["dataset"]}/pred.csv'):
                pred = pd.read_csv(
                    f'{RESULTS_PATH}/{self.config["dataset"]}/pred.csv', index_col=0)
                pred = pd.merge(pred, pred_csv, on='PID')
            else:
                pred = pred_csv
            pred.to_csv(f'{RESULTS_PATH}/{self.config["dataset"]}/pred.csv')
        self.accelerator.print(
            f'=============the test spearman correlation for early stop: {sr}==================')


@stub.local_entrypoint()
def main(
    config: str,
):
    # Read config and data source files and pass their contents to the remote function.
    with open(config, "r") as cfg:
        ConFitTrainer(cfg.read()).train.remote()
