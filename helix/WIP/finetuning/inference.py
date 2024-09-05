import os
import torch
import pandas as pd
import numpy as np
from accelerate import Accelerator
from .common import stub, VOLUME_CONFIG, image
import modal
from .utils import compute_score, spearman
N_INFERENCE_GPU = 1


@stub.cls(
    gpu=modal.gpu.H100(count=N_INFERENCE_GPU),
    image=image,
    volumes=VOLUME_CONFIG,
    allow_concurrent_inputs=30,
    container_idle_timeout=900,
)
class Inference:
    def __init__(self, dataset, checkpoint_dir='/confit/checkpoint', predicted_dir='predicted', model_name='ESM-1b', model_seed=1, no_retrieval=False, retrieval_alpha=0.8):
        from transformers import EsmForMaskedLM, EsmTokenizer
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        self.predicted_dir = predicted_dir
        self.model_name = model_name
        self.model_seed = model_seed
        self.no_retrieval = no_retrieval
        self.retrieval_alpha = retrieval_alpha

        self.accelerator = Accelerator()

        if self.model_name == 'ESM-1v':
            self.basemodel = EsmForMaskedLM.from_pretrained(
                f'facebook/esm1v_t33_650M_UR90S_{self.model_seed}')
            self.tokenizer = EsmTokenizer.from_pretrained(
                f'facebook/esm1v_t33_650M_UR90S_{self.model_seed}')
        elif self.model_name == 'ESM-2':
            self.basemodel = EsmForMaskedLM.from_pretrained(
                'facebook/esm2_t48_15B_UR50D')
            self.tokenizer = EsmTokenizer.from_pretrained(
                'facebook/esm2_t48_15B_UR50D')
        elif self.model_name == 'ESM-1b':
            self.basemodel = EsmForMaskedLM.from_pretrained(
                'facebook/esm1b_t33_650M_UR50S')
            self.tokenizer = EsmTokenizer.from_pretrained(
                'facebook/esm1b_t33_650M_UR50S')

        self.load_model()

    def load_model(self):
        """
        Load the fine-tuned ConFit model from the checkpoint if available, otherwise use the base pre-trained model.
        """
        from peft import PeftModel
        save_path = os.path.join(
            self.checkpoint_dir, self.dataset, f'seed{self.model_seed}')
        if os.path.exists(save_path):
            self.model = PeftModel.from_pretrained(self.basemodel, save_path)
        else:
            self.model = self.basemodel
        self.model = self.accelerator.prepare(self.model)

    @modal.method()
    def evaluate(self):
        from .utils import Mutation_Set
        from torch.utils.data import DataLoader
        self.model.eval()
        score_list = []
        gscore_list = []
        pid_list = []
        test_csv = pd.read_csv(f'/confit/data/{self.dataset}/data.csv')
        testset = Mutation_Set(
            data=test_csv, fname=self.dataset,  tokenizer=self.tokenizer)
        testloader = DataLoader(testset, batch_size=2,
                                collate_fn=testset.collate_fn)
        with torch.no_grad():
            for step, data in enumerate(testloader):
                seq, mask = data[0].cuda(), data[1].cuda()
                wt, wt_mask = data[2].cuda(), data[3].cuda()
                pos = data[4]
                golden_score = data[5].cuda()
                pid = data[6].cuda()

                score, logits = compute_score(
                    self.model, seq, mask, wt, pos, self.tokenizer)

                score = self.accelerator.gather(score)
                golden_score = self.accelerator.gather(golden_score)
                pid = self.accelerator.gather(pid)

                score_list.extend(score.cpu().numpy())
                gscore_list.extend(golden_score.cpu().numpy())
                pid_list.extend(pid.cpu().numpy())

        score_list = np.asarray(score_list)
        gscore_list = np.asarray(gscore_list)
        pid_list = np.asarray(pid_list)

        if not self.no_retrieval:
            elbo = pd.read_csv(
                f'/confit/data/{self.dataset}/vae_elbo.csv', index_col=0)
            perf = pd.DataFrame({'avg': score_list, 'PID': pid_list})
            perf = pd.merge(perf, elbo, on='PID')
            perf['retrieval'] = self.retrieval_alpha * perf['avg'] + \
                (1 - self.retrieval_alpha) * perf['elbo']
            print(perf.columns)
            score = list(perf['retrieval'])
            gscore = list(perf['log_fitness'])
            score = np.asarray(score)
            gscore = np.asarray(gscore)
            sr = spearman(score, gscore)
        else:
            score = score_list
            gscore = gscore_list
            sr = spearman(score, gscore)

        return sr, score_list, pid_list

    @modal.method()
    def save_prediction(self, score_list, pid_list):
        if self.accelerator.is_main_process:
            if not os.path.isdir(os.path.join(self.predicted_dir, self.dataset)):
                os.makedirs(os.path.join(self.predicted_dir, self.dataset))
            pred_csv = pd.DataFrame(
                {f'{self.model_seed}': score_list, 'PID': pid_list})
            if os.path.exists(os.path.join(self.predicted_dir, self.dataset, 'pred.csv')):
                pred = pd.read_csv(os.path.join(
                    self.predicted_dir, self.dataset, 'pred.csv'), index_col=0)
                pred = pd.merge(pred, pred_csv, on='PID')
            else:
                pred = pred_csv
            pred.to_csv(os.path.join(
                self.predicted_dir, self.dataset, 'pred.csv'))

    @modal.method()
    def save_summary(self, sr, shot):
        if self.accelerator.is_main_process:
            if not os.path.isdir(os.path.join('results', self.dataset)):
                os.makedirs(os.path.join('results', self.dataset))
            summary = pd.DataFrame(
                {'spearman': sr, 'shot': shot}, index=[f'{self.dataset}'])
            summary.to_csv(os.path.join(
                'results', self.dataset, 'summary.csv'))


@stub.local_entrypoint()
def inference_main():
    inference = Inference("GB1_Olson2014_ddg", no_retrieval=True)
    sr, score_list, pid_list = inference.evaluate.remote()
    print(sr)
