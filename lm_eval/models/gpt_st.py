import os
import json

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import AutoTokenizer
from lm_eval.base import BaseLM

from .sparse_transformer.model import GPT

class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        model_path="/dccstor/codeai/yikang/SparseGPT/checkpoints/all_the_pile_vt-350m_BSZ6_BLOCKSZ512_ATTstickbreaking_HLENGTH512_SAMPLETOPK0_MOEPDROP0_GATING256_TOPK1_AUXmi0.01",
        batch_size=1,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(model_path, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        checkpoint_config = json.load(open(model_path + "/config.json", 'r'))

        # load tokenizer and dataset
        print("Loading vocab...")
        # vocab_path = os.path.join('/dccstor/codeai/yikang/datasets/vocabularies', checkpoint_config['dataset']['tokenizer'])
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_config['dataset']['tokenizer'])

        model_config = GPT.get_default_config()
        model_config.merge_from_dict(checkpoint_config['model'])
        model_config.vocab_size = len(self.tokenizer)
        self.model = GPT(model_config).to(self.device)
        self.model.eval()

        # checkpoint_path = os.path.join(model_path, ('checkpoint/rank_%d.pt' % 0))
        checkpoint_path = os.path.join(model_path, 'checkpoint/latest.pt')
        checkpoint = torch.load(checkpoint_path)['model_state_dict']
        new_checkpoint = {}
        for k, v in checkpoint.items():
            if 'module' in k:
                new_checkpoint[k.replace('module.', '')] = v
            else:
                new_checkpoint[k] = v
        self.model.load_state_dict(new_checkpoint)

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ):
            assert self.tokenizer.encode("hello\n\nhello") == [
                31373,
                198,
                198,
                31373,
            ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 512 * 24

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        # inps = inps.chunk(inps.shape[1] // 512 + 1, dim=1)
        split_size = [512] * (inps.shape[1] // 512) 
        if inps.shape[1] % 512 > 0:
            split_size += [inps.shape[1] % 512]
        inps = torch.split(inps, split_size, dim=1)
        outputs = []
        hidden = None
        for inp in inps:
            with torch.no_grad():
                logits, _, _, hidden = self.model(inp, hidden=hidden)
                outputs.append(logits)
        return torch.cat(outputs, dim=1)

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_new_tokens=max_length, eos_token_id=eos_token_id, do_sample=False
        )


# for backwards compatibility
GPT2LM = HFLM
