import os
import re
import time

import torch

import sys
sys.path.append(os.getenv('FIREREDASR_PATH'))

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper


class FireRedAsrConfig:
    def __init__(
        self,
        use_gpu=True,
        beam_size=3,
        nbest=1,
        decode_max_len=0,
        softmax_smoothing=1.25,
        aed_length_penalty=0.6,
        eos_penalty=1.0,
        return_timestamp=1,
        decode_min_len=0,
        repetition_penalty=1.0,
        llm_length_penalty=0.0,
        temperature=1.0,
        use_half=False
    ):
        self.use_gpu = use_gpu
        self.beam_size = beam_size
        self.nbest = nbest
        self.decode_max_len = decode_max_len
        self.softmax_smoothing = softmax_smoothing
        self.aed_length_penalty = aed_length_penalty
        self.eos_penalty = eos_penalty
        self.return_timestamp = return_timestamp
        self.decode_min_len = decode_min_len
        self.repetition_penalty = repetition_penalty
        self.llm_length_penalty = llm_length_penalty
        self.temperature = temperature
        self.use_half = use_half


class FireRedAsr:
    @classmethod
    def from_pretrained(cls, asr_type, model_dir, config):
        assert asr_type in ["aed", "llm"]

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path = os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir
            )
        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer, config)

    def __init__(self, asr_type, feat_extractor, model, tokenizer, config):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        if self.config.use_gpu:
            if self.config.use_half:
                self.model.half()
            self.model.cuda()
        else:
            self.model.cpu()

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path):
        feats, lengths, durs = self.feat_extractor(batch_wav_path)
        total_dur = sum(durs)
        if self.config.use_gpu:
            feats, lengths = feats.cuda(), lengths.cuda()
            if self.config.use_half:
                feats = feats.half()

        if self.asr_type == "aed":
            start_time = time.time()

            try:
                hyps = self.model.transcribe(
                    feats,
                    lengths,
                    self.config.beam_size,
                    self.config.nbest,
                    self.config.decode_max_len,
                    self.config.softmax_smoothing,
                    self.config.aed_length_penalty,
                    self.config.eos_penalty,
                )
            except Exception as e:
                hyps = []

            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0

            results = []
            for uttid, wav, hyp in zip(batch_uttid, batch_wav_path, hyps):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                results.append(
                    {
                        "uttid": uttid,
                        "text": text.lower(),
                        "confidence": round(hyp["confidence"].cpu().item(), 3),
                        "rtf": f"{rtf:.4f}",
                    }
                )
                results[-1]["lang"] = get_lang_by_text(text)
                if type(wav) == str:
                    results[-1]["wav"] = wav

            return results

        elif self.asr_type == "llm":
            input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
                origin_texts=[""] * feats.size(0),
                tokenizer=self.tokenizer,
                max_len=128,
                decode=True,
            )
            if self.config.use_gpu:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            start_time = time.time()

            try:
                generated_ids = self.model.transcribe(
                    feats,
                    lengths,
                    input_ids,
                    attention_mask,
                    self.config.beam_size,
                    self.config.decode_max_len,
                    self.config.decode_min_len,
                    self.config.repetition_penalty,
                    self.config.llm_length_penalty,
                    self.config.temperature,
                )

                texts = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            except Exception as e:
                texts = []
            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0
            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                results.append(
                    {"uttid": uttid, "text": text, "wav": wav, "rtf": f"{rtf:.4f}"}
                )
            return results


def load_fireredasr_aed_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    print("model args:", package["args"])
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer


def get_lang_by_text(txt_ori):
    lang = "zh"
    txt = txt_ori.replace(" ", "").strip().lower()
    if len(txt) == 0:
        return lang

    n = 0
    for c in txt:
        if re.match("[a-z]", c.lower()):
            n += 1

    ratio = 100.0 * n / len(txt)
    if ratio >= 80:
        lang = "en"

    return lang
