import math
import torch.nn.functional as F
import collections
import torch
import numpy
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
import numpy as np
from gramformer import Gramformer
from fairseq.sequence_generator import SequenceGenerator
from fairseq import utils, bleu
from . import FairseqCriterion, register_criterion
import sys 
MLM_SCORER_DIR=''
sys.path.insert(1,MLM_SCORER_DIR) #OK

from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
from evaluate import load


gf = Gramformer(models = 1, use_gpu=True) # 1=corrector, 2=detector

# import logging
device = "cuda" if torch.cuda.is_available() else "cpu"

bertscore = load("bertscore")




ctxs = [mx.gpu(0)] # or, e.g., [mx.gpu(0), mx.gpu(1)]

mlm_model, mlm_vocab, mlm_tokenizer = get_pretrained(ctxs, 'bert-base-multilingual-uncased')
mlm_model = mlm_model.to(device)
# mlm_tokenizer = mlm_tokenizer.to(device)
mlm_scorer = MLMScorerPT(mlm_model, mlm_vocab, mlm_tokenizer, ctxs)
from fairseq.sequence_generator import SequenceGenerator
from fairseq import utils, bleu

@dataclass
class V2CriterionConfig(FairseqDataclass):
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )

@register_criterion('mrt_mlm',dataclass=V2CriterionConfig)
class MrtMLM(FairseqCriterion):

    def __init__(self, task):
        super().__init__( task)
        self.task = task


    # def forward(self, model, sample, reduce=True):
    #     """Compute the loss for the given sample.
    #
    #     Returns a tuple with three elements:
    #     1) the loss
    #     2) the sample size, which is used as the denominator for the gradient
    #     3) logging outputs to display while training
    #     """
    #     net_output = model(**sample['net_input'])
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     lprobs = lprobs.view(-1, lprobs.size(-1))
    #     target = model.get_targets(sample, net_output).view(-1)
    #     loss = F.nll_loss(lprobs, target, size_average=False,
    #                       ignore_index=self.padding_idx,
    #                       reduce=reduce)
    #     sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
    #     logging_output = {
    #         'loss': utils.item(loss.data) if reduce else loss.data,
    #         'ntokens': sample['ntokens'],
    #         'sample_size': sample_size,
    #     }
    #     return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        # sample mode
        #print('!!!RL loss.')
        model.eval()
        src_dict = self.task.source_dictionary
        tgt_dict = self.task.target_dictionary
        eos_idx = self.task.target_dictionary.eos()
        sample_beam = 5
        translator = SequenceGenerator([model], tgt_dict=tgt_dict,beam_size=sample_beam)
        translator = translator.cuda()
        ct = 0
        translations = []

        s = utils.move_to_cuda(sample)
        input = s
        max_len = 200

        # Generate samples and store them
        with torch.no_grad():
            hypos = translator.generate(model,input)
        for i, id in enumerate(s['id'].data):
            src = input['net_input']['src_tokens'].data[i, :]
            # remove padding from ref
            ref = utils.strip_pad(s['target'].data[i, :], tgt_dict.pad()) if s['target'] is not None else None
            translations.append((id, src, ref, hypos[i]))
            ct += 1

        model.train()

        num_sents_in_batch = len(translations)
        total_idx = -1

        mle_tokens = sample['ntokens']

        batch_rl_loss = 0
        batch_tokens = 0
        sample_ind = 0
        rewards = torch.Tensor(num_sents_in_batch*sample_beam).float().cuda()
        mlm_scores = torch.Tensor(num_sents_in_batch*sample_beam).float().cuda()
        logprobs = torch.Tensor(num_sents_in_batch*sample_beam).float().cuda()
        final_loss = torch.Tensor(num_sents_in_batch).float().cuda()
        hypo_strs_bertscore = []
        src_strs = []
        src_strs_2 = []

        for sample_id, src_tokens, tgt_tokens, hypos in translations:

            # calculate bleu
            sample_ind += 1

            src_str = src_dict.string(src_tokens,escape_unk=True, bpe_symbol="@@ ",extra_symbols_to_ignore=["<pad>"])
            src_str = src_str.strip()

            for _ in range(sample_beam):
                src_strs_2.append(src_str) #Orig
            # For generation, question mark is not appended, but for scoring, we append a question mark
            if src_str[-1] != "?":
                src_str = src_str + "?"
            

            corrects = []
            corrected_sentences = gf.correct(src_str, max_candidates=1)
            for corrected_sentence in corrected_sentences:

                corrects.append(corrected_sentence)
            src_str_gf = "".join(corrects)

            for _ in range(sample_beam):
                src_strs.append(src_str_gf)

            for i in range(sample_beam):
                total_idx += 1
                hypo = hypos[i]
                trans_tokens = hypo['tokens']

                hypo_str = tgt_dict.string(trans_tokens,escape_unk=True, bpe_symbol="@@ ",extra_symbols_to_ignore=["<pad>"])
                gen_str = hypo_str
                hypo_str = hypo_str.replace("@@ ","")
                hypo_str = hypo_str.replace("@@","")
                hypo_str = hypo_str.replace("@","")
                hypo_str = hypo_str.replace("<<unk>> ","")
                # Trim the generated output before passing to the MLM scorer to avoid passing very long sentences, 1000 is arbitrarily chosen
                hypo_str = hypo_str[:1000]

                hypo_str = hypo_str.strip()
                print(hypo_str)
                if hypo_str[-1] == "।":
                    hypo_str = hypo_str[:-1] + "?"
                if hypo_str[-1] != "?":
                    hypo_str = hypo_str +"?"

                hypo_strs_bertscore.append(hypo_str)

                try:
                    mlm_scores[total_idx] = -1* mlm_scorer.score_sentences([hypo_str])[0]
                except:
                    # When model predicts way too long sequence to be scored by MLM scorer. 20 is arbitrarily taken as a bad score for the translation as average score is lower than 20 
                    print("Error occured when calculating MLM Score. Assigning 20 as the reward")
                    mlm_scores[total_idx] = 20


                # one_sample loss calculation
                tgt_input_tokens = trans_tokens.new(trans_tokens.shape).fill_(0)
                assert trans_tokens[-1] == eos_idx
                tgt_input_tokens[0] = eos_idx
                tgt_input_tokens[1:] = trans_tokens[:-1]
                train_sample = {
                    'net_input': {
                        'src_tokens': src_tokens.view(1, -1),
                        'src_lengths': torch.LongTensor(src_tokens.numel()).view(1, -1),
                        'prev_output_tokens': tgt_input_tokens.view(1, -1),
                    },
                    'target': trans_tokens.view(1, -1)
                }
                train_sample = utils.move_to_cuda(train_sample)
                net_output = model(**train_sample['net_input'])
                lprobs = model.get_normalized_probs(net_output, log_probs=True)
                lprobs = lprobs.view(-1, lprobs.size(-1))
                target = model.get_targets(train_sample, net_output).view(-1, 1)
                non_pad_mask = target.ne(tgt_dict.pad())
                lprob = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
                logprobs[total_idx] = torch.sum(lprob)
                ntokens = len(train_sample['net_input']['src_tokens'])
                # print("ntokens:",ntokens)
                batch_tokens += ntokens
        rewards_1 = bertscore.compute(predictions=hypo_strs_bertscore, references=src_strs, model_type='facebook/mbart-large-50-one-to-many-mmt',device="cuda")['f1']
        rewards_3 = bertscore.compute(predictions=hypo_strs_bertscore, references=src_strs_2, model_type='facebook/mbart-large-50-one-to-many-mmt',device="cuda")['f1']
        rewards_ = np.maximum.reduce([rewards_1,rewards_3])

        for idx in range(len(rewards_)):
            rewards[idx] = 1 - rewards_[idx]
       
        rewards =  0.85 * rewards  + 0.15 * mlm_scores

        idx = 0
        for i in range(0,num_sents_in_batch*sample_beam,sample_beam):
            print(i)
            final_loss[idx] = torch.sum((logprobs[i:i+sample_beam] * rewards[i:i+sample_beam])) 
            idx += 1

        print(final_loss)
        rl_loss = torch.sum(final_loss) / num_sents_in_batch  # one sample loss            
        batch_rl_loss += rl_loss


        print("Batch size: ", num_sents_in_batch)
        avg_rl_loss = batch_rl_loss 
        print('avg_rl_loss:', avg_rl_loss)
        total_loss = avg_rl_loss
        total_tokens = mle_tokens

        logging_output = {
            'loss': utils.item(total_loss.data),
            'ntokens': total_tokens,
            'sample_size': total_tokens,
        }
        print('total: ',total_loss)
        return total_loss, total_tokens, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output


    