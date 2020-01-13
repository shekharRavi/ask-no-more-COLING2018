import argparse
import torch
import logging
from torchvision.models import vgg16
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

import utils
from dataset import GuessWhatDataset
from models import QGen, Guesser, Oracle, DM1, DM2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_printoptions(profile='full')

def main(args):

    print(args)

    logging.basicConfig(filename='inference_%s_%i.log'%(args.mode, args.max_num_questions),level=logging.INFO)

    splits = (['train'] if args.train_set else list()) + ['valid'] + (['test'] if args.test_set else list())

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = GuessWhatDataset(
            split=split,
            data_dir=args.data_dir,
            model='inference',
            coco_dir=args.coco_dir,
            successful_only=False,
            min_occ=args.min_occ,
            max_sequence_length=args.max_sequence_length,
            h5File='vgg_fc8.hdf5',
            mapping_file='imagefile2id.json')


    qgen = QGen(
        num_embeddings=datasets['valid'].vocab_size,
        embedding_dim=args.qgen_embedding_dim,
        hidden_size=args.qgen_hidden_size,
        visual_embedding_dim=args.qgen_visual_embedding_dim,
        padding_idx=datasets['valid'].pad
        )
    qgen.to(device)
    qgen.load_state_dict(torch.load('bin/qgenX.pt', map_location=lambda storage, loc: storage))

    # vgg = vgg16(pretrained=True)
    # vgg.eval()
    # vgg.to(device)

    guesser = Guesser(
        num_word_embeddings=datasets['valid'].vocab_size,
        word_embedding_dim=args.guesser_word_embedding_dim,
        hidden_size=args.guesser_hidden_size,
        num_cat_embeddings=datasets['valid'].num_categories,
        cat_embedding_dim=args.guesser_cat_embedding_dim,
        mlp_hidden=args.guesser_mlp_hidden
        )
    guesser.to(device)
    guesser.load_state_dict(torch.load('bin/guesser.pt', map_location=lambda storage, loc: storage))


    oracle = Oracle(
        num_word_embeddings=datasets['valid'].vocab_size,
        word_embedding_dim=args.oracle_word_embedding_dim,
        hidden_size=args.oracle_hidden_size,
        num_cat_embeddings=datasets['valid'].num_categories,
        cat_embedding_dim=args.oracle_cat_embedding_dim,
        mlp_hidden=args.oracle_mlp_hidden
        )
    oracle.to(device)
    oracle.load_state_dict(torch.load('bin/oracle.pt', map_location=lambda storage, loc: storage))


    if args.mode == 'dm1':
        dm = DM1(
            rnn_hidden_size=args.qgen_hidden_size,
            attention_hidden=512
            )
        dm.load_state_dict(torch.load('bin/dm1_nomask128.pt', map_location=lambda storage, loc: storage))

    elif args.mode == 'dm2':
        dm = DM2()
        dm.load_state_dict(torch.load('bin/dm2.pt', map_location=lambda storage, loc: storage))

    if args.mode != 'baseline':
        dm.to(device)
        dm.eval()


    torch.no_grad()

    logs = defaultdict(lambda: defaultdict(float))

    for split in splits:

        data_loader = DataLoader(
            dataset=datasets[split],
            batch_size=args.batch_size,
            num_workers=args.num_workers
            )

        for iteration, sample in enumerate(data_loader):

            for k, v in sample.items():
                if torch.is_tensor(v):
                    sample[k] = v.to(device)

            dialogue = sample['input'].clone()
            dialogue_lengths = sample['input'].new_zeros(sample['input'].size(0))

            fc8 = sample['image']

            # get first question
            questions, questions_lengths, h, c, hidden_states = qgen.inference(
                sample['input'],
                fc8=fc8,
                end_of_question_token=datasets['valid'].w2i['<eoq>'],
                hidden=None,
                strategy=args.strategy
            )

            B               = sample['input'].size(0)
            idx             = sample['input'].new_tensor(list(range(B))).long()
            running_idx     = sample['input'].new_tensor(list(range(B))).long()
            mask_current    = sample['input'].new_ones((B)).byte()

            target_category = sample['target_category']
            target_spatial  = sample['target_spatial']
            categories      = sample['categories']
            bboxes          = sample['bboxes']

            answer_logits   = sample['input'].new_empty((B, 3)).float()

            if args.mode != 'baseline':
                num_questions_asked = sample['input'].new_ones(B)

            for qi in range(1, args.max_num_questions+1):

                # add question to dialogue
                dialogue = append_to_padded_sequence(
                    padded_sequence=dialogue,
                    sequence_lengths=dialogue_lengths,
                    appendix=questions,
                    appendix_lengths=questions_lengths,
                    mask_current=mask_current
                    )

                dialogue_lengths[running_idx] += questions_lengths

                # get answers
                answer_logits = oracle.forward(
                    question=questions,
                    length=questions_lengths,
                    target_category=target_category[running_idx],
                    target_spatial=target_spatial[running_idx]
                    )
                answers = answer_logits.topk(1)[1].long()
                answers = answer_class_to_token(answers, datasets['valid'].w2i)

                # add answers to dialogue
                dialogue = append_to_padded_sequence(
                    padded_sequence=dialogue,
                    sequence_lengths=dialogue_lengths,
                    appendix=answers,
                    appendix_lengths=answers.new_ones(answers.size(0)),
                    mask_current=mask_current
                    )
                dialogue_lengths[running_idx] += 1


                # ask next question
                questions, questions_lengths, h[:, running_idx], c[:, running_idx], next_hidden_states = qgen.inference(
                    input=answers,
                    fc8=fc8[running_idx],
                    end_of_question_token=datasets['valid'].w2i['<eoq>'],
                    hidden=(h[:, running_idx], c[:, running_idx]),
                    strategy=args.strategy
                )


                if args.mode == 'dm1':

                    # add hidden state from answer
                    hidden_states_incl_answer = append_to_padded_sequence(
                        padded_sequence=hidden_states,
                        sequence_lengths=dialogue_lengths,
                        appendix=next_hidden_states,
                        appendix_lengths=dialogue_lengths.new_ones((B)),
                        mask_current=mask_current
                    )

                    # update hidden states to include all hidden states of next question
                    hidden_states = append_to_padded_sequence(
                        padded_sequence=hidden_states,
                        sequence_lengths=dialogue_lengths,
                        appendix=next_hidden_states,
                        appendix_lengths=questions_lengths,
                        mask_current=mask_current
                        )

                    dm_logits = dm(
                        hidden_states=hidden_states_incl_answer[running_idx],
                        lengths=dialogue_lengths[running_idx]+1,
                        fc8=fc8[running_idx],
                        masking=False
                    )


                elif args.mode == 'dm2':
                    object_logits, guesser_hidden_states = guesser(
                        sequence=dialogue[running_idx],
                        sequence_length=dialogue_lengths[running_idx],
                        objects=categories[running_idx],
                        spatial=bboxes[running_idx],
                        return_hidden=True
                    )

                    dm_logits = dm(
                        hidden_states=guesser_hidden_states,
                        fc8=fc8[running_idx],
                    )

                if args.mode != 'baseline':
                    decision_logits, decisions = torch.max(dm_logits, 1)

                    # update running idx
                    mask_previous = mask_current.clone()
                    mask_current[running_idx] = (decisions != 1)
                    if mask_current.sum() > 0 and qi < args.max_num_questions:
                        running_idx = idx.masked_select(mask_current)
                        num_questions_asked[running_idx] += 1

                        # remove stopped questions
                        _, qS = questions.size()
                        running_questions = questions.new_zeros(B, qS)
                        running_questions.masked_scatter_(mask_previous.unsqueeze(1).repeat(1, qS), questions)
                        running_questions = running_questions[mask_current]
                        questions = running_questions

                        running_questions_lengths = questions_lengths.new_zeros(B)
                        running_questions_lengths.masked_scatter_(mask_previous, questions_lengths)
                        running_questions_lengths = running_questions_lengths[mask_current]
                        questions_lengths = running_questions_lengths

                    else:
                        break

            object_logits = guesser(
                sequence=dialogue,
                sequence_length=dialogue_lengths,
                objects=sample['categories'],
                spatial=sample['bboxes']
            )

            acc = utils.accuracy(predictions=object_logits, targets=sample['target'])
            logs[split]['running_acc'] += 1/(iteration+1) * (acc - logs[split]['running_acc'])

            if args.mode != 'baseline':
                logs[split]['avg_num_questions'] += 1/(iteration+1) * (torch.mean(num_questions_asked.float()).item() - logs[split]['avg_num_questions'])

            # bookkeeping
            if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                if args.mode != 'baseline':
                    s = "Running-Mean-No-Questions %.3f"%(logs[split]['avg_num_questions'])
                else:
                    s = ""
                print("%s Batch %04d/%04d Batch-Acc %.3f Running-Mean-Acc %.3f %s"
                %(split.upper(), iteration, len(data_loader)-1, acc, logs[split]['running_acc'], s))



        logging.info("++++++++++%s++++++++++"%split.upper())
        logging.info("Set Accuracy %.5f (MQ=%i)"%(logs[split]['running_acc'] * 100, args.max_num_questions))
        if args.mode != 'baseline':
            logging.info("Avg. Number of Questions %.3f"%(logs[split]['avg_num_questions']))
        logging.info("++++++++++%s++++++++++"%('+'*len(split)))

def append_to_padded_sequence(padded_sequence, sequence_lengths, appendix, appendix_lengths, mask_current):

    assert mask_current.sum().item() == appendix.size(0)

    sequence = list()
    lengths = list()

    # get the max length of the new sequences
    appendix_lengths_padded = appendix_lengths.new_zeros(padded_sequence.size(0))
    appendix_lengths_padded.masked_scatter_(mask_current, appendix_lengths)
    appended_sequences_lengths = sequence_lengths + appendix_lengths_padded
    max_length = torch.max( appended_sequences_lengths )

    mi = 0
    for si in range(padded_sequence.size(0)):

        # if dialogue is still running, add item from appendix
        if mask_current[si] == 1:
            # remove padding from padded_sequence; remove padding from appendix; concate both
            sequence.append( torch.cat( (padded_sequence[si, :sequence_lengths[si]], appendix[mi, :appendix_lengths[mi]]), dim=0) )
            mi += 1
        else:
            sequence.append(padded_sequence[si, :sequence_lengths[si]])

        lengths.append(len(sequence[-1]))

        # pad new sequence up to max_length
        pad = sequence[-1].new_zeros( ( (max_length-lengths[-1]), *list(sequence[-1].size()[1:])) )
        sequence[-1] = torch.cat( (sequence[-1], pad) )


    sequence = torch.stack(sequence)
    return sequence

def answer_class_to_token(answers, w2i):

    yes_mask = answers == 0
    no_mask = answers == 1
    na_mask = answers == 2

    answers.masked_fill_(yes_mask, w2i['<yes>'])
    answers.masked_fill_(no_mask, w2i['<no>'])
    answers.masked_fill_(na_mask, w2i['<n/a>'])

    return answers


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Dataset Settings
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-cd', '--coco_dir', type=str, default='/Users/timbaumgartner/MSCOCO')
    parser.add_argument('-mo', '--min_occ', type=int, default=3)
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=100)

    # Experiment Settings
    parser.add_argument('-m', '--mode', type=str, choices=['baseline', 'dm1', 'dm2'], default='baseline')
    parser.add_argument('-mq', '--max_num_questions', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-nw', '--num_workers', type=int, default=2)
    parser.add_argument('-v', '--print_every', type=int, default=100)
    parser.add_argument('-train', '--train_set', action='store_true')
    parser.add_argument('-test', '--test_set', action='store_true')


    # Hyperparameter
    parser.add_argument('--qgen_embedding_dim', type=int, default=512)
    parser.add_argument('--qgen_hidden_size', type=int, default=1024)
    parser.add_argument('--qgen_visual_embedding_dim', type=int, default=512)
    parser.add_argument('--strategy', type=str, choices=['greedy', 'sampling'], default='greedy')

    parser.add_argument('--guesser_word_embedding_dim', type=int, default=512)
    parser.add_argument('--guesser_hidden_size', type=int, default=512)
    parser.add_argument('--guesser_cat_embedding_dim', type=int, default=256)
    parser.add_argument('--guesser_mlp_hidden', type=int, default=512)

    parser.add_argument('--oracle_word_embedding_dim', type=int, default=300)
    parser.add_argument('--oracle_hidden_size', type=int, default=512)
    parser.add_argument('--oracle_cat_embedding_dim', type=int, default=512)
    parser.add_argument('--oracle_mlp_hidden', type=int, default=128)

    parser.add_argument('--dm_mlp_hidden', type=int, default=512)

    args = parser.parse_args()
    args.num_workers = min(cpu_count(), args.num_workers)

    main(args)
