import torch
import torch.nn as nn


class QGen(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, hidden_size, visual_embedding_dim, visual_dim=1000, padding_idx=0):

        super().__init__()

        self.visual_emb = nn.Linear(visual_dim, visual_embedding_dim)
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim+visual_embedding_dim, hidden_size, batch_first=True)
        self.h2v = nn.Linear(hidden_size, num_embeddings)

    def forward(self, input, length, fc8, max_length=None, return_hidden=False):

        sorted_length, sorted_idx = torch.sort(length, descending=True)
        input = input[sorted_idx]
        fc8 = fc8[sorted_idx]
        visual_emb = self.visual_emb(fc8)
        visual_emb = visual_emb.unsqueeze(1).repeat(1, input.size(1), 1)

        emb = self.emb(input)
        rnn_input = torch.cat((emb, visual_emb), dim=-1)
        packed_emb = nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length, batch_first=True)

        packed_outputs, _ = self.rnn(packed_emb)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=(max_length.item() if max_length is not None else None))

        _, reversed_idx = torch.sort(sorted_idx)

        if return_hidden:
            outputs = outputs[reversed_idx]
            return outputs
        else:
            logits = self.h2v(outputs)
            logits = logits[reversed_idx]
            return logits

    def inference(self, input, fc8, end_of_question_token, hidden=None, strategy='greedy'):

        B               = input.size(0)
        idx             = input.new_tensor(list(range(B))).long()
        running_idx     = input.new_tensor(list(range(B))).long()
        m               = input.new_ones((B)).byte()
        lengths         = input.new_zeros((B)).long()
        if hidden is None:
            h = input.new_zeros((1, B, self.rnn.hidden_size)).float()
            c = input.new_zeros((1, B, self.rnn.hidden_size)).float()
        else:
            h, c = hidden

        fc8 = fc8.unsqueeze(1)

        generations = list()
        hidden_states = list()

        i = 0
        while True and torch.max(lengths[running_idx]).item() < 100:

            # print(input[running_idx])

            # QGen forward pass
            emb = self.emb(input[running_idx]).view(len(running_idx), 1, -1) # B x S x F
            visual_emb = self.visual_emb(fc8[running_idx])
            emb = torch.cat((emb, visual_emb), dim=-1)
            outputs, (h[:, running_idx], c[:, running_idx]) = self.rnn(emb, (h[:, running_idx], c[:, running_idx]))
            logits = self.h2v(outputs)

            # get generated token
            logits = logits.squeeze(1)
            if strategy == 'greedy':
                _, input[running_idx] = logits.topk(1)
            elif strategy == 'sampling':
                probs = nn.functional.softmax(logits, dim=-1)
                input[running_idx] = probs.multinomial(num_samples=1)
            else:
                raise ValueError()

            # save hidden states
            step_hidden_states = h.new_zeros(B, self.rnn.hidden_size)
            step_hidden_states.masked_scatter_(m.unsqueeze(1).repeat(1, self.rnn.hidden_size), h[:, running_idx].squeeze(0))
            hidden_states.append(step_hidden_states)

            # update running idx
            m = (input != end_of_question_token).squeeze(1)
            if m.sum() > 0:
                running_idx = idx.masked_select(m)

                # save generation (excluding eoq tokens)
                generated_idx = input.new_zeros((B))
                generated_idx.masked_scatter_(m, input[running_idx].squeeze(1))
                generations.append(generated_idx)

                # update lengths
                lengths[running_idx] = lengths[running_idx] + 1

            else:
                break

        #print("hidden before stacking \n", hidden_states, len(hidden_states), hidden_states[0].size())
        r = list()
        r.append(torch.stack(generations, dim=1))
        r.append(lengths)
        r.append(h)
        r.append(c)
        r.append(torch.stack(hidden_states, dim=0).transpose(1,0))

        return r

class Guesser(nn.Module):

    def __init__(self, num_word_embeddings, word_embedding_dim, hidden_size, num_cat_embeddings,
                 cat_embedding_dim, mlp_hidden=128, num_spatial=8, word_padding_idx=0, cat_padding_idx=0):

        super().__init__()

        self.word_emb = nn.Embedding(num_word_embeddings, word_embedding_dim, padding_idx=word_padding_idx)
        self.rnn = nn.LSTM(word_embedding_dim, hidden_size, batch_first=True)

        self.cat_emb = nn.Embedding(num_cat_embeddings, cat_embedding_dim, padding_idx=cat_padding_idx)
        self.mlp = nn.Sequential(
            nn.Linear(cat_embedding_dim+num_spatial, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden_size)
            )

    def forward(self, sequence, sequence_length, objects, spatial, return_hidden=False):

        sorted_length, sorted_idx = torch.sort(sequence_length, descending=True)
        sequence = sequence[sorted_idx]

        word_emb = self.word_emb(sequence)
        packed_emb = nn.utils.rnn.pack_padded_sequence(word_emb, sorted_length, batch_first=True)
        _, last_hidden = self.rnn(packed_emb)

        last_hidden = last_hidden[0].squeeze(0).unsqueeze(-1) # B x H x 1

        _, reversed_idx = torch.sort(sorted_idx)
        last_hidden = last_hidden[reversed_idx]

        cat_emb = self.cat_emb(objects)
        mlp_input = torch.cat((cat_emb, spatial), dim=-1)
        obj_emb = self.mlp(mlp_input) # B x 20 x H

        logits = torch.bmm(obj_emb, last_hidden).squeeze(-1) # B x 20 # TODO mask padding objects

        if return_hidden:
            return logits, last_hidden.squeeze(2)
        else:
            return logits

class Oracle(nn.Module):

    def __init__(self, num_word_embeddings, word_embedding_dim, hidden_size, num_cat_embeddings,
        cat_embedding_dim, mlp_hidden, word_padding_idx=0, cat_padding_idx=0, num_spatial=8):

        super().__init__()

        self.word_emb = nn.Embedding(num_word_embeddings, word_embedding_dim, padding_idx=word_padding_idx)
        self.rnn = nn.LSTM(word_embedding_dim, hidden_size, batch_first=True)

        self.cat_emb = nn.Embedding(num_cat_embeddings, cat_embedding_dim, padding_idx=cat_padding_idx)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size+cat_embedding_dim+num_spatial, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 3)
            )

    def forward(self, question, length, target_category, target_spatial):
        sorted_length, sorted_idx = torch.sort(length, descending=True)
        question = question[sorted_idx]

        word_emb = self.word_emb(question)
        packed_emb = nn.utils.rnn.pack_padded_sequence(word_emb, sorted_length, batch_first=True)
        _, last_hidden = self.rnn(packed_emb)

        last_hidden = last_hidden[0].squeeze(0)

        _, reversed_idx = torch.sort(sorted_idx)
        last_hidden = last_hidden[reversed_idx]

        cat_emb = self.cat_emb(target_category)

        mlp_input = torch.cat((last_hidden, cat_emb, target_spatial), dim=-1)

        logits = self.mlp(mlp_input)

        return logits

class DM1(nn.Module):

    def __init__(self, rnn_hidden_size, attention_hidden):

        super().__init__()

        # https://arxiv.org/abs/1703.03130
        self.scores = nn.Sequential(
            nn.Linear(rnn_hidden_size, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1)
        )

        self.visual_emb = nn.Linear(1000, 512)

        self.mlp = nn.Sequential(
            nn.Linear(1536, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

        # self.mlp = nn.Sequential(
        #     nn.Linear(1024+512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 2)
        # )

    def forward(self, hidden_states, lengths, fc8, masking=True):
        #print(hidden_states)
        attnScores = self.scores(hidden_states)

        if masking:
            # mask attention scores
            B, S, _ = attnScores.size()
            idx = lengths.new_tensor(torch.arange(0, S).unsqueeze(0).repeat(B, 1)).long()
            lengths = lengths.unsqueeze(1).repeat(1, S)

            mask = (idx >= lengths).unsqueeze(2)
            attnScores.masked_fill_(mask, float('-inf'))

        attnWeights = nn.functional.softmax(attnScores, dim=1)

        # multiply attention weights with hidden states
        # bmm: [b, n, m] @ [b, m, p] = [b, n, p]
        attnWeights = attnWeights.transpose(2,1)
        hidden_states = torch.bmm(attnWeights, hidden_states).squeeze(1)

        #raise
        visual_emb = self.visual_emb(fc8)

        logits = self.mlp(torch.cat([hidden_states, visual_emb], dim=-1))

        return logits

class DM2(nn.Module):

    def __init__(self):

        super().__init__()

        self.visual_emb = nn.Linear(1000, 512)

        self.mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

        # self.mlp = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2)
        # )

    def forward(self, hidden_states, fc8):

        visual_emb = self.visual_emb(fc8)

        logits = self.mlp(torch.cat([hidden_states, visual_emb], dim=-1))

        return logits
