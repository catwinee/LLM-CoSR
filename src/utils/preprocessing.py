import json
import torch
from transformers import BertModel, BertTokenizer
import pickle
from gensim.models import KeyedVectors

from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple
import torch.nn as nn
from torch import Tensor
import numpy as np
import pandas as pd
from contractions import contractions_dict
import string
import nltk
nltk.download('punkt')

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def text_processes(text: str) -> List[str]:
    """
    process the raw text
    """
    tokens = word_tokenize(text)
    # filter punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # filter stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # replace abbreviation
    tokens = [contractions_dict[token] if token in contractions_dict.keys() else token for token in tokens]
    # lemmatization
    wnl = WordNetLemmatizer()
    tags = pos_tag(tokens)
    res = []
    for t in tags:
        wordnet_pos = get_wordnet_pos(t[1]) or wordnet.NOUN
        res.append(wnl.lemmatize(t[0], pos=wordnet_pos))
    return res


def delete_long_text(words: List[str], length: int):
    if len(words) > length:
        return words[:length]
    return words

class DataReader:
    def __init__(self, api_path: str, mashup_path: str):
        self.api_path = api_path
        self.mashup_path = mashup_path
        self.apis = json.load(open(api_path, 'r', encoding='utf-8'))
        self.mashups = json.load(open(mashup_path, 'r', encoding='utf-8'))
        self.glove_path = '../../data/glove/glove_300d.txt'
        model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def clean(self, is_save: bool = False):
        r"""
        Delete mashups and APIs that contain empty fields and mashups that have used uncatalogued APIs.

        Args:
            is_save: If true, will save the processed data to the original file.
        """

        apis = []
        mashups = []
        for item in self.apis:
            if item is None or item['title'] == '' or len(item['tags']) == 0 or item['description'] == '':
                continue
            apis.append(item)
        api_titles = [item['title'] for item in apis]
        for item in self.mashups:
            if item is None or item['title'] == '' or item['description'] == '':
                continue
            # delete mashups that have used uncatalogued APIs
            is_legal = 1
            for api in item['related_apis']:
                if api is None or api['title'] not in api_titles:
                    is_legal = 0
                    break
            if is_legal:
                mashups.append(item)
        self.apis = apis
        self.mashups = mashups
        if is_save:
            json.dump(apis, open(self.api_path, 'w', encoding='utf-8'))
            json.dump(mashups, open(self.mashup_path, 'w', encoding='utf-8'))

    def get_mashups(self):
        r"""
        Get mashups.

        """
        return self.mashups

    def get_apis(self, is_total: bool = False):
        r"""
        Get apis.

        Args:
            is_total (bool): If set true, will return the full amount of apis. Else, will return the partial amount of
            APIs without unused APIs. (default: :obj:`False`)
        """
        if is_total:
            return self.apis
        used_apis = []
        for mashup in self.mashups:
            for item in mashup['related_apis']:
                try:
                    used_apis.extend([item['title']])
                except:
                    pass
        used_apis = set(used_apis)
        apis = []
        for item in self.apis:
            if item is not None and item['title'] in used_apis:
                apis.append(item)
        return apis

    def get_mashup_embeddings(self, model: str = 'BERT', num_token: int = 72, description=False) -> Tuple[
        Tensor, Tensor]:
        r"""
        Get the word embeddings of mashups.

        Args:
            model (str): The name of pre-trained language model. The options available are "BERT", "GloVe"
            num_token (int): The number of tokens of each text.

        Returns:
            Word embeddings of mashup, including word-based embeddings and text-based embeddings.
        """
        with open('data/stop_word.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [word[:-1] for word in lines]
        descriptions = ['[CLS]' + ' '.join([word for word in item['description'].split() if word.lower() not in lines]) for item in self.mashups if item['description'] not in lines]
        # if description is True:
        #     descriptions = ['this is the description of mushup:' + item['description'] + \
        #                     ', please describe the features of api that can meet it.' for item in self.mashups]
        # else:
        #     descriptions = [item['description'] for item in self.mashups]
        if model.lower() == 'glove':
            wv = KeyedVectors.load_word2vec_format(self.glove_path)
            descriptions = [text_processes(item) for item in descriptions]
            embeddings = []
            for des in descriptions:
                embeddings = []
                for word in delete_long_text(des, num_token):
                    try:
                        vector = wv[word]
                        vector = np.expand_dims(vector, axis=0)
                    except KeyError as e:
                        vector = np.zeros(shape=(1, 300), dtype=np.float32)
                    embeddings.append(vector)
                if len(embeddings) == 0:
                    embeddings.append(np.zeros(shape=(1, 300), dtype=np.float32))
                embeddings = np.concatenate(embeddings, axis=0)
                if embeddings.shape[0] < num_token:
                    embeddings = np.pad(embeddings, ((0, num_token - embeddings.shape[0]), (0, 0)),
                                        'constant', constant_values=(0.0, 0.0))
                embeddings.append(np.expand_dims(embeddings, axis=0))
            embeddings = np.concatenate(embeddings, axis=0)
        elif model.lower() == 'bert':
            output = self.tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt", max_length=60)
            with torch.no_grad():
                embeddings = self.model(**output).last_hidden_state.detach()
        else:
            raise ValueError('Illegal pre-trained model')

        if description is False:
            torch.save(embeddings.mean(axis=1), 'data/mashup_embeddings.emb')
        else:
            torch.save(embeddings.mean(axis=1), 'data/bert_mashup_embeddings.emb')

        return embeddings, embeddings.mean(axis=1)

    def get_api_embeddings(self, model: str = 'bert', num_token: int = 72, is_total: bool = False) -> Tuple[
        Tensor, Tensor]:
        r"""
        Get the word embeddings of apis.

        Args:
            model (str): The name of pre-trained language model. The options available are "BERT", "GloVe".
            num_token (int): the number of tokens of each text.
            is_total (bool): If True, will return the full amount of data. Else, will return the partial amount of data
            without unused APIs. (default: :obj:`False`)

        Returns:
            Word embeddings of mashup, including word-based embeddings and text-based embeddings.
        """
        with open('stop_word.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [word[:-1] for word in lines]
        apis = self.get_apis(is_total)
        descriptions = [' '.join([word for word in item['description'].split() if word.lower() not in lines]) for item in apis if item['description'] not in lines]
        names = [[item['title'][-18:] for item in apis]]
        tags = [list(set(item['tags'])) for item in apis]
        if model.lower() == 'glove':
            wv = KeyedVectors.load_word2vec_format(self.glove_path)
            descriptions = [text_processes(item) for item in descriptions]
            embeddings = []
            for des in descriptions:
                embeddings = []
                for word in delete_long_text(des, num_token):
                    try:
                        vector = wv[word]
                        vector = np.expand_dims(vector, axis=0)
                    except KeyError as e:
                        vector = np.zeros(shape=(1, 300), dtype=np.float32)
                    embeddings.append(vector)
                if len(embeddings) == 0:
                    embeddings.append(np.zeros(shape=(1, 300), dtype=np.float32))
                embeddings = np.concatenate(embeddings, axis=0)
                if embeddings.shape[0] < num_token:
                    embeddings = np.pad(embeddings, ((0, num_token - embeddings.shape[0]), (0, 0)),
                                        'constant', constant_values=(0.0, 0.0))
                embeddings.append(np.expand_dims(embeddings, axis=0))
            embeddings = np.concatenate(embeddings, axis=0)
        elif model.lower() == 'bert':
            output = self.tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt", max_length=100)
            with torch.no_grad():
                embeddings = self.model(**output).last_hidden_state.detach()
        else:
            raise ValueError('Illegal pre-trained model')

        return embeddings, embeddings.mean(axis=1)

    def get_invocation(self, is_total: bool = False) -> pd.DataFrame:
        r"""
        Get invocation between mashups and APIs.

        Args:
            is_total (bool): If True, will return the full amount of data. Else, will return the partial amount of data
            without unused APIs. (default: :obj:`False`)
        """
        apis = self.get_apis(is_total=is_total)
        apis_idx = [api['title'] for api in apis]
        related_apis = []
        for m in self.mashups:
            mashups = []
            for ral in m['related_apis']:
                try:
                    mashups.append(str(apis_idx.index(ral['title'])))
                except:
                    pass
            related_apis.append(mashups.copy())
        Xs = []
        Ys = []
        times = []
        for i in range(len(self.mashups)):
            inv = [str(i)]
            inv.extend(related_apis[i])
            Xs.append(i)
            Ys.append([int(api) for api in related_apis[i]])
            times.append(self.mashups[i]['date'].strip())
        df = pd.DataFrame({
            'index': pd.Series(range(len(self.mashups))),
            'X': pd.Series(Xs),
            'Y': pd.Series(Ys),
            'time': pd.Series(times)
        })
        return df

    def get_service_domain_embeddings(self, is_total: bool = False, service_embeddings: np.ndarray = None):
        r"""
        Get embeddings of service domain. A service domain refers to a collection of Web APIs of the same category.

        Args:
            is_total (bool): If True, will return the service domains created from the full amount of data. Else, will
            return the service domains created from the partial amount of data. (default: :obj:`False`)
            service_embeddings (List[np.ndarray]): Embeddings of service (Web API).

        """
        apis = self.get_apis(is_total)
        api_categories = [a['tags'][0] if len(a['tags']) > 0 else 'None' for a in apis]
        categories = list(set(api_categories))
        domains = [[] for _ in range(len(categories))]
        for idx, api_cate in enumerate(api_categories):
            domains[categories.index(api_cate)].append(idx)
        domain_embeddings = []
        for apis in domains:
            domain_embeddings.append(np.expand_dims(service_embeddings[apis].mean(axis=0), axis=0))
        domain_embeddings = np.concatenate(domain_embeddings, axis=0)
        return domain_embeddings

    def get_invoked_matrix(self, is_total: bool = False) -> np.ndarray:
        r"""
        Get the invoked matrix M between mashups and APIs, whose size is (num_mashup, num_api). $M_{ij}=1$ if the $i$-th
        mashup used the $j$-th API. Else, $M_{ij}=0$

        Args:
            is_total (bool): If True, will return the invoked matrix created from the full amount of data. Else, will
            return the invoked matrix created from the partial amount of data. (default: :obj:`False`)

        """
        num_mashup = len(self.mashups)
        num_api = len(self.get_apis(is_total))
        invoked_df = self.get_invocation(is_total)
        Xs = invoked_df['X'].tolist()
        Ys = invoked_df['Y'].tolist()
        invoked_matrix = np.zeros(shape=(num_mashup, num_api), dtype=np.int64)
        edges = []
        for x, y in zip(Xs, Ys):
            for index in y:
                invoked_matrix[x][index] = 1
                edges.append(np.array([x, index + num_mashup]))
                # edges.append(np.array([index+num_mashup, x]))
        return np.array(edges)

    def word_preprocessing(self, remake_dataset=True, save_text_len=True):
        if remake_dataset:
            with open('stop_word.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            lines = [word[:-1] for word in lines]
            bert_common_words = list(self.tokenizer.vocab.keys())
            apis = self.get_apis(False)
            api_descriptions = [' '.join([word for word in item['description'].split() if word.lower() not in lines]) for item in apis if item['description'] not in lines]
            api_names = [item['title'][:-18] for item in apis]
            api_tags = [list(set(item['tags'])) for item in apis]
            mashup_descriptions = ['[CLS]' + ' '.join([word for word in item['description'].split() if word.lower() not in lines])
                            for item in self.mashups if item['description'] not in lines]
            mashup_names = [item['title'][8:] for item in self.mashups]
            entity_list = api_names + mashup_names
            tag_list = list(set(tag for tags in api_tags for tag in tags))
            tag_word_to_id = {word: idx for idx, word in enumerate(tag_list)}
            tag_embeddings = nn.Embedding(len(tag_word_to_id), embedding_dim=768)

            original_weight_matrix = tag_embeddings.weight.data

            # 创建一个全零向量，维度与Embedding层的嵌入维度一致
            zero_vector = torch.zeros(1, original_weight_matrix.size(1))

            # 将全零向量添加到权重矩阵的末尾
            extended_weight_matrix = torch.cat((original_weight_matrix, zero_vector), dim=0)

            # 创建一个新的Embedding层，并将新的权重矩阵应用到该层中
            tag_embeddings = nn.Embedding.from_pretrained(extended_weight_matrix)

            torch.save(tag_embeddings, '../preprocessed_data/sentence_embedding_with_textlen/tag.emb')
            with open('../preprocessed_data/sentence_embedding_with_textlen/tag_word_id', 'wb') as f:
                pickle.dump(tag_word_to_id, f)
            id_to_word = {}
            for idx, word in enumerate(entity_list):
                id_to_word[idx] = word
            # word_to_id = {word:idx for idx, word in enumerate(entity_list)}
            entity_embeddings = nn.Embedding(len(id_to_word), embedding_dim=768)
            with open('../preprocessed_data/sentence_embedding_with_textlen/entity_id_word', 'wb') as f:
                pickle.dump(id_to_word, f)
            torch.save(entity_embeddings, '../preprocessed_data/sentence_embedding_with_textlen/entity_embeddings.emb')
            entity_discriptions = api_descriptions + mashup_descriptions
            entity_discriptions_after_match = []
            for entity, discription in zip(entity_list, entity_discriptions):
                uttr_words = [word for word in word_tokenize(discription)]
                result = []
                i = 0
                while i < len(uttr_words):
                    matched = False
                    for j in range(len(uttr_words), i, -1):
                        if ' '.join(uttr_words[i:j]) in entity and ' '.join(uttr_words[i:j]) not in bert_common_words:
                            result.append('[UNK]')
                            i = j
                            matched = True
                            break
                    if not matched:
                        result.append(uttr_words[i])
                        i += 1
                entity_discriptions_after_match.append(' '.join(result))
        else:
            with open('../preprocessed_data/sentence_embedding_with_textlen/entity_discriptions_after_match', 'rb') as f:
                entity_discriptions_after_match = pickle.load(f)
        # unkown_entity_list = [entity for entity in entity_list+tag_list if entity not in bert_common_words]
        # for word in unkown_entity_list:
        #     self.tokenizer.add_tokens(word)
        # print(len(self.tokenizer.vocab.keys()))
        batch_size = 128
        discription_list = []
        # for i in range(len(entity_discriptions_after_match)//batch_size+1):
        #     if len(entity_discriptions_after_match) < batch_size * (i + 1):
        #         end = len(entity_discriptions_after_match)
        #     else:
        #         end = batch_size * (i + 1)
        #     discription = entity_discriptions_after_match[batch_size * i:end]
        #     output = self.tokenizer(discription, padding=True, truncation=True, return_tensors="pt", max_length=100)
        #     with torch.no_grad():
        #         embeddings = self.model(**output).last_hidden_state.detach()
        #     discription_list.append(embeddings)
        for i in range(len(entity_discriptions_after_match)):
            discription = entity_discriptions_after_match[i][:200]
            output = self.tokenizer(discription, padding=True, truncation=True, return_tensors="pt", max_length=200).to('cuda:0')
            with torch.no_grad():
                embeddings = self.model(**output).last_hidden_state.detach()
            discription_list.append(embeddings)
        if save_text_len:
            from torch.nn.utils.rnn import pad_sequence
            discription_list1 = [i.squeeze(0) for i in discription_list]
            padded_tensor = pad_sequence(discription_list1, batch_first=True)
            torch.save(padded_tensor, '../preprocessed_data/sentence_embedding_with_textlen/description_textlen.emb')
        else:
            discription_list1 = [torch.mean(discription, dim=1) for discription in discription_list]
            discription_list1 = torch.cat(discription_list1, dim=0)
            discription_list2 = [discription[:, 0, :] for discription in discription_list]
            discription_list2 = torch.cat(discription_list2, dim=0)
            torch.save(discription_list1, '../preprocessed_data/sentence_embedding_with_textlen/description.emb')
            torch.save(discription_list2, '../preprocessed_data/sentence_embedding_with_textlen/description_cls.emb')
            print(1)


if __name__ == '__main__':
    data = DataReader('api_mashup/active_apis_data.txt', 'api_mashup/active_mashups_data.txt')
    a = data.word_preprocessing()
    a = data.get_invocation()
    print('ok')
