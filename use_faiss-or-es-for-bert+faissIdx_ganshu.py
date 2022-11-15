#!/usr/bin/env python
# coding: utf-8
import os.path
import time

import jieba
from transformers import AutoModelWithLMHead, BertForMaskedLM, AutoTokenizer, AutoConfig, AutoModelForMaskedLM, RoFormerModel, RoFormerTokenizer
import torch

import numpy as np
import faiss
import argparse
from sanic import Sanic
from sanic.response import json as sanic_json
import pickle
import es_search
import requests
import json
import torch.nn as nn
import math
from thefuzz import process

app = Sanic("aaa")


class RoFormerModelWithPooler(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.roformer = RoFormerModel.from_pretrained(model_path)

        model_file = os.path.join(model_path, "pytorch_model.bin")
        assert os.path.isfile(model_file)
        params_dict = torch.load(model_file)
        pooler_weight = params_dict["pooler.dense.weight"]
        pooler_bias = params_dict["pooler.dense.bias"]
        del params_dict
        self.pooler = nn.Linear(pooler_weight.shape[0], pooler_weight.shape[0])
        self.activation = nn.Tanh()
        self.pooler.weight.data = pooler_weight
        self.pooler.bias.data = pooler_bias

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)

        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        pooled_output = self.pooler(cls_output)
        pooled_output = self.activation(pooled_output)
        return (sequence_output, pooled_output) + outputs[1:]


def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def cos_by_string(str1, str2):
    """
    计算字符串的余弦距离
    :param str1: 字符串1
    :param str2: 字符串2
    :return: 返回相似度"""

    cut_str1 = list(str1.replace(" ", ""))
    cut_str2 = list(str2.replace(" ", ""))

    all_char = set(cut_str1 + cut_str2)
    #     print("all_char:", all_char)

    freq_str1 = [cut_str1.count(x) for x in all_char]
    #     print("freq_str1:", freq_str1)

    freq_str2 = [cut_str2.count(x) for x in all_char]
    #     print("freq_str2:", freq_str2)

    sum_all = sum(map(lambda z, y: z * y, freq_str1, freq_str2))
    #     print(list(map(lambda z, y: z * y, freq_str1, freq_str2)))

    sqrt_str1 = math.sqrt(sum(x ** 2 for x in freq_str1))
    sqrt_str2 = math.sqrt(sum(x ** 2 for x in freq_str2))
    return sum_all / (sqrt_str1 * sqrt_str2)


def get_args():
    parser = argparse.ArgumentParser(description="set arg ...")
    parser.add_argument("--bert_sentence_avg_vec_path_list", type=str, default="bert_sentence_avg_vec_path_list")
    parser.add_argument("--bert_sentence_path_list", type=str, default="bert_sentence_path_list")

    parser.add_argument("--query_bert_sentence_avg_vec_path", type=str, default="query_bert_sentence_avg_vec_path_list")
    parser.add_argument("--query_bert_sentence_path", type=str, default="query_bert_sentence_path_list")
    parser.add_argument("--ncentroids", type=int, default=8)
    parser.add_argument("--niter", type=int, default=200)
    parser.add_argument("--top_size_list", type=str, default=10)
    args = parser.parse_args()
    return args


def get_bert_sentence_vev_docid_by_args(args):
    bert_sentence_avg_vec_path_list = args.bert_sentence_avg_vec_path_list.split(";")
    bert_sentence_path_list = args.bert_sentence_path_list.split(";")

    query_bert_sentence_avg_vec_path = args.query_bert_sentence_avg_vec_path
    query_bert_sentence_path = args.query_bert_sentence_path

    ncentroids = args.ncentroids
    niter = args.niter
    top_size_list = [int(top_size) for top_size in args.top_size_list.split(";")]


    print("bert_sentence_avg_vec_path_list ",bert_sentence_avg_vec_path_list)
    print("bert_sentence_path_list ",bert_sentence_path_list)
    print("query_bert_sentence_avg_vec_path ",query_bert_sentence_avg_vec_path)
    print("query_bert_sentence_path ", query_bert_sentence_path)
    print("ncentroids ", ncentroids)
    print("niter ", niter)
    print("top_size_list ",top_size_list)
    print(type(top_size_list[0]))

    return bert_sentence_avg_vec_path_list, bert_sentence_path_list, query_bert_sentence_avg_vec_path, query_bert_sentence_path, ncentroids, niter, top_size_list


def mean_pooling(token_embeddings, attention_mask):
    attention_mask = torch.unsqueeze(attention_mask, dim=-1)
    token_embeddings = token_embeddings * attention_mask
    seqlen = torch.sum(attention_mask, dim=1)
    embeddings = torch.sum(token_embeddings, dim=1) / seqlen
    return embeddings


def encode(sentences, batch_size=8, normalize_to_unit=True, convert_to_numpy=True, tokenizer=None, model=None, get_sen_vector_method="pool", mode=None):
    input_was_string = False
    if isinstance(sentences, str):
        sentences = [sentences]
        input_was_string = True

    all_embeddings = []
    if len(sentences) < 1:
        return all_embeddings
    length_sorted_idx = np.argsort(
        [-len(sen) for sen in sentences])  # 句子从长到短排序时，每个句子所在的位置。如第一个句子应该处于第几个位置，第二个句子应该处于什么位置
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]  # 根据顺序将句子从长到短进行排序
    num_batches = int((len(sentences) - 1) / batch_size) + 1
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(sentences_sorted))
            inputs = tokenizer(
                sentences_sorted[start:end],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)

            if isinstance(model, BertForMaskedLM):
                inputs["output_hidden_states"] = True

            outputs = model(**inputs)

            if mode in ["simbert-base", "roformer-sim-base", "roformer-sim-small"] and not (
            isinstance(model, BertForMaskedLM)):
                embeddings = outputs[1]
            else:
                if isinstance(model, BertForMaskedLM):
                    outputs = outputs["hidden_states"][-1]  # [766, 33, 768]
                    # print("is BertForMaskedLM model, outputs.shape", outputs.shape)
                if get_sen_vector_method == "pool":
                    # print(" \t\t pool output.shape", outputs.shape)
                    # print(" \t\t pool outputs[0].shape", outputs.shape)
                    embeddings = mean_pooling(outputs, inputs["attention_mask"]) # [37, 768]
                    # print("\t\t pool embeddings.shape", embeddings.shape) # [766, 768]
                else:
                    # print("\toutputs.shape", outputs.shape)
                    embeddings = outputs[:, 0]
                    # print("\t\t cls output", embeddings.shape)

            if normalize_to_unit:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            if convert_to_numpy:
                embeddings = embeddings.cpu()

            # print(embeddings.shape)
            all_embeddings.extend(embeddings)

    # 对排序后的索引再次进行排序，应该是为了回复成原始形式
    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

    if convert_to_numpy:
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings]).astype('float32')
    else:
        all_embeddings = torch.stack(all_embeddings)
    if input_was_string:
        all_embeddings = all_embeddings[0]
    return all_embeddings


def read_sentence_docid_vec(sentence_path_list=None, vec_path_list=None, verbose=False):
    """
    :param sentence_and_docId_path_list: 是一个列表，包含了所有句子的路径
    :param vec_path_list: 是一个列表，包含了所有向量的路径
    :param verbose: 是否打印日志
    :param docID_path_list: 就在的
    :return:
    """
    assert sentence_path_list and vec_path_list
    assert len(sentence_path_list) == len(vec_path_list)

    bert_sentence_path_list = sentence_path_list
    bert_sentence_avg_vec_path_list = vec_path_list

    sentence_vec_list = []
    all_sentence_list = []
    for idx, (vec_path, sen_path) in enumerate(zip(bert_sentence_avg_vec_path_list, bert_sentence_path_list)):
            # sentence_avg_vec_0 = pickle.load(open("./bert_sentence_avg_vec-0_np.pkl", "rb"))
            if idx == 0:
                if verbose:
                    start_time = time.time()
            else:
                if verbose:
                    print(time.time()-start_time)
                    start_time = time.time()

            print("load vec path {}, sen_path {} , cost time: ".format(vec_path, sen_path), end="\t")
                 
            sentence_vec_list.append(pickle.load(open(vec_path, "rb")))

            if sen_path:
                sen_docid_json_list = pickle.load(open(sen_path, "rb"))
                for sen_docid_json in sen_docid_json_list:
                    all_sentence_list.append(sen_docid_json["title"])

            if len(sentence_vec_list) == 2:
                # all_avg_vec = np.concatenate([sentence_avg_vec_0, sentence_avg_vec_1], axis=0)
                sentence_vec_list[0] = np.concatenate([sentence_vec_list[0], sentence_vec_list[1]], axis=0)
                del sentence_vec_list[1]
                assert len(sentence_vec_list) == 1
    print("len(all_sentence_list):", len(all_sentence_list))
    return sentence_vec_list[0], all_sentence_list


def read_sentence(sentence_path_list=None, verbose=True):
    assert sentence_path_list
    bert_sentence_path_list = sentence_path_list
    all_sentence_list = []
    for idx, sen_path in enumerate(bert_sentence_path_list):
        # sentence_avg_vec_0 = pickle.load(open("./bert_sentence_avg_vec-0_np.pkl", "rb"))
        if idx == 0:
            if verbose:
                start_time = time.time()
        else:
            if verbose:
                print(time.time() - start_time)
                start_time = time.time()
        print("load sen_path {} , cost time: ".format(sen_path), end="\t")
        if sen_path:
            
            sen_docid_json_list = pickle.load(open(sen_path, "rb"))
            for sen_docid_json in sen_docid_json_list:
                all_sentence_list.append(sen_docid_json["title"])
    print("len(all_sentence_list):", len(all_sentence_list))
    return all_sentence_list


def get_faiss_index(all_sentence_vec=None, ncentroids=50, niter=200, verbose=True, faiss_idx_use_method=None, faiss_idx_use_kmeans=True, efSearch=128, nprobe=64, bounded_queue=False):
    xb = all_sentence_vec
    print(all_sentence_vec.shape)
    d = all_sentence_vec.shape[1]

    final_faiss_index = None
    # 聚类
    faiss.normalize_L2(all_sentence_vec)

    if faiss_idx_use_method == "IndexFlatIP":
        if faiss_idx_use_kmeans:
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
            kmeans.train(all_sentence_vec)
        # 构建索引
        final_faiss_index = faiss.IndexFlatIP(d)
        final_faiss_index.add(all_sentence_vec)
    elif faiss_idx_use_method == "HNSW":
        print("Testing HNSW Flat")
        final_faiss_index = faiss.IndexHNSWFlat(d, 32)
        # training is not needed
        # this is the default, higher is more accurate and slower to construct
        final_faiss_index.hnsw.efConstruction = 40
        print("add")
        # to see progress
        final_faiss_index.verbose = True
        final_faiss_index.add(all_sentence_vec)
        # 在搜索时使用
        print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
        final_faiss_index.hnsw.search_bounded_queue = bounded_queue
        final_faiss_index.hnsw.efSearch = efSearch
    elif faiss_idx_use_method == "hnsw_sq":  # 注意搜索的时候，方法也会改变
        print("Testing HNSW with a scalar quantizer")
        # also set M so that the vectors and links both use 128 bytes per
        # entry (total 256 bytes)
        final_faiss_index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 16)

        print("training")
        # training for the scalar quantizer
        final_faiss_index.train(all_sentence_vec)

        # this is the default, higher is more accurate and slower to
        # construct
        final_faiss_index.hnsw.efConstruction = 40

        print("add")
        # to see progress
        final_faiss_index.verbose = True
        final_faiss_index.add(all_sentence_vec)
        print("efSearch", efSearch, end=' ')
        final_faiss_index.hnsw.efSearch = efSearch
    elif faiss_idx_use_method == "ivf":
        print("Testing IVF Flat (baseline)")
        quantizer = faiss.IndexFlatL2(d)
        final_faiss_index = faiss.IndexIVFFlat(quantizer, d, 16384)
        final_faiss_index.cp.min_points_per_centroid = 5  # quiet warning
        # to see progress
        final_faiss_index.verbose = True
        print("training")
        final_faiss_index.train(xb)
        print("add")
        final_faiss_index.add(xb)
        print("nprobe", nprobe, end=' ')
        final_faiss_index.nprobe = nprobe
    elif faiss_idx_use_method == "ivf_hnsw_quantizer":
        print("Testing IVF Flat with HNSW quantizer")
        quantizer = faiss.IndexHNSWFlat(d, 32)
        final_faiss_index = faiss.IndexIVFFlat(quantizer, d, 16384)
        final_faiss_index.cp.min_points_per_centroid = 5  # quiet warning
        final_faiss_index.quantizer_trains_alone = 2

        # to see progress
        final_faiss_index.verbose = True

        print("training")
        final_faiss_index.train(xb)

        print("add")
        final_faiss_index.add(xb)

        print("search")
        quantizer.hnsw.efSearch = 64
        print("nprobe", nprobe, end=' ')
        final_faiss_index.nprobe = nprobe
    return final_faiss_index


def get_finetune_model(model_name="bert", model_name_or_path="../../train_chinese_wwn_bert/continue-300w_model_output_path", device="cuda:4"):
    assert model_name in ["bert", "roformer-sim"]
    if model_name == "bert":
        mode = "bert"
        model_name_or_path = model_name_or_path
        model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
        model.to(torch.device(device if torch.cuda.is_available() else "cpu"))
        model.eval()
    elif model_name == "roformer-sim":
        mode = "roformer-sim"
        model_name_or_path = "../finetune_roformer_sim/test-mlm"
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir="./roformer_sim_finetune"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model.resize_token_embeddings(len(tokenizer))
    model.to(torch.device(device if torch.cuda.is_available() else "cpu"))
    model.eval()
    return tokenizer, model


def similarity(input_title=None, retrieve_title=None, model=None, get_sen_vector_method=None, tokenizer=None):
    print("input_title:", input_title)
    query_vecs = encode(input_title, normalize_to_unit=True, model=model, get_sen_vector_method=get_sen_vector_method, tokenizer=tokenizer)
    # retrieve_title = eval(retrieve_title)

    if isinstance(retrieve_title, str):
        all_key_title_list = retrieve_title.split("|||")
    elif isinstance(retrieve_title, list):
        all_key_title_list = retrieve_title

    # print("\t all similarity title: ", all_key_title_list)

    key_vecs = encode(all_key_title_list, batch_size=100,
                      normalize_to_unit=True, model=model, get_sen_vector_method=get_sen_vector_method, tokenizer=tokenizer)

    single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1

    if single_query:
        query_vecs = query_vecs.unsqueeze(0)
    if single_key:
        if isinstance(key_vecs, np.ndarray):
            key_vecs = key_vecs.reshape(1, -1)
        else:
            key_vecs = key_vecs.unsqueeze(0)

    # similarity_list = (query_vecs @ key_vecs.T)[0].tolist()
    similarity_list = torch.cosine_similarity(torch.tensor(query_vecs), torch.tensor(key_vecs), dim=-1).tolist()
    return similarity_list


def es_search_and_filter(input_title=None, search_size=None, new_good_word_dict=None, es=None, jieba_stop=None, all_title_and_docid_dict=None, use_sim_former=False):
    # es_search_title_main(title, num=3, min_score=30, size=5, new_good_word_dict=None, es=None, jieba_stop=None)
    es_result_title = es_search.es_search_title_main(input_title, size=search_size,
                                                     new_good_word_dict=new_good_word_dict, es=es,
                                                     jieba_stop=jieba_stop)
    print("input_title:", input_title)
    result_list = []

    for data in es_result_title:
        if all_title_and_docid_dict.get(data["_source"]["title"], None):
            result_list.append(
                (data["_source"]["title"], all_title_and_docid_dict[data["_source"]["title"]], data["_score"]))
        elif all_title_and_docid_dict.get(data["_source"]["title"].upper(), None):
            result_list.append(
                (data["_source"]["title"].upper(), all_title_and_docid_dict[data["_source"]["title"].upper()],
                 data["_score"]))
        elif all_title_and_docid_dict.get(data["_source"]["title"].lower(), None):
            result_list.append(
                (data["_source"]["title"].lower(), all_title_and_docid_dict[data["_source"]["title"].lower()],
                 data["_score"]))
        else:
            result_list.append(
                (data["_source"]["title"], "", data["_score"]))

    # 对所有的es分数使用最大最小进行归一化或softmax归一化
    all_es_title = [tmp[0] for tmp in result_list]
    all_es_score = [tmp[-1] for tmp in result_list]
    softmax_score_for_es = softmax(np.array([all_es_score]))[0]

    # 对es检索出的句子再使用simformer计算相似度
    if use_sim_former:
        print("src={}; tgt={}".format(input_title, all_es_title))

        input_es_similarity_dict = sim_former(src=input_title, tgt=all_es_title)
        all_roformer_score = list(input_es_similarity_dict.values())
        softmax_score_for_roformer = softmax(np.array([all_roformer_score]))[0]
    

    # 对es检索出的的句子再使用字符串余弦距离计算相似度，并进行归一化
    cos_by_string_score = [cos_by_string(input_title, tmp) for tmp in all_es_title]
    softmax_score_for_cosString = softmax(np.array([cos_by_string_score]))[0]

    # 对es检索出的句子再使用模糊匹配计算相似度，并进行归一化
    fuzzy_match_score_sort_list = process.extract(input_title, all_es_title, limit=len(all_es_title))
    fuzzy_match_score_dict = {}
    for title_score_tuple in fuzzy_match_score_sort_list:
        fuzzy_match_score_dict[title_score_tuple[0]] = title_score_tuple[1]
    fuzzy_match_score_list = [fuzzy_match_score_dict[tmp] for tmp in all_es_title]
    softmax_score_for_fuzzyMatch = softmax(np.array([fuzzy_match_score_list]))[0]

    # 将四个的分数进行结合
    total_score = []
    if use_sim_former:
        for es_score, roformer_score, cosString_score, fuzzyMath_score in zip(softmax_score_for_es,
                                                                          softmax_score_for_roformer,
                                                                          softmax_score_for_cosString,
                                                                          softmax_score_for_fuzzyMatch):
            total_score.append(
            1 / 3 * es_score + 2 / 4 * roformer_score + 1 / 6 * cosString_score + 1 / 6 * fuzzyMath_score)
    else:
        for es_score, cosString_score, fuzzyMath_score in zip(softmax_score_for_es,
                                                                          softmax_score_for_cosString,
                                                                          softmax_score_for_fuzzyMatch):
            total_score.append(
            1 / 3 * es_score + 1 / 6 * cosString_score + 1 / 6 * fuzzyMath_score)
    # 得到top-5的标题、分数
    top_5_idx_list = np.array(total_score).argsort()[::-1][:5].tolist()
    top_5_title_docid_esScore = [result_list[top_5_idx] for top_5_idx in top_5_idx_list]

    return top_5_title_docid_esScore


def sim_former(src, tgt, tag="single", logger=None):
    url_content = "http://192.168.11.247:60052/compute_sim"
    if tag == "single":
        data = {"src": [src], "tgt": [tgt]}
        # content = requests.post(url_content, data=json.dumps(data))
        headers = {'Content-Type': 'application/json'}
        content = requests.post(url_content, headers=headers, data=json.dumps(data))

        knowledge_output = content.json()["result"]
        knowledge_output = dict(zip(tgt, knowledge_output[0]))
        # knowledge_output = sorted(knowledge_output.items(), key=lambda x: x[1], reverse=True)

        return knowledge_output

    else:
        data = {"src": src, "tgt": tgt}

        headers = {'Content-Type': 'application/json'}
        content = requests.post(url_content, headers=headers, data=json.dumps(data))
        knowledge_output = content.json()["result"]

        return knowledge_output


global all_sentence_vec, all_sentence_list, sentence_path_list, vec_path_list
# sentence_path_list = "./bert_sentence_sentence-0_list.pkl;./bert_sentence_sentence-1_list.pkl;./bert_sentence_sentence-2_list.pkl;./bert_sentence_sentence-3_list.pkl".split(";")
# docID_path_list = "./bert_sentence_sentence-0_list.pkl;./bert_sentence_sentence-1_list.pkl;./bert_sentence_sentence-2_list.pkl;./bert_sentence_sentence-3_list.pkl".split(";")
# vec_path_list = "./bert_sentence_avg_vec-0_np.pkl;./bert_sentence_avg_vec-1_np.pkl;./bert_sentence_avg_vec-2_np.pkl;./bert_sentence_avg_vec-3_np.pkl".split(";")

sentence_path_list = "./new_bert_sentence-0_list.pkl;./new_bert_sentence-1_list.pkl;./new_bert_sentence-2_list.pkl;./new_bert_sentence-3_list.pkl".split(";")
vec_path_list = "./new_bert_sentence_avg_vec-0_np.pkl;./new_bert_sentence_avg_vec-1_np.pkl;./new_bert_sentence_avg_vec-2_np.pkl;./new_bert_sentence_avg_vec-3_np.pkl".split(";")


if os.path.exists("all_title_and_docid_dict.pkl"):
    all_title_and_docid_dict=pickle.load(open("all_title_and_docid_dict.pkl", "rb"))
else:
    all_title_and_docid_dict={}
    with open("new_all_title_and_dockid_from_done_jsonl.txt", "r", encoding="utf-8") as fin:
        for line in fin:
            sen_docid_json = json.loads(line)
            if sen_docid_json["title"] not in all_title_and_docid_dict:
                all_title_and_docid_dict[sen_docid_json["title"]] = [sen_docid_json["docId"]]
            else:
                all_title_and_docid_dict[sen_docid_json["title"]].append(sen_docid_json["docId"])
    pickle.dump(all_title_and_docid_dict, open("all_title_and_docid_dict.pkl", "wb"))

print("\t 读取句子完成....., len(all_title_and_docid_dict):", len(all_title_and_docid_dict))
#etimport sys;sys.exit()

global faiss_index
es, jieba_stop, new_good_word_dict = es_search.get_esObject_jiebaStop_newGoddWordDict()
# 默认加载bert-pool索引

################ HNSW
efSearch=64
bounded_queue=True
if os.path.exists("./faiss-bert-pool-HNSW-efSearch{}_add_data.index".format(efSearch)):
    print("\t load index: ./faiss-bert-pool-HNSW-efSearch{}_add_data.index".format(efSearch))
    faiss_index = faiss.read_index("./faiss-bert-pool-HNSW-efSearch{}_add_data.index".format(efSearch))
    # 只读取句子
    all_sentence_list = read_sentence(sentence_path_list)
else:
    all_sentence_vec, all_sentence_list = read_sentence_docid_vec(sentence_path_list=sentence_path_list,
                                                                  vec_path_list=vec_path_list, verbose=True)

    print("all_sentence_vec.shape:", all_sentence_vec.shape)

    # get_faiss_index(all_sentence_vec=None, ncentroids=50, niter=200, verbose=True, faiss_idx_use_method=None, faiss_idx_use_kmeans=True, efSearch=128, nprobe=64, bounded_queue=False)
    faiss_index = get_faiss_index(all_sentence_vec=all_sentence_vec, ncentroids=5, niter=200, verbose=True,
                                  efSearch=efSearch, bounded_queue=bounded_queue, faiss_idx_use_method="HNSW")
    # 保存索引
    faiss.write_index(faiss_index, "./faiss-bert-pool-HNSW-efSearch{}_add_data.index".format(efSearch))
    print("\t save index: ./faiss-bert-pool-HNSW-efSearch{}_add_data.index".format(efSearch))

device = "cuda:2"
global tokenizer, model
try:
    tokenizer, model = get_finetune_model(model_name="bert", model_name_or_path="../../train_chinese_wwn_bert/continue-300w_model_output_path", device=device)
except:
    tokenizer, model = get_finetune_model(model_name="bert",
                                          model_name_or_path="./chinese_bert")


global faiss_idx_use_method_before
faiss_idx_use_method_before = "HNSW"

# @app.post("/modify_outline")
@app.route("/retrieve_similarity_sentence", methods=['POST'])
async def retrieve_similarity_sentence(request):
    global faiss_index, all_sentence_list, tokenizer, model, vec_path_list, sentence_path_list, faiss_idx_use_method_before

    start_time = time.time()

    item = request.json
    input_title = item["input_title"]
    retrieve_method = item.get("retrieve_method", "faiss")
    cal_cos_similarity_when_es = item.get("cal_cos_similarity_when_es", False)
    assert retrieve_method in ["faiss", "es", "both", "faiss_es"]

    faiss_idx_use_method = item.get("faiss_idx_use_method", "IndexFlatIP")
    efSearch = int(item.get("efSearch", 28))
    nprobe = int(item.get("nprobe", 64))
    bounded_queue = bool(item.get("bounded_queue", True))

    use_sim_former = bool(item.get("use_sim_former", False))

    # 默认加载使用的是IndexFlatIP,如果不是该索引方式，则会重新加载索引，记得注意保存索引
    if faiss_idx_use_method_before != faiss_idx_use_method:
        faiss_idx_use_method_before = faiss_idx_use_method
        if faiss_idx_use_method == "HNSW":
            efSearch = 64
            bounded_queue = True
            if os.path.exists("./faiss-bert-pool-HNSW-efSearch{}.index".format(efSearch)):
                print("\t load index: ./faiss-bert-pool-HNSW-efSearch{}.index".format(efSearch))
                faiss_index = faiss.read_index("./faiss-bert-pool-HNSW-efSearch{}.index".format(efSearch))
                # 只读取句子
                all_sentence_list = read_sentence(sentence_path_list)
            else:
                all_sentence_vec, all_sentence_list = read_sentence_docid_vec(sentence_path_list=sentence_path_list,
                                                                              vec_path_list=vec_path_list, verbose=True)
                # get_faiss_index(all_sentence_vec=None, ncentroids=50, niter=200, verbose=True, faiss_idx_use_method=None, faiss_idx_use_kmeans=True, efSearch=128, nprobe=64, bounded_queue=False)
                faiss_index = get_faiss_index(all_sentence_vec=all_sentence_vec, ncentroids=5, niter=200, verbose=True,
                                              efSearch=efSearch, bounded_queue=bounded_queue,
                                              faiss_idx_use_method="HNSW")
                # 保存索引
                faiss.write_index(faiss_index, "./faiss-bert-pool-HNSW-efSearch{}.index".format(efSearch))
                print("\t save index: ./faiss-bert-pool-HNSW-efSearch{}.index".format(efSearch))
        elif faiss_idx_use_method == "hnsw_sq":
            # 默认加载bert-pool索引
            if os.path.exists("./faiss-bert-pool-hnsw_sq.index"):
                print("\t load index: faiss-bert-pool-hnsw_sq.index ......")
                faiss_index = faiss.read_index("./faiss-bert-pool-hnsw_sq.index")
                # 只读取句子
                all_sentence_list = read_sentence(sentence_path_list)
            # 默认加载bert-pool索引
        elif faiss_idx_use_method == "ivf":
            if os.path.exists("./faiss-bert-pool-ivf.index"):
                print("\t load index: faiss-bert-pool-ivf.index ......")
                faiss_index = faiss.read_index("./faiss-bert-pool-ivf.index")
                # 只读取句子
                all_sentence_list = read_sentence(sentence_path_list)
        elif faiss_idx_use_method == "ivf_hnsw_quantizer":
            # 默认加载bert-pool索引
            if os.path.exists("./faiss-bert-pool-ivf_hnsw_quantizer.index"):
                print("\t load index: faiss-bert-pool-ivf_hnsw_quantizer.index ......")
                faiss_index = faiss.read_index("./faiss-bert-pool-ivf_hnsw_quantizer.index")
                # 只读取句子
                all_sentence_list = read_sentence(sentence_path_list)
        elif faiss_idx_use_method == "IndexFlatIP":
            if os.path.exists("./faiss-bert-pool.index"):
                print("\t load index: faiss-bert-pool.index ......")
                faiss_index = faiss.read_index("./faiss-bert-pool.index")
                # 只读取句子
                all_sentence_list = read_sentence(sentence_path_list)
            else:
                all_sentence_vec, all_sentence_list = read_sentence_docid_vec(sentence_path_list=sentence_path_list,
                                                                              vec_path_list=vec_path_list, verbose=True)
                faiss_index = get_faiss_index(all_sentence_vec=all_sentence_vec, ncentroids=5, niter=200, verbose=True)
                # 保存索引
                faiss.write_index(faiss_index, "./faiss-bert-pool.index")
                print("\t save index: faiss-bert-pool.index .................")
    # 两者共有的内容：返回搜索的个数
    search_size = int(item.get("search_size", 5))

    if retrieve_method == "es":
        query_sentence_list = [title.strip() for title in input_title.split(";")]
        result_list = [[] for _ in range(len(query_sentence_list))]

        for idx, title in enumerate(query_sentence_list):

            es_result_title = es_search.es_search_title_main(title, size=search_size, new_good_word_dict=new_good_word_dict, es=es, jieba_stop=jieba_stop)
            print("title:", title)
            for data in es_result_title:
                # print(data["_source"]["title"])
                # print(all_title_and_docid_dict[data["_source"]["title"]])
                # print("\t es search result:", data["_source"]["title"])
                if all_title_and_docid_dict.get(data["_source"]["title"], None):
                    result_list[idx].append((data["_source"]["title"], all_title_and_docid_dict[data["_source"]["title"]], data["_score"]))
                elif all_title_and_docid_dict.get(data["_source"]["title"].upper(), None):
                    result_list[idx].append(
                        (data["_source"]["title"].upper(), all_title_and_docid_dict[data["_source"]["title"].upper()], data["_score"]))
                elif all_title_and_docid_dict.get(data["_source"]["title"].lower(), None):
                    result_list[idx].append(
                        (data["_source"]["title"].lower(), all_title_and_docid_dict[data["_source"]["title"].lower()],
                         data["_score"]))
                else:
                    result_list[idx].append(
                        (data["_source"]["title"], "", data["_score"]))

            # 计算输入标题和检索标题的余弦相似度
            if cal_cos_similarity_when_es:
                all_key_title_list = [t[0] for t in result_list[idx]]
                print(type(title))
                print(all_key_title_list)
                print(type(all_key_title_list))
                # print(all_key_title_list)  # similarity(input_title=None, retrieve_title=None)
                # query_sentence_list = [title.strip() for title in input_title.split(";")]
                #         query_sentence_vec = encode(query_sentence_list, model=model, get_sen_vector_method=get_vec_method, tokenizer=tokenizer)
                similarity_list = similarity(input_title=[title], retrieve_title=all_key_title_list, model=model, get_sen_vector_method="pool", tokenizer=tokenizer)

                title_similarity_dict = sim_former(title, all_key_title_list)
                for j, (sim, sim_form) in enumerate(zip(similarity_list, title_similarity_dict.values())):
                    result_list[idx][j] += (sim, sim_form)

        print("\t es retrieve one sentence cost time: ", time.time() -start_time)
        return sanic_json({"result": result_list})

    elif retrieve_method == "faiss":
        get_vec_method = item.get("get_vec_method", "pool")
        assert get_vec_method in ["pool", "cls"]

        model_name = item.get("model_name", "bert")
        assert model_name in ["bert", "roformer-sim"]

        global ncentroids, niter, verbose
        ncentroids = item.get("ncentroids", 50)
        niter = item.get("niter", 200)
        verbose = item.get("verbose", True)

        # 1.默认加载bert中的pool向量，否则根据需要进行添加;并加载相应的模型、tokenizer
        if get_vec_method != "pool" and model_name != "bert":
            if get_vec_method == "cls" and model_name == "bert":
                # 1.1 得到所有句子向量
                vec_path_list = "./bert_sentence_cls_vec-0_np.pkl;./bert_sentence_cls_vec-1_np.pkl;./bert_sentence_cls_vec-2_np.pkl;./bert_sentence_cls_vec-3_np.pkl".split(
                    ";")

                # 2. 将所有句子向量放到fiass中, 加载faiss索引
                # global faiss_index
                ########################################################
                if os.path.exists("./faiss-bert-cls.index"):
                    print("\t load faiss-bert-cls.index ..........")
                    faiss_index = faiss.read_index("./faiss-bert-cls.index")
                else:
                    all_sentence_vec, all_sentence_list = read_sentence_docid_vec(
                    sentence_path_list=[None for _ in range(len(vec_path_list))], vec_path_list=vec_path_list,
                    verbose=verbose)

                    # faiss_index = get_faiss_index(all_sentence_vec=all_sentence_vec, ncentroids=50, niter=200, verbose=True)
                    faiss_index = get_faiss_index(all_sentence_vec=all_sentence_vec, ncentroids=ncentroids, niter=niter,
                                                  verbose=verbose)
                    print("\t write index : ./faiss-bert-cls.index")
                    # 保存索引
                    faiss.write_index(faiss_index, "./faiss-bert-cls.index")
                ################
                # 1.2 加载模型
                tokenizer, model = get_finetune_model(model_name="bert")
            elif get_vec_method == "avg" and model_name == "roformer-sim":
                vec_path_list = "./roformer/roformer-sim_sentence_avg_vec-0_np.pkl;./roformer/roformer-sim_sentence_avg_vec-1_np.pkl;./roformer/roformer-sim_sentence_avg_vec-2_np.pkl;./roformer/roformer-sim_sentence_avg_vec-3_np.pkl".split(
                    ";")
                all_sentence_vec, _, _ = read_sentence_docid_vec(
                    sentence_path_list=[None for _ in range(len(vec_path_list))], docID_path_list=[None for _ in range(len(vec_path_list))], vec_path_list=vec_path_list,
                    verbose=verbose)

                tokenizer, model = get_finetune_model(model_name="roformer-sim")

                # 2. 将所有句子向量放到fiass中, 加载faiss索引
                faiss_index = get_faiss_index(all_sentence_vec=all_sentence_vec, ncentroids=ncentroids, niter=niter,
                                              verbose=verbose)
            elif get_vec_method == "cls" and model_name == "roformer-sim":
                vec_path_list = "./roformer/roformer-sim_sentence_cls_vec-0_np.pkl;./roformer/roformer-sim_sentence_cls_vec-1_np.pkl;./roformer/roformer-sim_sentence_cls_vec-2_np.pkl;./roformer/roformer-sim_sentence_cls_vec-3_np.pkl".split(
                    ";")
                all_sentence_vec, _, _ = read_sentence_docid_vec(
                    sentence_path_list=[None for _ in range(len(vec_path_list))], docID_path_list=[None for _ in range(len(vec_path_list))], vec_path_list=vec_path_list,
                    verbose=True)

                tokenizer, model = get_finetune_model(model_name="roformer-sim")

                # 2. 将所有句子向量放到fiass中, 加载faiss索引
                faiss_index = get_faiss_index(all_sentence_vec=all_sentence_vec, ncentroids=ncentroids, niter=niter,
                                              verbose=verbose)

        # 2. 得到输入句子的向量表示
        start_time = time.time()
        query_sentence_list = [title.strip() for title in input_title.split(";")]
        query_sentence_vec = encode(query_sentence_list, model=model, get_sen_vector_method=get_vec_method, tokenizer=tokenizer)

        print("\t get query_sentence_vec time:", time.time()-start_time)

        start_time = time.time()

        # 3. 利用faiss进行查找
        # faiss.normalize_L2(q_vec)
        q_vec = np.array(query_sentence_vec).astype('float32')
        D, I = faiss_index.search(q_vec, search_size)

        print("\t faiss retrieve time:", time.time()-start_time)
        get_result_start_time = time.time()
        result_list = [[] for _ in range(q_vec.shape[0])]

        print(D)
        print(q_vec.shape[0])
        for n in range(q_vec.shape[0]):
            for i, j in zip(I[n], D[n]):
                match_score = process.extract(query_sentence_list[n], [all_sentence_list[i]], limit=1)[0][-1]

                if use_sim_former:
                    roformer_sim_score = sim_former(query_sentence_list[n], [all_sentence_list[i]])[all_sentence_list[i]]
                else:
                    roformer_sim_score = 10000
                #
                # print("\t\t sen:{}, roformer_sim_score:{}, cos_by_string:{}, match_score:{}".format(all_sentence_list[i], roformer_sim_score, str(cos_by_string(query_sentence_list[n], all_sentence_list[i])), str(match_score)))

                result_list[n].append((all_sentence_list[i], all_title_and_docid_dict[all_sentence_list[i]], float(j), str(roformer_sim_score), str(cos_by_string(query_sentence_list[n], all_sentence_list[i])), str(match_score)))
        print("\t get one sentence result cost time: ", time.time() - get_result_start_time)
        return sanic_json({"result": result_list})

    elif retrieve_method == "both":
        get_vec_method = item.get("get_vec_method", "pool")
        assert get_vec_method in ["pool", "cls"]

        model_name = item.get("model_name", "bert")
        assert model_name in ["bert", "roformer-sim"]

        ncentroids = item.get("ncentroids", 50)
        niter = item.get("niter", 200)
        verbose = item.get("verbose", True)


        ######################################
        # 2. 得到输入句子的向量表示
        start_time = time.time()
        query_sentence_list = [title.strip() for title in input_title.split(";")]
        query_sentence_vec = encode(query_sentence_list, model=model, get_sen_vector_method=get_vec_method,
                                    tokenizer=tokenizer)

        print("\t get query_sentence_vec time:", time.time() - start_time)

        start_time = time.time()

        # 3. 利用faiss进行查找
        # faiss.normalize_L2(q_vec)
        q_vec = np.array(query_sentence_vec).astype('float32')
        D, I = faiss_index.search(q_vec, search_size)

        print("\t faiss retrieve time:", time.time() - start_time)
        get_result_start_time = time.time()
        result_list = [[] for _ in range(q_vec.shape[0])]

        print(D)
        print(q_vec.shape[0])
        for n in range(q_vec.shape[0]):
            for i, j in zip(I[n], D[n]):
                # 计算检索句子和输入句子的字符相似度

                if use_sim_former:
                    roformer_sim_score = sim_former(query_sentence_list[n], [all_sentence_list[i]])[all_sentence_list[i]]
                
                match_score = process.extract(query_sentence_list[n], [all_sentence_list[i]], limit=1)[0][-1]
                # 对检索回的句子进行垄断
                # 1.如果模糊分数低于20，则直接过滤；字符相似度低于30的直接过滤掉
                # sim_former(src=title, tgt=all_es_title)
                # roformer_sim_score = sim_former(query_sentence_list[n], [all_sentence_list[i]])[all_sentence_list[i]]
                #
                # print("\t\t sen:{}, roformer_sim_score:{}, cos_by_string:{}, match_score:{}".format(all_sentence_list[i], roformer_sim_score, str(cos_by_string(query_sentence_list[n], all_sentence_list[i])), str(match_score)))
                if use_sim_former:
                    if roformer_sim_score < 0.7:
                        print("all_sentence_list[i]: ", all_sentence_list[i])
                        print("dockid:", all_title_and_docid_dict[all_sentence_list[i]])
                        print("all_title_and_docid_dict", len(all_title_and_docid_dict))
                        filter_img = (all_sentence_list[i], all_title_and_docid_dict[all_sentence_list[i]], float(j),
                                  roformer_sim_score, str(cos_by_string(query_sentence_list[n], all_sentence_list[i])), str(match_score))
                        print("\t\t 通过roformer过滤句子的信息为", filter_img)
                        continue

                if match_score<20:
                    filter_img = (all_sentence_list[i], all_title_and_docid_dict[all_sentence_list[i]], float(j), str(cos_by_string(query_sentence_list[n], all_sentence_list[i])), str(match_score))
                    print("\t\t 通过匹配分数过滤的句子信息为：", filter_img)
                    continue

                # 在统计单词的时候，可以将停用词去掉试一试，如”和“、”的“、”呢“、”吗“、”啊“
                stopWords_list = ["的", "呢", "啊", ]
                input_title_word_list = [w for w in jieba.lcut(query_sentence_list[n]) if w not in stopWords_list]
                retrieve_title_word_list = jieba.lcut(all_sentence_list[i])
                num = 0
                for word in input_title_word_list:
                    if word in retrieve_title_word_list:
                        num += 1

                if num/len(input_title_word_list) < 1/3:
                    filter_img = (all_sentence_list[i], all_title_and_docid_dict[all_sentence_list[i]], float(j),
                                  str(cos_by_string(query_sentence_list[n], all_sentence_list[i])), str(match_score), str(num/len(input_title_word_list)))
                    print("\t\t 过滤的句子信息(输入句子中的有1/3的单词没有出现在检索句子中)为：", filter_img)
                    continue

                result_list[n].append((all_sentence_list[i], all_title_and_docid_dict[all_sentence_list[i]], float(j), str(cos_by_string(query_sentence_list[n], all_sentence_list[i])), str(match_score)))

            # 如果检索的个数低于5个，则使用es进行检索
            # 使用es检索时，我们首先对检索的句子进行过滤，过滤时加上
            if len(result_list[n]) < 5:
                all_faiss_title = [tmp[0] for tmp in result_list[n]]
                print("all_faiss_title:", all_faiss_title)

                # 召回80个，再对这80个进行排序，
                # (input_title=None, search_size=None, new_good_word_dict=None, es=None, jieba_stop=None)
                es_title_docid_esScore_list = es_search_and_filter(input_title=input_title, search_size=60, new_good_word_dict=new_good_word_dict, es=es, jieba_stop=jieba_stop, all_title_and_docid_dict=all_title_and_docid_dict, use_sim_former=use_sim_former)

                for es_title_docid_esScore in es_title_docid_esScore_list:
                    if es_title_docid_esScore[0] in all_faiss_title:
                        continue
                    # 统计当前保存的结果是否达到了5个，如果达到了5个，则直接返回
                    if len(result_list[n])>=5:
                        break

                    if isinstance(es_title_docid_esScore, tuple):
                        es_title_docid_esScore += ("es", )
                    elif isinstance(es_title_docid_esScore, list):
                        es_title_docid_esScore += ["es"]
                    result_list[n].append(es_title_docid_esScore)

        print("\t get one sentence result cost time: ", time.time() - get_result_start_time)
        return sanic_json({"result": result_list})

    elif retrieve_method == "fass_es":
        # 1.先通过es检索前n个相似句子
        query_sentence_list = [title.strip() for title in input_title.split(";")]
        result_list = [[] for _ in range(len(query_sentence_list))]

        final_retrieve_sentence_list = [[] for _ in range(len(query_sentence_list))]
        remain_sentence_list = [[] for _ in range(len(query_sentence_list))]

        for idx, title in enumerate(query_sentence_list):
            es_result_title = es_search.es_search_title_main(title, size=search_size,
                                                             new_good_word_dict=new_good_word_dict, es=es,
                                                             jieba_stop=jieba_stop)
            print("title:", title)
            for data in es_result_title:
                if all_title_and_docid_dict.get(data["_source"]["title"], None):
                    result_list[idx].append(
                        (data["_source"]["title"], all_title_and_docid_dict[data["_source"]["title"]], data["_score"]))
                elif all_title_and_docid_dict.get(data["_source"]["title"].upper(), None):
                    result_list[idx].append(
                        (data["_source"]["title"].upper(), all_title_and_docid_dict[data["_source"]["title"].upper()],
                         data["_score"]))
                elif all_title_and_docid_dict.get(data["_source"]["title"].lower(), None):
                    result_list[idx].append(
                        (data["_source"]["title"].lower(), all_title_and_docid_dict[data["_source"]["title"].lower()],
                         data["_score"]))
                else:
                    result_list[idx].append(
                        (data["_source"]["title"], "", data["_score"]))

            # 对所有的es分数使用最大最小进行归一化或softmax归一化
            all_es_title = [tmp[0] for tmp in result_list[idx]]
            all_es_score = [tmp[-1] for tmp in result_list[idx]]
            softmax_score_for_es = softmax(np.array([all_es_score]))[0]

            # 对es检索出的句子再使用simformer计算相似度
            input_es_similarity_dict = sim_former(src=title, tgt=all_es_title)
            all_roformer_score = list(input_es_similarity_dict.values())
            softmax_score_for_roformer = softmax(np.array([all_roformer_score]))[0]

            # 对es检索出的的句子再使用字符串余弦距离计算相似度，并进行归一化
            cos_by_string_score = [cos_by_string(title, tmp) for tmp in all_es_title]
            softmax_score_for_cosString = softmax(np.array([cos_by_string_score]))[0]

            # 对es检索出的句子再使用模糊匹配计算相似度，并进行归一化
            fuzzy_match_score_sort_list = process.extract(title, all_es_title, limit=len(all_es_title))
            fuzzy_match_score_dict = {}
            for title_score_tuple in fuzzy_match_score_sort_list:
                fuzzy_match_score_dict[title_score_tuple[0]] = title_score_tuple[1]
            fuzzy_match_score_list = [fuzzy_match_score_dict[tmp] for tmp in all_es_title]
            softmax_score_for_fuzzyMatch = softmax(np.array([fuzzy_match_score_list]))[0]

            # 将四个的分数进行结合
            total_score = []
            for es_score, roformer_score, cosString_score, fuzzyMath_score in zip(softmax_score_for_es, softmax_score_for_roformer, softmax_score_for_cosString, softmax_score_for_fuzzyMatch):
                total_score.append(1/3*es_score + 2/4*roformer_score + 1/6*cosString_score + 1/6*fuzzyMath_score)

            for tmp, s in zip(all_es_title, total_score):
                print("tmp:{}; score:{}".format(tmp, s))

            # 得到top-5的标题、分数
            top_5_idx_list = np.array(total_score).argsort()[::-1][:5].tolist()
            top_5_title_docid_esScore = [result_list[idx][top_5_idx] for top_5_idx in top_5_idx_list]

            top_5_title_list = [all_es_title[idx] for idx in top_5_idx_list]
            top_5_softmax_score_for_es = [softmax_score_for_es[idx] for idx in top_5_idx_list]
            top_5_softmax_score_for_roformer = [softmax_score_for_roformer[idx] for idx in top_5_idx_list]
            top_5_softmax_score_for_cosString = [softmax_score_for_cosString[idx] for idx in top_5_idx_list]
            top_5_softmax_score_for_fuzzyMatch = [softmax_score_for_fuzzyMatch[idx] for idx in top_5_idx_list]

            top_5_all_es_score = [all_es_score[idx] for idx in top_5_idx_list]
            top_5_all_roformer_score = [all_roformer_score[idx] for idx in top_5_idx_list]
            top_5_cos_by_string_score = [cos_by_string_score[idx] for idx in top_5_idx_list]
            top_5_fuzzy_match_score_list = [fuzzy_match_score_list[idx] for idx in top_5_idx_list]
            # 再次
            if all([True if 0.85<=tmp else False for tmp in top_5_all_roformer_score]):
                final_retrieve_sentence_list[idx].extend(top_5_title_docid_esScore)
            elif sum([True if 0.75<=tmp else False for tmp in top_5_all_roformer_score]) > len(top_5_all_roformer_score)*3/4:
                select_idx_list = np.array(top_5_all_roformer_score).argsort()[::-1][:2]
                for select_idx in select_idx_list:
                    final_retrieve_sentence_list[idx].append(top_5_title_docid_esScore[select_idx])
            elif sum([True if 0.65<tmp<0.80 else False for tmp in top_5_all_roformer_score]) > len(top_5_all_roformer_score)*3/4:
                select_idx = np.array(top_5_all_roformer_score).argmax()
                final_retrieve_sentence_list[idx].append(top_5_title_docid_esScore[select_idx])

            # 3. 统计最终句子列表中有多个合适的句子，如果合适的句子数量不超过5个，则利用bert，从剩下的句子中去寻找
            if len(final_retrieve_sentence_list[idx]) >= 5:
                # 3.1 统计最终句子列表中有多个合适的句子，如果合适的句子数量超过5个，则直接返回
                final_retrieve_sentence_list[idx] = final_retrieve_sentence_list[idx][:5]
                # 可以使用bert进行排序
            else:
                need_add_num = 5-len(final_retrieve_sentence_list[idx])
                # print("remain_sentence_list[idx]:", remain_sentence_list[idx])
                remain_sentence_list[idx] = result_list[idx]
                remain_sentence_title_list = [tmp[0] for tmp in remain_sentence_list[idx]]
                remain_sentence_title_list = [tmp[0] for tmp in result_list[idx]]
                # 3.2 答案不足5个，使用bert计算相似度
                similarity_list = similarity(input_title=[title], retrieve_title=remain_sentence_title_list, model=model, get_sen_vector_method="pool", tokenizer=tokenizer)

                # title_similarity_dict = sim_former(title, remain_sentence_title_list)
                # similarity_list_simFormer = title_similarity_dict.values()
                # similarity_list_simFormer = list(similarity_list_simFormer)
                # print("similarity_list_simFormer", similarity_list_simFormer)
                # print("similarity_list_simFormer type:", type(similarity_list_simFormer))

                idx_large2small_list = np.argsort(np.array(similarity_list))[::-1][: need_add_num].tolist()
                # idx_large2small_list_simFormer = np.argsort(np.array(similarity_list_simFormer))[::-1][: need_add_num].tolist()
                # 3.3 根据排序，从剩下的句子中进行选择
                remain_se = [remain_sentence_list[idx][idx_large2small] + (similarity_list[idx_large2small],) for idx_large2small in idx_large2small_list]

                # remain_se_simFormer = [remain_sentence_list[idx][idx_large2small] + (similarity_list[idx_large2small], similarity_list_simFormer[idx_large2small]) for
                             # idx_large2small in idx_large2small_list_simFormer]

                # 3.4 保留相似度高于0.94的句子
                for tmp in remain_se:
                    if tmp[-1]>=0.94:
                        final_retrieve_sentence_list[idx].append(tmp)

                need_add_num = search_size - len(final_retrieve_sentence_list[idx])
                if need_add_num != 0:
                    # 4. 使用faiss进行检索
                    # global get_vec_method, model_name
                    get_vec_method = item.get("get_vec_method", "pool")
                    assert get_vec_method in ["pool", "cls"]

                    model_name = item.get("model_name", "bert")
                    assert model_name in ["bert", "roformer-sim"]

                    # global ncentroids, niter, verbose
                    ncentroids = item.get("ncentroids", 50)
                    niter = item.get("niter", 200)
                    verbose = item.get("verbose", True)

                    # 4.1 得到输入句子的向量表示
                    query_sentence_list = [title]
                    query_sentence_vec = encode(query_sentence_list, model=model, get_sen_vector_method=get_vec_method,
                                                tokenizer=tokenizer)

                    # print("\t get query_sentence_vec time:", time.time() - start_time)
                    # start_time = time.time()

                    # 3. 利用faiss进行查找
                    # faiss.normalize_L2(q_vec)
                    q_vec = np.array(query_sentence_vec).astype('float32')
                    D, I = faiss_index.search(q_vec, 5)

                    # print("\t faiss retrieve time:", time.time() - start_time)
                    # get_result_start_time = time.time()
                    result_list = [[] for _ in range(q_vec.shape[0])]
                    for n in range(q_vec.shape[0]):
                        for i, j in zip(I[n], D[n]):
                            result_list[n].append(
                                (all_sentence_list[i], all_title_and_docid_dict.get(all_sentence_list[i], ""), float(j), "faiss"))
                    result_list = result_list[0]

                    # print("\t get one sentence result cost time: ", time.time() - get_result_start_time)
                    # return sanic_json({"result": result_list})
                    # print("result_list[0]", result_list[0])
                    # print("final_retrieve_sentence_list[idx]:", final_retrieve_sentence_list[idx])
                    # print("final_retrieve_sentence_list[idx][0]:", final_retrieve_sentence_list[idx][0])
                    exist_title_list = []

                    for tmp in final_retrieve_sentence_list[idx]:
                        try:
                            exist_title_list.append(tmp[0])
                        except:
                            print("tmp:", tmp)
                            raise Exception("error ...........")

                    # exist_title_list = [tmp[0] for tmp in final_retrieve_sentence_list[idx][0]]
                    for tmp in result_list:
                        # print("tmp:", tmp)
                        # print("exist_title_list:", exist_title_list)
                        if tmp[0].lower() in exist_title_list:
                            continue
                        else:
                            final_retrieve_sentence_list[idx].append(tmp)

                        if len(final_retrieve_sentence_list[idx]) == 5:
                            break

        return sanic_json({"result": final_retrieve_sentence_list})



@app.route("/cal_sentence_similarity", methods=['POST'])
def cal_similarity(request):
    item = request.json
    input_title = item["input_title"]
    retrieve_title = item["retrieve_title"]

    query_vecs = encode(input_title, normalize_to_unit=True)
    retrieve_title = eval(retrieve_title)

    if isinstance(retrieve_title, str):
        all_key_title_list = retrieve_title.split("|||")
    elif isinstance(retrieve_title, list):
        all_key_title_list = retrieve_title

    # print("\t all similarity title: ", all_key_title_list)

    key_vecs = encode(all_key_title_list, batch_size=100,
                      normalize_to_unit=True)

    single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1

    if single_query:
        query_vecs = query_vecs.unsqueeze(0)
    if single_key:
        if isinstance(key_vecs, np.ndarray):
            key_vecs = key_vecs.reshape(1, -1)
        else:
            key_vecs = key_vecs.unsqueeze(0)

    # similarity_list = (query_vecs @ key_vecs.T)[0].tolist()
    similarity_list = torch.cosine_similarity(query_vecs, key_vecs, dim=-1).tolist()
    return sanic_json({"result": similarity_list})




if __name__ == '__main__':
    """
    # 如果你能克服“刺痛因素”，你有一个很容易使用的有机氮肥来源近在咫尺。
    # 伊朗称已捕获无人机，美国否认失去一架
    # 美军在阿富汗南部遭到枪击后，打死了近二十名塔利班武装分子嫌疑人，这是一系列此类袭击中的最新一次。
    # 滑板运动员从楼梯上跳下来。
    input_title = "如果你能克服“刺痛因素”，你有一个很容易使用的有机氮肥来源近在咫尺。;伊朗称已捕获无人机，美国否认失去一架; 美军在阿富汗南部遭到枪击后，打死了近二十名塔利班武装分子嫌疑人，这是一系列此类袭击中的最新一次。; 滑板运动员从楼梯上跳下来。"
    retrieve_similarity_sentence_({"input_title":input_title})
    """
    app.run(host="0.0.0.0", port=50515)
