#!/usr/bin/env python
# coding=utf-8
import hashlib
import os
import tarfile
import warnings
import zipfile

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import requests
except ImportError:
    class requests_failed_to_import(object):
        pass
    requests = requests_failed_to_import
    
__all__ = ["down_url", "datasets_name", "set_root_dir",
           "check_sha1", "down", "extract_archive",
           "_read_dictionary", "_read_triplets",
           "_read_triplets_as_list", "KGDataSet"]

down_url = "https://raw.githubusercontent.com/YaoShuang-long/KnowledgeGraphDataSets/master/datasets/"

datasets_name = ['FB15k', 'FB15k-237', 'WN18', 'WN18RR', 'YAGO3-10', 'kinship', 'nations', 'umls']

def set_root_dir():
    root = os.path.join(os.path.expanduser('~'), '.KnowledgeGraphDataSets')
    dirname = os.environ.get('KG_DIR', root)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Codes borrowed from mxnet/gluon/utils.py

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash

def download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True, log=True):
    """Download a given URL.

    Codes borrowed from mxnet/gluon/utils.py

    Parameters
    ----------
    url : str
        URL to download.
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with the same name as in url.
    overwrite : bool, optional
        Whether to overwrite the destination file if it already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt downloading in case of failure or non 200 return codes.
    verify_ssl : bool, default True
        Verify SSL certificates.
    log : bool, default True
        Whether to print the progress for download

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries+1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print('Downloading %s from %s...' % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not check_sha1(fname, sha1_hash):
                    raise UserWarning('File {} is downloaded but the content hash does not match.'
                                      ' The repo may be outdated or download may be incomplete. '
                                      'If the "repo_url" is overridden, consider switching to '
                                      'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        print("download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))

    return fname

def extract_archive(file, target_dir):
    """Extract archive file.

    Parameters
    ----------
    file : str
        Absolute path of the archive file.
    target_dir : str
        Target directory of the archive to be uncompressed.
    """
    if os.path.exists(target_dir):
        return
    if file.endswith('.gz') or file.endswith('.tar') or file.endswith('.tgz'):
        archive = tarfile.open(file, 'r')
    elif file.endswith('.zip'):
        archive = zipfile.ZipFile(file, 'r')
    else:
        raise Exception('Unrecognized file type: ' + file)
    print('Extracting file to {}'.format(target_dir))
    archive.extractall(path=target_dir)
    archive.close()

def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l


class KGDataSet(object):
    
    __solt__ = ['name', 'dir', 'train', 'valid', 'test', 'num_nodes', 'num_rels',
                'add_reverse']
    
    def __init__(self, name='FB15k-237', add_reverse=True):
        self.name = name
        self.dir = set_root_dir()
        url = down_url + '{}.tar.gz'.format(self.name)
        tgz_path = download(url, self.dir)
        self.dir = os.path.join(self.dir, self.name)
        extract_archive(tgz_path, self.dir)
        
        self.add_reverse = add_reverse
        self.load()
    
    def __call__(self):
        return self.train, self.valid, self.test, self.num_nodes, self.num_rels, self.add_reverse
    
    def load(self):
        
        def add_reverse(data, num_rels):
            src, rel, dst = data.transpose()
            rel = np.concatenate((rel, rel+num_rels))
            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            data = np.stack((src, rel, dst)).transpose()
            return data
        
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        
        entity_path = os.path.join(self.dir, 'entities.dict')
        relation_path = os.path.join(self.dir, 'relations.dict')
        
        if os.path.exists(entity_path) and os.path.exists(relation_path):
            entity_dict = _read_dictionary(entity_path)
            relation_dict = _read_dictionary(relation_path)
            self.train = np.array(_read_triplets_as_list(train_path, entity_dict, relation_dict))
            self.valid = np.array(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
            self.test = np.array(_read_triplets_as_list(test_path, entity_dict, relation_dict))
            self.num_nodes = len(entity_dict)
            self.num_rels = len(relation_dict)
            if self.add_reverse:
                self.train = add_reverse(self.train, self.num_rels)
                self.num_rels *= 2
                
        else:
            entity_dict, relation_dict = self.__read(train_path, valid_path, test_path)
            entity_dict.to_csv(entity_path, header=False, sep='\t')
            relation_dict.to_csv(relation_path, header=False, sep='\t')
    
    def __read(self, train_path, valid_path, test_path):
        train = pd.read_csv(train_path, header=None, sep='\t', names=['h', 'r', 't'])
        valid = pd.read_csv(valid_path, header=None, sep='\t', names=['h', 'r', 't'])
        test = pd.read_csv(test_path, header=None, sep='\t', names=['h', 'r', 't'])
        
        rel_list = pd.unique(pd.concat([train.r, valid.r, test.r]))
        rel_list = pd.Series(rel_list, index=np.arange(rel_list.shape[0]))
        
        ent_list = pd.unique(pd.concat([train.h, train.t, valid.h, valid.t, test.h, test.t]))
        ent_list = pd.Series(ent_list, index=np.arange(ent_list.shape[0]))
        
        rel_id = pd.Series(rel_list.index, index=rel_list.values)
        ent_id = pd.Series(ent_list.index, index=ent_list.values)
        
        self.num_nodes = ent_id.shape[0]
        self.num_rels = rel_id.shape[0]
        
        def merge_id(data, e_id, r_id):
            data['h_id'] = e_id[data.h].values
            data['r_id'] = r_id[data.r].values
            data['t_id'] = e_id[data.t].values
        
        def add_reverse(train):
            def add_reverse_for_data(data):
                reversed_data = data.rename(columns={'h_id': 't_id', 't_id': 'h_id'})
                reversed_data.r_id += self.num_rels
                data = pd.concat(([data, reversed_data]), ignore_index=True, sort=False)
                return data
            train = add_reverse_for_data(train)
            self.num_rels *= 2
        
        def reindex_kb(train, valid, test, ent_id, rel_id):
            eids = pd.concat([train.h_id, train.t_id], ignore_index=True)
            
            tv_eids = np.unique(pd.concat([test.h_id, test.t_id, valid.h_id, valid.t_id]))
            not_train_eids = tv_eids[~np.in1d(tv_eids, eids)]
            
            rids = pd.concat([train.r_id,], ignore_index=True)
            
            def gen_map(eids, rids):
                e_num = eids.groupby(eids.values).size().sort_values()[::-1]
                not_train = pd.Series(np.zeros_like(not_train_eids), index=not_train_eids)
                e_num = pd.concat([e_num, not_train])
                
                r_num = rids.groupby(rids.values).size().sort_values()[::-1]
                e_map = pd.Series(range(e_num.shape[0]), index=e_num.index)
                r_map = pd.Series(range(r_num.shape[0]), index=r_num.index)
                return e_map, r_map
            
            def remap_kb(kb, e_map, r_map):
                kb.loc[:, 'h_id'] = e_map.loc[kb.h_id.values].values
                kb.loc[:, 'r_id'] = r_map.loc[kb.r_id.values].values
                kb.loc[:, 't_id'] = e_map.loc[kb.t_id.values].values
                kb = kb[['h_id', 'r_id', 't_id']]
                return kb
            
            def remap_id(s, rm):
                data = rm.loc[s.values].values
                data = pd.Series(s.index, index=data).reindex(np.arange(len(s.index)))
                return data
            
            e_map, r_map = gen_map(eids, rids)
            
            self.train = remap_kb(train, e_map, r_map).values
            self.valid = remap_kb(valid, e_map, r_map).values
            self.test = remap_kb(test, e_map, r_map).values
            
            ent_id = remap_id(ent_id, e_map)
            rel_id = remap_id(rel_id, r_map)
            
            return ent_id, rel_id
        
        merge_id(train, ent_id, rel_id)
        merge_id(valid, ent_id, rel_id)
        merge_id(test, ent_id, rel_id)
        
        if self.add_reverse:
            add_reverse(train)
        return reindex_kb(train, valid, test, ent_id, rel_id)

#############################################################################################

#                            Evaluation

#############################################################################################
def avg_both(mr, mrrs, hits):
    n = (mr['lhs'] + mr['rhs']) / 2.
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MR': n, 'MRR': m, 'Hits@[1, 3, 10]': list(h)}


class Evaluation:
    
    __solt__ = ['num_nodes', 'num_rels', 'data', 'add_reverse_data']
    
    def __init__(self, train, valid, test, num_nodes, num_rels, add_reverse=True):
        
        self.num_nodes = num_nodes
        self.num_rels = num_rels // 2
        self.data = self.__get_filter(train, valid, test)
        
        if add_reverse:
            valid_src, valid_rel, valid_dst = valid.transpose()
            test_src, test_rel, test_dst = test.transpose()
            add_reverse_valid = np.stack((valid_dst, valid_rel+self.num_rels, valid_src)).transpose()
            add_reverse_test = np.stack((test_dst, test_rel+self.num_rels, test_src)).transpose()
            self.add_reverse_data = self.__get_filter(train, add_reverse_valid, add_reverse_test)
    
        print('The total of entities: {}'.format(self.num_nodes))
        print('The total of relations: {}'.format(self.num_rels))
    
    @ staticmethod
    def __get_filter(train, valid, test):
        valid_src, valid_rel, _ = valid.transpose()
        test_src, test_rel, _ = test.transpose()
        
        valid_pairs = set(zip(valid_src, valid_rel))
        test_pairs = set(zip(test_src, test_rel))
        
        data = {'train': train, 'valid': valid, 'test': test}
        
        valid_filter = {}
        valid_filter_set = set()
        
        test_filter = {}
        test_filter_set = set()
        
        valid = {}
        valid_set = set()
        test = {}
        test_set = set()
        
        for d in data.keys():
            for triplet in data[d]:
                pair = (triplet[0], triplet[1])
                
                if not d == 'valid':
                    if pair in valid_pairs:
                        if pair in valid_filter_set:
                            valid_filter[pair].append(triplet[2])
                        else:
                            valid_filter[pair] = [triplet[2]]
                            valid_filter_set.add(pair)
                else:
                    if pair in valid_set:
                        valid[pair].append(triplet[2])
                    else:
                        valid[pair] = [triplet[2]]
                        valid_set.add(pair)
                
                if not d == 'test':
                    if pair in test_pairs:
                        if pair in test_filter_set:
                            test_filter[pair].append(triplet[2])
                        else:
                            test_filter[pair] = [triplet[2]]
                            test_filter_set.add(pair)
                else:
                    if pair in test_set:
                        test[pair].append(triplet[2])
                    else:
                        test[pair] = [triplet[2]]
                        test_set.add(pair)
        
        valid_data = {
            'data_dict': valid,
            'pairs': np.array([[x[0], x[1]] for x in valid_set]),
            'filter': valid_filter,
            'filter_pairs': valid_filter_set & valid_set
        }
        
        test_data = {
            'data_dict': test,
            'pairs': np.array([[x[0], x[1]] for x in test_set]),
            'filter': test_filter,
            'filter_pairs': test_filter_set & test_set
        }        
        return valid_data, test_data
  
    @ staticmethod
    def __sample(data, num_nodes, input_triplet=True):
        data_dict = data['data_dict']
        pairs = data['pairs']
        
        def get_rank(score, targets):
            targets = set(targets)
            ranks = np.argsort(score * (-1))    # from big to small
            ranks = [i for i, x in enumerate(ranks) if x in targets]
            ranks = np.sort(ranks)
            ranks = [x - i for i, x in enumerate(ranks)]
            return np.array(ranks) + 1
        
        def produce(consumer):
            consumer.send(None)
            franks = np.array([])
            if input_triplet:
                for pair in tqdm(pairs):
                    _pair = pair.reshape(1, 2)
                    _pair = np.tile(_pair, (num_nodes, 1))
                    tail = np.arange(num_nodes).reshape(num_nodes, 1)
                    triplet = np.concatenate((_pair, tail), 1)
                    score = consumer.send(triplet)
                    
                    # filter
                    if (pair[0], pair[1]) in data['filter_pairs']:
                        tmp = np.min(score)
                        score[data['filter'][(pair[0], pair[1])]] = tmp
                    
                    franks = np.append(franks, get_rank(score, data_dict[(pair[0], pair[1])]))
            else:
                for pair in tqdm(pairs):
                    pair = pair.reshape(1, 2)
                    score = consumer.send(pair)
                    
                    # filter
                    if (pair[0], pair[1]) in data['filter_pairs']:
                        tmp = np.min(score)
                        score[data['filter'][(pair[0], pair[1])]] = tmp
                    
                    franks = np.append(franks, get_rank(score, data_dict[(pair[0], pair[1])]))
            
            consumer.close()
            return franks

        return produce
    
    @ staticmethod            
    def __consumer(predict):
        pred_scores = None
        with torch.no_grad():
            while True:
                data = yield pred_scores
                if data is None:
                    return
                data = torch.from_numpy(data)
                pred_scores = predict(data).cpu().numpy()
    
    def __eval(self, model, flags='valid', missing_eval='both', at=[1, 3, 10], input_triplet=True):
        model.eval()
        pred_function = model.predict if "predict" in dir(model) else model
        
        l = {'valid': 0, 'test': 1}[flags]
        
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']
        
        mean_ranks = {}
        mean_reciprocal_rank = {}
        hits_at = {}
        
        for m in missing:
            if m == 'rhs':
                produce = self.__sample(self.data[l], self.num_nodes, input_triplet=input_triplet)
                ranks = produce(self.__consumer(pred_function))
            else:
                produce = self.__sample(self.add_reverse_data[l], self.num_nodes, input_triplet=input_triplet)
                ranks = produce(self.__consumer(pred_function))
            
            mean_ranks[m] = np.mean(ranks)
            mean_reciprocal_rank[m] = np.mean(1. / ranks) * 100
            hits_at[m] = np.asarray(list(map(
                lambda x: np.mean((ranks <= x) * 100), at
            )))
            print(m + ' MR: {:.2f}, MRR: {:.2f}%, Hits@[1, 3, 10]: [{:.2f}%, {:.2f}%, {:.2f}%]'.format(
                mean_ranks[m], mean_reciprocal_rank[m], hits_at[m][0], hits_at[m][1], hits_at[m][2]))

        return avg_both(mean_ranks, mean_reciprocal_rank, hits_at)
    
    def valid(self, model, missing_eval='both', at=[1, 3, 10], input_triplet=True):
        return self.__eval(model, flags='valid', missing_eval=missing_eval, at=at, input_triplet=input_triplet)
    
    def test(self, model, missing_eval='both', at=[1, 3, 10], input_triplet=True):
        return self.__eval(model, flags='test', missing_eval=missing_eval, at=at, input_triplet=input_triplet)


if __name__ ==  "__main__":

    data = KGDataSet('nations')
    eva = Evaluation(*data())
    
    class TransE(torch.nn.Module):
        def __init__(self, num_e, num_r, embed_size):
            super(TransE, self).__init__()
            
            self.e = torch.nn.Embedding(num_e, embed_size)
            self.r = torch.nn.Embedding(num_r, embed_size)
            self.init()
            
            # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.device = 'cpu'
            self.to(self.device)
        
        def init(self):
            torch.nn.init.xavier_uniform_(self.e.weight)
            torch.nn.init.xavier_uniform_(self.r.weight)
        
        def forward(self, triplet):
            triplet = triplet.to(self.device)
            h = self.e(triplet[:, 0]).squeeze()
            r = self.r(triplet[:, 1]).squeeze()
            
            return torch.mean(torch.abs(h + r - self.e.weight.data), dim=1)
            

        # def predict(self, triplet):
            
        #     triplet = triplet.to(self.device).squeeze()
        #     # print(triplet)
            
        #     h = self.e(triplet[:, 0])
        #     r = self.r(triplet[:, 1])
        #     t = self.e(triplet[:, 2])
            
        #     return torch.mean(torch.abs(h + r - t), dim=1)
    
    model = TransE(data.num_nodes, data.num_rels, 100)
    print(model)
    print("predict" in dir(model))
    
    # print(eva.data[0]['filter'])
    
    eva.valid(model)
    eva.test(model)
