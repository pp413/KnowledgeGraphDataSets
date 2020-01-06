# KnowledgeGraphDataSets
The benchmark data sets of Knowledge Graphs.

kinship; nations; umls; WN18RR; YAGO3-10

ConvE

FB15k; FB15k-237; WN18

dgl

## Example

### TransE

```python
import torch
from knowledgegraphs import KGDataSet, Evaluation

if __name__ ==  "__main__":

    # load the data
    data = KGDataSet('nations')
    
    # training set
    train_data = data.train
    
    # evaluation
    eva = Evaluation(*data())
    
    class TransE(torch.nn.Module):
        def __init__(self, num_e, num_r, embed_size):
            super(TransE, self).__init__()
            
            self.e = torch.nn.Embedding(num_e, embed_size)
            self.r = torch.nn.Embedding(num_r, embed_size)
            self.init()
            
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
        
        def init(self):
            torch.nn.init.xavier_uniform_(self.e.weight)
            torch.nn.init.xavier_uniform_(self.r.weight)
        
        def forward(self, triplet):
            triplet = triplet.to(self.device)
            h = self.e(triplet[:, 0]).squeeze()
            r = self.r(triplet[:, 1]).squeeze()
            
            return torch.mean(torch.abs(h + r - self.e.weight.data), dim=1)
            

        def predict(self, triplet):
            
            triplet = triplet.to(self.device).squeeze()
            # print(triplet)
            
            h = self.e(triplet[:, 0])
            r = self.r(triplet[:, 1])
            t = self.e(triplet[:, 2])
            
            return torch.mean(torch.abs(h + r - t), dim=1)
    
    model = TransE(data.num_nodes, data.num_rels, 100)
    print(model)
    
    eva.valid(model)
    eva.test(model)
```

The results:

```shell
The total of entities: 14
The total of relations: 55
TransE(
  (e): Embedding(14, 100)
  (r): Embedding(110, 100)
)
100%|███████████████████████████████████████████████████████████████████████| 155/155 [00:00<00:00, 5126.38it/s]
rhs MR: 4.42, MRR: 38.39%, Hits@[1, 3, 10]: [16.58%, 46.73%, 94.47%]
100%|███████████████████████████████████████████████████████████████████████| 145/145 [00:00<00:00, 4926.48it/s]
lhs MR: 3.91, MRR: 45.30%, Hits@[1, 3, 10]: [24.62%, 55.78%, 95.98%]
100%|███████████████████████████████████████████████████████████████████████| 143/143 [00:00<00:00, 4412.85it/s]
rhs MR: 4.64, MRR: 37.29%, Hits@[1, 3, 10]: [17.41%, 42.79%, 94.53%]
100%|███████████████████████████████████████████████████████████████████████| 145/145 [00:00<00:00, 4781.96it/s]
lhs MR: 3.96, MRR: 47.81%, Hits@[1, 3, 10]: [29.35%, 54.23%, 95.02%]
```

