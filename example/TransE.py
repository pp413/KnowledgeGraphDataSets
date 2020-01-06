import torch
from knowledgegraphs import KGDataSet, Evaluation

if __name__ ==  "__main__":

    data = KGDataSet('nations')
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
    print("predict" in dir(model))
    
    eva.valid(model)
    eva.test(model)