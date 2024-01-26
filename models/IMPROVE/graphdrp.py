# how to get GraphDRP here....

from improve import framework as frm
import torch
import model_utils.models.ginconv as ginconv
import torch_geometric

x=torch.zeros((70,78), dtype=torch.float32)
edge_index=torch.randint(low=64,high=70,size=(2,150), dtype=torch.int64)
batch=torch.ones((70), dtype=torch.int64)
target=torch.zeros((2,958), dtype=torch.float32)

#loader = torch_geometric.loader.DataLoader([torch_geometric.data.Data(x=x, edge_index=edge_index, batch=batch, target=target)])
#for data in loader:
#    pass

torch_geometric.data.batch.Batch.from_data_list([torch_geometric.data.Data(x=x, edge_index=edge_index, batch=batch, target=target)])

model = ginconv.GroqGINConvNet()
model.eval()

model(x, edge_index, batch, target)