
# Import Libraries
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, download_url, Data
import numpy as np 
import os
from tqdm import tqdm
import os.path as osp
import geopandas as gpd 
import adjacency


class OpioidDataset(Dataset):
    def __init__(self, root,test=False, transform=None, pre_transform=None, pre_filter=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        # self.filename = filename
        super(OpioidDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return ['annotation_NOD_without_missing_county.csv']

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0],dtype={'STCNTY': str}).reset_index()
        self.county = self.data['STCNTY'].tolist()

        if self.test:
            return [f'data_test_{i}.pt' for i in self.county]
        else:
            return [f'data_{i}.pt' for i in self.county]

    def download(self):
        pass

    def get_folders(self,location):
        """Get list of folders in a directory."""
        return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]
    
    def flatten_and_unique(self,nested_list):
        # Flatten the nested list using a list comprehension
        flat_list = [item for sublist in nested_list for item in sublist]
        # Use a set to get unique elements and convert it back to a list
        unique_list = list(set(flat_list))
        return unique_list
    
    def flatten(self,nested_list):
        # Flatten the nested list using a list comprehension
        flat_list = [item for sublist in nested_list for item in sublist]
        return flat_list
    
    def _get_node_features(self, state_data,var_names, node_ids):
        """ This will return a matrix / 2d array of the shape
        [Number of Nodes, Node feature size]

        In here, node_ids are the gnn_ids of the nodes that we want to extract the features from.
        """

        # Filter the dataframe to include only the nodes by gnn_id
        filtered_graph_df = state_data[state_data['gnn_id'].isin(node_ids)]

        # Make index same as gnn_id
        filtered_graph_df.set_index('gnn_id', inplace=True)

        # Node features
        attributes = filtered_graph_df[var_names]

        return torch.tensor(attributes.to_numpy(), dtype=torch.float)





    def process(self):
        # print("This should not be running")
        # pass

        DATA_DIR = "/home/h6x/git_projects/data_processing/processed_data/SVI/SVI2018_MIN_MAX_SCALED_MISSING_REMOVED_GNN_ID"
        VARIABLE = 'EP_POV'
        NODE_FEATURES = [
        'EP_POV', 'EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
        'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
        ]
        states = self.get_folders(DATA_DIR)
        # print(states)

        annotaions = pd.read_csv(self.raw_paths[0], dtype={'STCNTY': str})
        # print(annotaions)

        # print(f'Annotation length: ',len(annotaions))


        data_list = []
        ids_list = []

        for state in tqdm(states, desc="Processing states"):
            # print(f'Processing {state}')
            state_path = os.path.join(DATA_DIR, state)
            state_data = gpd.read_file(state_path)

            # Get the list of counties
            counties = state_data['STCNTY'].unique().tolist()

            # add each county STCNTY to the ids_list
            #might not need this
            ids_list.append(counties)
            ids_list = self.flatten(ids_list)

            # print(f'ids_list: ',ids_list)

            for county in counties:
                # print(f'Processing {county}')
                graph_data = adjacency.process_county(state, county, VARIABLE, DATA_DIR)

                simplices = graph_data['simplices']
                dataframe = graph_data['dataframe']


                # Edge list
                edges = []
                for i in range(len(simplices)):
                    
                    if len(simplices[i]) == 2:
                        edge = []
                        for j in range(len(simplices[i])):
                            edge.append(dataframe[dataframe['sortedID']==simplices[i][j]]['gnn_id'].values[0])
                        edges.append(edge)
                        edges.append(edge[::-1])

                # Node features
                node_ids  = self.flatten_and_unique(edges)

                # # print(f'Node IDs: ',node_ids)

                # # Filter the dataframe to include only the nodes by gnn_id
                # filtered_graph_df = state_data[state_data['gnn_id'].isin(node_ids)]

                # # Make index same as gnn_id
                # filtered_graph_df.set_index('gnn_id', inplace=True)

                # # print(f'Filtered Graph DF: ',filtered_graph_df)

                # # Node features
                # attributes = filtered_graph_df[NODE_FEATURES]

                attrs = self._get_node_features(state_data,NODE_FEATURES,node_ids)

                # print(f'Node Features: ',node_features)

                # Graph label
                label = annotaions[annotaions['STCNTY'] == county]['percen_US'].values

                # print(f'Label: ',label)


                # # Convert the Dataframe into tensors

                # Normalize the edges indices

                # print(f'Edges: ',type(edges[0][0]))
                # edge_idx = torch.tensor(edges.to_numpy(), dtype=torch.float) # this is not working
                edge_idx = torch.tensor(edges, dtype=torch.float)

                map_dict = {v.item():i for i,v in enumerate(torch.unique(edge_idx))}
                # print(f'Map Dict: ',map_dict)

                # Map the edge indices
                map_edge = torch.zeros_like(edge_idx)
                # print(f'Map Edge: ',map_edge)

                for k,v in map_dict.items():
                    map_edge[edge_idx == k] = v
                # print(f'Map Edge: ',map_edge)

                # Convert the Dataframe into tensors
                # attrs = torch.tensor(attributes.to_numpy(), dtype=torch.float) # no index at this point # not required because the function already returns a tensor
                pad =torch.zeros((attrs.shape[0],4), dtype=torch.float)   # what is 4 here? # why add padding?

                x =torch.cat((attrs,pad),dim=-1)

                # print(f'x : ',x)

                edge_idx = map_edge.long() 

                # print(f'Edge Index: ',edge_idx)

                # print(f'County: ',county)
                # print(f'Graph label: ',label)


                # There are missing labels

                if len(label) != 0:

                    # np_lab = label.to_numpy()  # already numpy.ndimentional - not required
                    y= torch.tensor([1] if label[0]==4 else [0], dtype=torch.long)

                    graph = Data(x=x, edge_index=edge_idx, y=y)

                    # data_list.append(graph)

                    # print(self.processed_dir)


                    if self.pre_filter is not None and not self.pre_filter(graph):
                        continue

                    if self.pre_transform is not None:
                        graph = self.pre_transform(graph)

                    torch.save(graph, osp.join(self.processed_dir, f'data_{county}.pt'))

                # break
            # break


    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """

        # if idx is 4 digits make it a string and add 0s to the front
        if len(str(idx)) == 4:
            idx = '0' + str(idx)
        else:
            idx = str(idx)
            
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
    


# # Define the main function
if __name__ == "__main__":
    # Main execution

    root_name = "/home/h6x/git_projects/gnn/model_1/data"

    train_dataset =OpioidDataset(root_name, test=False)

    print(train_dataset[1001])
    print(train_dataset[1001].x.shape)




#     print(od.len())

#     print(od[1001])


