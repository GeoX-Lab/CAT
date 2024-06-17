from FilmUtils import load_film_data
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url
import os.path as osp
from typing import Callable, List, Optional


class FilmDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['out1_graph_edges.txt', 'out1_node_feature_label.txt']
        return [osp.join(self.root, 'raw', names[0]), osp.join(self.root, 'raw', names[1])]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data = load_film_data(self.raw_file_names[0], self.raw_file_names[1])

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


if __name__ == '__main__':
    root = r'data/Film'
    dataset = FilmDataset(root)
    dataset.process()
    print("yeah")