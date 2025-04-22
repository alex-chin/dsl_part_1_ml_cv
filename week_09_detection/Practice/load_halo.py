from datasets import load_from_disk
import pandas as pd


class Load_ds:
    ds = None  # Статическая переменная класса

    def __init__(self, path = '..\\halo-infinite-angel-videogame'):
        self.path = path
        self.start()

    def get_train(self):
        return pd.DataFrame(Load_ds.ds['train'])

    def get_test(self):
        return pd.DataFrame(Load_ds.ds['test'])

    def start(self):
        if Load_ds.ds is None:
            Load_ds.ds = load_from_disk(self.path)
        
class Download_ds:
    def __init__(self):
        self.splits = {'train': 'data/train-00000-of-00001-0d6632d599c29801.parquet',
                  'validation': 'data/validation-00000-of-00001-c6b77a557eeedd52.parquet',
                  'test': 'data/test-00000-of-00001-866d29d8989ea915.parquet'}
    def get_train(self):
        return pd.read_parquet("hf://datasets/Francesco/halo-infinite-angel-videogame/" + self.splits["train"])

    def get_test(self):
        return pd.read_parquet("hf://datasets/Francesco/halo-infinite-angel-videogame/" + self.splits["test"])