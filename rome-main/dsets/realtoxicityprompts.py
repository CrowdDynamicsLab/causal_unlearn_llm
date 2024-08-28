import typing
from datasets import load_dataset

def load_realtoxicityprompts():
    dataset = load_dataset("allenai/real-toxicity-prompts")
    return dataset

class RealToxicityPromptsDataset(Dataset):
    def __init__(
        self, size: typing.Optional[int] = None, *args, **kwargs
    ):
        self.dataset = load_realtoxicityprompts()
        self.data = self.dataset["train"].filter(lambda x: x["challenging"] == True)
        
        if size is not None:
            self.data = self.data.select(range(size))

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
