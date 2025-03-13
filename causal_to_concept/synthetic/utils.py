from collections import OrderedDict

class LossCollection:
    def __init__(self):
        self.reset()

    def add_loss(self, loss_updates, bs):
        for key, value in loss_updates.items():
            if key in self.loss_dict:
                self.loss_dict[key] = self.loss_dict[key] + bs * value
            else:
                self.loss_dict[key] = bs * value
        self.steps += bs

    def reset(self):
        self.steps = 0
        self.loss_dict = OrderedDict()

    def get_mean_loss(self):
        mean_loss_dict = OrderedDict()
        for key, value in self.loss_dict.items():
            mean_loss_dict[key] = value / self.steps
        return mean_loss_dict

    def print_mean_loss(self):
        print("Current mean losses are:")
        print(self.get_mean_loss())