import torch
import torch.nn as nn

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

# class Collector():
#     collect_state = True
#     collect_action = False  
#     def __init__(self, multiplier, head):
#         self.head = head
#         self.states = []
#         self.actions = []
#     def reset(self):
#         self.states = []
#         self.actions = []
#     def __call__(self, b, s): 
#         if self.head == -1:
#             self.states.append(b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
#         else:
#             self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
#         return b


class Collector:
    collect_state = True
    collect_action = False

    def __init__(self, multiplier, head, num_heads=None):
        self.head = head
        self.num_heads = num_heads  # Only needed if you want head-specific slicing
        self.states = []
        self.actions = []

    def reset(self):
        self.states = []
        self.actions = []

    def __call__(self, b, s):
        # Shape: (batch_size=1, seq_len, hidden_dim)
        final_token = b[0, -1].detach().clone()

        if self.head == -1 or self.num_heads is None:
            self.states.append(final_token)
        else:
            try:
                # Automatically infer head size
                d = final_token.shape[-1]
                head_dim = d // self.num_heads
                reshaped = final_token.reshape(self.num_heads, head_dim)
                self.states.append(reshaped[self.head])
            except Exception as e:
                print(f"[Collector Warning] Could not reshape: {e}")
                self.states.append(final_token)  # Fallback

        return b

    
class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.direction.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
        return b