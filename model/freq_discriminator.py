import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, nf_prev, nf, stride):
        self.block1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(
                nf_prev,
                nf,
                kernel_size=3,
                stride=2,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv1d(
                nf,
                nf,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
        )

        self.block2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(
                nf,
                nf,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv1d(
                nf,
                nf,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
        )

        self.shortcut = nn.utils.weight_norm(nn.Conv1d(
                nf_prev,
                nf,
                kernel_size=1,
                stride=2,
            ))

    def forward(self, x):
        x1 = self.block1(x)
        x1 = x1 + self.shortcut(x)
        return self.block2(x1)