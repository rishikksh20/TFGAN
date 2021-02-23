import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDomainLoss_v1(nn.Module):
    """Time domain loss module."""
    def __init__(self, batch_size ,segment_size=3200, 
                 T_frame_sizes=[1, 240, 480, 960],
                 T_hop_sizes=[1, 120, 240, 480]):
        super(TimeDomainLoss_v1, self).__init__()
        self.shapes = []
        self.strides = []
        self.seg_size = segment_size
        for i in range(len(T_frame_sizes)):
            no_over_lap = T_frame_sizes[i] - T_hop_sizes[i]
            self.shapes.append((batch_size,
                               (segment_size - no_over_lap)//T_hop_sizes[i],
                                T_frame_sizes[i]
                                ))
            self.strides.append((segment_size,
                                 T_hop_sizes[i],
                                 1
                                 ))
        self.len = len(self.shapes)
        
    def forward(self, y, y_hat):
        """Calculate time domain loss

        Args:
            y (Tensor): real waveform
            y_hat (Tensor): fake waveform
        Return: 
            total_loss (Tensor): total loss of time domain
            
        """

        # Energy loss & Time loss & Phase loss
        loss_e = torch.zeros(self.len).to(y)
        loss_t = torch.zeros(self.len).to(y)
        loss_p = torch.zeros(self.len).to(y)
        
        for i in range(self.len):
            y_tmp = torch.as_strided(y, self.shapes[i], self.strides[i])
            y_hat_tmp = torch.as_strided(y_hat, self.shapes[i], self.strides[i])
            
            loss_e[i] = F.l1_loss(torch.mean(y_tmp**2, dim=-1), torch.mean(y_hat_tmp**2, dim=-1))
            loss_t[i] = F.l1_loss(torch.mean(y_tmp, dim=-1), torch.mean(y_hat_tmp, dim=-1))
            if i == 0:
                y_phase = F.pad(y_tmp.transpose(1, 2), (1, 0), "constant", 0) - F.pad(y_tmp.transpose(1, 2), (0, 1), "constant", 0)
                y_hat_phase = F.pad(y_hat_tmp.transpose(1, 2), (1, 0), "constant", 0) - F.pad(y_hat_tmp.transpose(1, 2), (0, 1), "constant", 0)
            else:
                y_phase = F.pad(y_tmp, (1, 0), "constant", 0) - F.pad(y_tmp, (0, 1), "constant", 0)
                y_hat_phase = F.pad(y_hat_tmp, (1, 0), "constant", 0) - F.pad(y_hat_tmp, (0, 1), "constant", 0)
            loss_p[i] = F.l1_loss(y_phase, y_hat_phase)
        
        total_loss = torch.sum(loss_e) + torch.sum(loss_t) + torch.sum(loss_p)
        
        return total_loss

      
def test():
    torch.manual_seed(1234)
    loss = TimeDomainLoss_v1(2, 12800)
    real = torch.randn(2, 1, 12800)
    fake = torch.randn(2, 1, 12800)
    final_loss = loss(real, fake)
    print(final_loss)
    


if __name__ == '__main__': 
    test()