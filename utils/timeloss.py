# Copyright 2020 Miralan
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDomainLoss_v1(nn.Module):
    """Time domain loss module."""
    def __init__(self, batch_size ,segment_size=3200, 
                 T_frame_sizes=[80, 160, 320],
                 T_hop_sizes=[40, 80, 160]):
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

        # Energy loss
        loss_e = torch.zeros(self.len).to(y)
        for i in range(self.len):
            y_energy = torch.as_strided(y**2, self.shapes[i], self.strides[i])
            y_hat_energy = torch.as_strided(y_hat**2, self.shapes[i], self.strides[i])
            loss_e[i] = F.l1_loss(torch.mean(y_energy, dim=-1), torch.mean(y_hat_energy, dim=-1))

        # Time loss
        loss_t = torch.zeros(self.len).to(y)
        for i in range(self.len):
            y_time = torch.as_strided(y, self.shapes[i], self.strides[i])
            y_hat_time = torch.as_strided(y_hat, self.shapes[i], self.strides[i])
            loss_t[i] = F.l1_loss(torch.mean(y_time, dim=-1), torch.mean(y_hat_time, dim=-1))
        
        # Phase loss
        y_phase = F.pad(y, (1, 0), "constant", 0) - F.pad(y, (0, 1), "constant", 0)
        y_hat_phase = F.pad(y_hat, (1, 0), "constant", 0) - F.pad(y_hat, (0, 1), "constant", 0)
        loss_p = F.l1_loss(y_phase, y_hat_phase)
        
        total_loss = torch.sum(loss_e) + torch.sum(loss_t) + loss_p
        
        return total_loss
    
class TimeDomainLoss_v2(nn.Module):
    """Time domain loss module."""
    def __init__(self, 
                 T_frame_sizes=[80, 160, 320],
                 T_hop_sizes=[40, 80, 160]):
        super(TimeDomainLoss_v2, self).__init__()
        self.filters = []
        self.len = len(T_frame_sizes)
        self.strides = []
        for i in range(len(T_frame_sizes)):
            self.filters.append((torch.ones(1, 1, T_frame_sizes[i]) / T_frame_sizes[i]).to(torch.float32))
            self.strides.append(T_hop_sizes[i])
            self.register_buffer(f'filters_{i}', self.filters[i])
        phase_filter =  torch.FloatTensor([-1, 1]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("phase_filter", phase_filter)       
            
    def forward(self, y, y_hat):
        """Calculate time domain loss

        Args:
            y (Tensor): real waveform
            y_hat (Tensor): fake waveform
        Return: 
            total_loss (Tensor): total loss of time domain
            
        """
        # Energy loss & Time loss
        loss_e = torch.zeros(self.len).to(y)
        loss_t = torch.zeros(self.len).to(y)
        for i in range(self.len):
            y_energy = F.conv1d(y**2, getattr(self, f'filters_{i}'), stride=self.strides[i])
            y_hat_energy = F.conv1d(y_hat**2, getattr(self, f'filters_{i}'), stride=self.strides[i])
            y_time = F.conv1d(y, getattr(self, f'filters_{i}'), stride=self.strides[i])
            y_hat_time = F.conv1d(y_hat, getattr(self, f'filters_{i}'), stride=self.strides[i])
            loss_e[i] = F.l1_loss(y_energy, y_hat_energy)
            loss_t[i] = F.l1_loss(y_time, y_hat_time)
        
        # Phase loss
        y_phase = F.conv1d(y, self.phase_filter, padding=1)
        y_hat_phase = F.conv1d(y_hat, self.phase_filter, padding=1)
        loss_p = F.l1_loss(y_phase, y_hat_phase)
        
        total_loss = torch.sum(loss_e) + torch.sum(loss_t) + loss_p
        
        return total_loss
        
def test(): 
    loss1 = TimeDomainLoss_v1(2, 12800)
    loss2 = TimeDomainLoss_v2()
    a = torch.randn(2, 1, 12800)
    b = torch.randn(2, 1, 12800)
    final1 = loss1(a, b)
    final2 = loss2(a, b)
    print(final1)
    print(final2)

if __name__ == '__main__': 
    test()