import torch
from torch import nn


class SCNet(nn.Module):
    def __init__(self, hparams: dict):
        super(SCNet, self).__init__()
        self.text_len = hparams['text_len']
        self.embed_channels = hparams['embed_channels']
        self.sc_convs = nn.ModuleList([
            # (batch_size, embed_channels, text_len)
            nn.Sequential(nn.Conv1d(in_channels=self.embed_channels,
                                    out_channels=hparams['conv_num_kernel'],
                                    kernel_size=window_size),  # (batch_size, conv_num_kernel, text_len-window_size+1)
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.text_len - window_size + 1))  # (batch_size, conv_num_kernel, 1)
            for window_size in hparams['conv_kernel_size']
        ])
        self.sc_fcl = nn.Linear(
            in_features=hparams['conv_num_kernel'] * len(hparams['conv_kernel_size']),
            out_features=hparams['num_api'],
        )
        self.fic_fc = nn.Linear(
            in_features=hparams['conv_num_kernel'] * len(hparams['conv_kernel_size']),
            out_features=hparams['feature_channels']
        )
        self.fic_api_feature_embedding = nn.Parameter(torch.rand(hparams['feature_channels'], hparams['num_api']))
        self.fic_mlp = nn.Sequential(
            nn.Linear(in_features=hparams['feature_channels'] * 2, out_features=hparams['feature_channels']),
            nn.Linear(in_features=hparams['feature_channels'], out_features=1),
            nn.Tanh()
        )
        self.fic_fcl = nn.Linear(in_features=hparams['num_api'] * 2, out_features=hparams['num_api'])
        self.fusion_layer = nn.Linear(in_features=hparams['num_api'] * 2, out_features=hparams['num_api'])
        self.api_task_layer = nn.Linear(in_features=hparams['num_api'], out_features=hparams['num_api'])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: (batch_size, text_len, embed_channels)
        :return (batch_size, num_api)
        """
        # semantic component
        x_trans = torch.transpose(x, 1, 2)  # (batch_size, embed_channels, text_len)
        e = [conv(x_trans) for conv in self.sc_convs]
        e = torch.cat(e, dim=2)  # (batch_size, conv_num_kernel, len(conv_num_kernel))
        e = e.view(e.size(0), -1)  # (batch_size, conv_num_kernel * len(conv_num_kernel))
        u_sc = self.sc_fcl(e)  # (batch_size, num_api)

        # feature interaction component
        u_sc_trans = self.fic_fc(e)  # (batch_size, feature_channels)
        u_mm = torch.matmul(u_sc_trans, self.fic_api_feature_embedding)  # (batch_size, num_api))
        u_concat = []
        for each_u_sc in u_sc_trans:  # (1, feature_channels)
            each_u_concate = torch.cat(
                (each_u_sc.repeat(self.fic_api_feature_embedding.size(1), 1), self.fic_api_feature_embedding.t()),
                dim=1
            )  # (num_api, feature_channels * 2)
            u_concat.append(self.fic_mlp(each_u_concate).squeeze())  # (num_api,)
        u_mlp = torch.cat(u_concat).view(u_mm.size(0), -1)  # (batch_size, num_api)
        u_fic = self.fic_fcl(torch.cat((u_mm, u_mlp), dim=1))
        u_fic = self.tanh(u_fic)  # (batch_size, num_api)

        # fusion layer
        u_mmf = self.fusion_layer(torch.cat((u_sc, u_fic), dim=1))  # (batch_size, num_api)
        task = self.api_task_layer(u_mmf)  # (batch_size, num_api)
        return task
