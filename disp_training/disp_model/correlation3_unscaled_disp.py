import torch.nn.init

from disp_model.disp_template import DistTemplate
from models.common import *
from utils.losses import *


class FPNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, recurrent=False):
        super(FPNEncoder, self).__init__()
        self.conv_bottom_0 = ConvBlock3(in_channels=in_channels, out_channels=32, n_convs=2,
                                        kernel_size=1, padding=0, downsample=False)

        self.conv_bottom_0_2 = ConvBlock3(in_channels=32, out_channels=32, n_convs=2,
                                          kernel_size=3, padding=1, downsample=True)

        self.conv_bottom_1 = ConvBlock3(in_channels=32, out_channels=64, n_convs=2,
                                        kernel_size=5, padding=0,
                                        downsample=False)
        self.conv_bottom_2 = ConvBlock3(in_channels=64, out_channels=128, n_convs=2,
                                        kernel_size=5, padding=0,
                                        downsample=False)
        self.conv_bottom_3 = ConvBlock3(in_channels=128, out_channels=256, n_convs=2,
                                        kernel_size=3, padding=0,
                                        downsample=True)
        self.conv_bottom_4 = ConvBlock3(in_channels=256, out_channels=out_channels, n_convs=2,
                                        kernel_size=3, padding=0, downsample=False)

        self.recurrent = recurrent
        if self.recurrent:
            self.conv_rnn = ConvLSTMCell(out_channels, out_channels, 1)

        self.conv_lateral_3 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv_lateral_2 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv_lateral_1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv_lateral_0 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, bias=True)

        self.conv_dealias_3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
        self.conv_dealias_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
        self.conv_dealias_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
        self.conv_dealias_0 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
        self.conv_out = nn.Sequential(
            ConvBlock3(in_channels=out_channels, out_channels=out_channels,
                       n_convs=1, kernel_size=3, padding=1, downsample=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                      bias=True)
        )

        self.conv_bottleneck_out = nn.Sequential(
            ConvBlock3(in_channels=out_channels, out_channels=out_channels,
                       n_convs=1, kernel_size=3, padding=1, downsample=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                      bias=True)
        )

    def reset(self):
        if self.recurrent:
            self.conv_rnn.reset()

    def forward(self, x):
        """
        :param x:
        :return: (highest res feature map, lowest res feature map)
        """

        # Bottom-up pathway
        c0_0 = self.conv_bottom_0(x)      # 61x61  Tracker: 31x31

        c0 = self.conv_bottom_0_2(c0_0)   # 31x31

        c1 = self.conv_bottom_1(c0)     # 23x23
        c2 = self.conv_bottom_2(c1)     # 15x15
        c3 = self.conv_bottom_3(c2)     # 5x5
        c4 = self.conv_bottom_4(c3)     # 1x1

        # Top-down pathway (with lateral cnx and de-aliasing)
        p4 = c4
        p3 = self.conv_dealias_3(self.conv_lateral_3(c3) + F.interpolate(p4, (c3.shape[2], c3.shape[3]), mode='bilinear'))
        p2 = self.conv_dealias_2(self.conv_lateral_2(c2) + F.interpolate(p3, (c2.shape[2], c2.shape[3]), mode='bilinear'))
        p1 = self.conv_dealias_1(self.conv_lateral_1(c1) + F.interpolate(p2, (c1.shape[2], c1.shape[3]), mode='bilinear'))
        p0 = self.conv_dealias_0(self.conv_lateral_0(c0) + F.interpolate(p1, (c0.shape[2], c0.shape[3]), mode='bilinear'))

        if self.recurrent:
            p0 = self.conv_rnn(p0)

        return self.conv_out(p0), self.conv_bottleneck_out(c4)


class JointEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JointEncoder, self).__init__()

        self.conv1 = ConvBlock3(in_channels=in_channels, out_channels=64, n_convs=2, downsample=True)
        self.conv2 = ConvBlock3(in_channels=64, out_channels=128, n_convs=2, downsample=True)
        self.convlstm0 = ConvLSTMCell(128, 128, 3)
        self.conv3 = ConvBlock3(in_channels=128, out_channels=256, n_convs=2, downsample=True)
        self.conv4 = ConvBlock3(in_channels=256, out_channels=256, kernel_size=3, padding=0,
                                n_convs=1, downsample=False)

        # Transformer Addition
        self.flatten = nn.Flatten()
        embed_dim = 256
        num_heads = 8
        self.multihead_attention0 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.prev_x_res = None
        self.gates = nn.Linear(2*embed_dim, embed_dim)
        self.ls_layer = LayerScale(embed_dim)

        # Attention Mask Transformer
        self.fusion_layer0 = nn.Sequential(nn.Linear(embed_dim*2, embed_dim),
                                           nn.LeakyReLU(0.1),
                                           nn.Linear(embed_dim, embed_dim),
                                           nn.LeakyReLU(0.1)
                                           )
        self.output_layers = nn.Sequential(nn.Linear(embed_dim, 512),
                                           nn.LeakyReLU(0.1)
                                           )

    def reset(self):
        self.convlstm0.reset()
        self.prev_x_res = None

    def forward(self, x, attn_mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convlstm0(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        if self.prev_x_res is None:
            self.prev_x_res = Variable(torch.zeros_like(x))

        x = self.fusion_layer0(torch.cat((x, self.prev_x_res), 1))

        x_attn = x[None, :, :].detach()
        x_attn = self.multihead_attention0(query=x_attn,
                                           key=x_attn,
                                           value=x_attn,
                                           attn_mask=attn_mask.bool())[0].squeeze(0)
        x = x + self.ls_layer(x_attn)

        gate_weight = torch.sigmoid(self.gates(torch.cat((self.prev_x_res, x), 1)))
        x = self.prev_x_res * gate_weight + x * (1 - gate_weight)

        self.prev_x_res = x

        x = self.output_layers(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class TrackerNetC(DistTemplate):
    def __init__(self, min_track_length, disp_patch_range, n_vis=8, patch_size=31, feature_dim=1024, **kwargs):
        super(TrackerNetC, self).__init__(min_track_length=min_track_length, n_vis=n_vis, patch_size=patch_size,
                                          **kwargs)
        # Configuration
        self.grayscale_ref = True
        self.channels_in_per_patch = 10

        # Architecture
        self.disp_patch_range = disp_patch_range
        self.feature_dim = feature_dim
        self.redir_dim = 128

        self.reference_encoder = FPNEncoder(3, self.feature_dim)

        self.target_encoder = FPNEncoder(self.channels_in_per_patch, self.feature_dim)

        # Correlation3 had k=1, p=0
        self.reference_redir = nn.Conv2d(self.feature_dim, self.redir_dim, kernel_size=3, padding=1)
        self.target_redir = nn.Conv2d(self.feature_dim, self.redir_dim, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=2)

        self.joint_encoder = JointEncoder(in_channels=1+2*self.redir_dim, out_channels=512)

        # Disp Adjustment
        self.target_vertical_reshape = nn.Conv2d(self.redir_dim + 1, self.redir_dim + 1,
                                                 stride=(2, 1), kernel_size=3, padding=1)
        self.predictor = nn.Linear(in_features=512, out_features=2, bias=False)

        # Operational
        self.loss = nn.L1Loss(reduction='none')

        self.correlation_maps = []
        self.inputs = []
        self.refs = []

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.fc_out.weight)

    def reset(self, _):
        self.joint_encoder.reset()

    def forward(self, frame_patches, event_patches, attn_mask=None):
        # Feature Extraction
        e_f0, _ = self.target_encoder(event_patches)
        f_f0, f_bottleneck = self.reference_encoder(frame_patches)
        f_f0 = self.reference_redir(f_f0)

        # Correlation and softmax
        f_corr = (e_f0 * f_bottleneck).sum(dim=1, keepdim=True)
        f_corr = self.softmax(f_corr.view(-1, 1, self.disp_patch_range // 2 * 31)).view(-1, 1, self.disp_patch_range // 2, 31)

        # Feature re-direction
        e_f0 = self.target_redir(e_f0)

        # Disp Adjustment
        f_combined_corr_e_f0 = self.target_vertical_reshape(torch.cat([e_f0, f_corr], dim=1))

        f = torch.cat([f_combined_corr_e_f0, f_f0], dim=1)
        f = self.joint_encoder(f, attn_mask)

        f = self.predictor(f)

        return f
