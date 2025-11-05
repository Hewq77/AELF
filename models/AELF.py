import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from .AMoFE import  DownSample, SkipUpSample, RSM, ResBlock, conv, AMoFE
from einops import rearrange

class CrossTransAttention(nn.Module):
    def __init__(self, num_heads=4, dim=64):
        super().__init__()
        self.num_heads = num_heads
        bias=True
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim*1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, feat_x, feat_y):
        b, c, h, w = feat_x.shape
        
        q = self.q_dwconv(self.q(feat_x))
        kv = self.kv_dwconv(self.kv(feat_y))
        k,v = kv.chunk(2, dim=1)

        # (B, C, H, W) -> (B, head, head_dim, HW)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class MutCrossAttention(nn.Module):
    """Mutual cross attention: HS->MS and MS->HS."""
    def __init__(self, num_heads=4, dim=64):
        super().__init__()
        self.mca = CrossTransAttention(num_heads=num_heads,dim=dim)

    def forward(self, feat_hs, feat_ms):
        feat_h2m = self.mca(feat_x=feat_hs, feat_y=feat_ms)
        feat_m2h = self.mca(feat_x=feat_ms, feat_y=feat_hs)
        return feat_h2m, feat_m2h

# Modal Complementary Guidance Module
class MCGM(nn.Module):
    def __init__(self, dim, bias):
        super(MCGM, self).__init__()
        self.in_x = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.in_y = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)

        pool_sizes = [8, 4, 2]
        self.pools_x = nn.ModuleList([nn.AvgPool2d(k, k) for k in pool_sizes])
        self.pools_y = nn.ModuleList([nn.AvgPool2d(k, k) for k in pool_sizes])
        self.convs_x = nn.ModuleList([nn.Conv2d(dim, dim, 3, 1, 1, bias=bias) for _ in pool_sizes])
        self.convs_y = nn.ModuleList([nn.Conv2d(dim, dim, 3, 1, 1, bias=bias) for _ in pool_sizes])
        self.attns = nn.ModuleList([MutCrossAttention(num_heads=4, dim=dim) for _ in pool_sizes])

        self.relu = nn.GELU()
        self.sum_x = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.sum_y = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)

    def forward(self, x, y):
        x_size = x.size()
        res_x = self.in_x(x)
        res_y = self.in_y(y)
        for i in range(len(self.pools_x)):
            if i == 0:
                x_, y_ = self.attns[i](self.convs_x[i](self.pools_x[i](x)), self.convs_y[i](self.pools_y[i](y)))
            else:
                x_, y_ = self.attns[i](self.convs_x[i](self.pools_x[i](x)+x_up), self.convs_y[i](self.pools_y[i](y)+y_up))
            res_x = torch.add(res_x, F.interpolate(x_, x_size[2:], mode='bilinear', align_corners=True))
            res_y = torch.add(res_y, F.interpolate(y_, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_x) - 1:
                x_up = F.interpolate(x_, scale_factor=2, mode='bilinear', align_corners=True)
                y_up = F.interpolate(y_, scale_factor=2, mode='bilinear', align_corners=True)
        out_x = x + self.sum_x(self.relu(res_x))
        out_y = y + self.sum_y(self.relu(res_y))

        return out_x, out_y
        
##########################################################################
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias, scale_unetfeats, csff, cross=False, num_heads=[1,2,4]):
        super(Encoder, self).__init__()

        self.encoder_level1 = ResBlock(n_feat,                     kernel_size, bias=bias, act=act)
        self.encoder_level2 = ResBlock(n_feat+scale_unetfeats,     kernel_size, bias=bias, act=act)
        self.encoder_level3 = ResBlock(n_feat+(scale_unetfeats*2), kernel_size, bias=bias, act=act)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        if cross:
            self.image_event_transformer1 = AMoFE(channels=n_feat,                       num_experts=3, k=3)
            self.image_event_transformer2 = AMoFE(channels=n_feat + scale_unetfeats,     num_experts=3, k=3)
            self.image_event_transformer3 = AMoFE(channels=n_feat + 2*scale_unetfeats,   num_experts=3, k=3)
            
        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                        n_feat,                       kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats,      n_feat + scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + 2 * scale_unetfeats, n_feat + 2 * scale_unetfeats, kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                       n_feat,                       kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats,     n_feat + scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + 2 * scale_unetfeats, n_feat + 2 * scale_unetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None, msi_branch=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])
        if msi_branch is not None:
            mb1, mb2, mb3 = msi_branch
            enc1, gates1 = self.image_event_transformer1(enc1, mb1)

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])
        if msi_branch is not None:
            enc2, gates2 = self.image_event_transformer2(enc2, mb2)

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
        if msi_branch is not None:
            enc3, gates3 = self.image_event_transformer3(enc3, mb3)

        if msi_branch is not None:
            return [enc1, enc2, enc3], [gates1, gates2, gates3]
        else:
            return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = ResBlock(n_feat,                       kernel_size, bias=bias, act=act)
        self.decoder_level2 = ResBlock(n_feat + scale_unetfeats,     kernel_size, bias=bias, act=act)
        self.decoder_level3 = ResBlock(n_feat + 2 * scale_unetfeats, kernel_size, bias=bias, act=act)

        self.skip_attn1 = conv(n_feat,                   n_feat,                   kernel_size, bias=bias)
        self.skip_attn2 = conv(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size, bias=bias)

        self.up21 = SkipUpSample(n_feat,                   scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]

def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)
##########################################################################
class AELF(nn.Module):
    def __init__(self, hsi_c=103, msi_c=5, scale_factor=4, n_feat=64, scale_unetfeats=32, kernel_size=3, bias=True):
        super(AELF, self).__init__()

        act = nn.PReLU()
        self.scale_factor = scale_factor

        self.shallow_feat1 = conv(hsi_c, n_feat, kernel_size, bias=bias)
        self.shallow_feat2 = conv(hsi_c, n_feat, kernel_size, bias=bias)
        self.shallow_feat_msi = conv(msi_c, n_feat, kernel_size, bias=bias)

        self.MCGM = MCGM(n_feat, bias=bias)

        self.s1_hsi_encoder = Encoder(n_feat, kernel_size, act, bias, scale_unetfeats, csff=False)
        self.s1_hsi_decoder = Decoder(n_feat, kernel_size, act, bias, scale_unetfeats)

        self.msi_encoder = Encoder(n_feat, kernel_size, act, bias, scale_unetfeats, csff=False)

        self.s2_hsi_encoder = Encoder(n_feat, kernel_size, act, bias, scale_unetfeats, csff=True, cross=True, num_heads=[1,2,4])
        self.s2_hsi_decoder = Decoder(n_feat, kernel_size, act, bias, scale_unetfeats)

        self.RSM = RSM(n_feat, n2_feat=hsi_c, kernel_size=1, bias=bias)
        
        self.concat  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.tail     = conv(n_feat, hsi_c, kernel_size, bias=bias)


    def forward(self, lrhs, hrms):

        hrhs = F.interpolate(lrhs, scale_factor=self.scale_factor, mode='bicubic')
        #--------1) Coarse Extraction and Guidance--------
        x1 = self.shallow_feat1(hrhs)
        feat1 = self.s1_hsi_encoder(x1)
        res1 = self.s1_hsi_decoder(feat1)

        x2_rsmfeats, stage1_img = self.RSM(res1[0], hrhs)
        x2  = self.shallow_feat2(hrhs)
        hrms  = self.shallow_feat_msi(hrms)
        x2_cat = self.concat(torch.cat([x2, x2_rsmfeats], 1))
        
        # Modal Complementary Guidance
        x2_cat, hrms = self.MCGM(x2_cat, hrms)

        #--------2) Fine-Grained Adaptive Fusion--------
        hrms1 = self.msi_encoder(hrms)
        feat2, gates = self.s2_hsi_encoder(x2_cat, feat1, res1, hrms1)
        res2 = self.s2_hsi_decoder(feat2)

        out = self.tail(res2[0]) + hrhs

        return out, stage1_img , gates
    
if __name__ == "__main__":
    import torch
    from thop import profile

    # Model
    model = AELF(hsi_c=103, msi_c=5, 
                   n_feat=64, scale_unetfeats=32, 
                   kernel_size=3).cuda(0)

    hs = torch.randn(1, 103, 32, 32).cuda(0)
    ms = torch.randn(1, 5, 128, 128).cuda(0)

    with torch.no_grad():
        output, stage1_img, gates = model(hs, ms)
    print('out shape:', output.shape)