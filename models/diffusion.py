import torch
import torch.nn as nn






class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k = 16)
        else:
            self.grouper = SimpleEncoder(k = 32, embed_dims=512)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )  
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        # query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        #线性层相当于只对每个点的维度进行加权和
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num) # b n c (B, 256, 3)/(B, 256, 128)
        pe =  self.pos_embed(coor)  #(B, 256, 384)
        x = self.input_proj(f)      #(B, 256, 384)

        #use self-attention to encoder, and loop 6 epoch
        x = self.encoder(x + pe, coor) # b n c  (B, 256, 384)
        global_feature = self.increase_dim(x) # B N 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        #这是预测出来的点
        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)    #(B, 512, 3)

        #这是原始点云的fps采样
        coarse_inp = misc.fps(xyz, self.num_query//2) # B 256 3
        #cat the fps of origin with predicted points
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 256+512 3

        mem = self.mem_link(x)      #(B, 256, 384)

        #transe pos dim to 1 for the coarse points
        query_ranking = self.query_ranking(coarse) # b 768 1
        #return the rank idx of coarse points
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b 768 1
        #downsample the coarse points from 768 points to 512 points
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))    #(B, 512, 3)

        if self.training:
            #在线过程中给训练集添加噪声
            # add denoise task
            # first pick some point : 64?
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1) # B 576 3
            denoise_length = 64     

            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b 576 384

            # forward decoder
            #q为动态的query，v为对输入特征的编码
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            return q, coarse, denoise_length

        else:
            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c
            
            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0