import torch
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier


class UttAVLModel(BaseModel):
    """
    Audio-Visual-Lexical multimodal fusion model
    This is a reconstructed version based on the project structure
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, 
                            choices=['last', 'maxpool', 'attention'])
        parser.add_argument('--embd_method_v', default='maxpool', type=str, 
                            choices=['last', 'maxpool', 'attention'])
        parser.add_argument('--cls_layers', type=str, default='128,128')
        parser.add_argument('--dropout_rate', type=float, default=0.3)
        parser.add_argument('--bn', action='store_true')
        return parser

    def __init__(self, opt):
        super(UttAVLModel, self).__init__(opt)
        self.loss_names = ['CE']
        self.model_names = ['A', 'V', 'L', 'C']
        
        # Audio encoder
        self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, 
                                embd_method=opt.embd_method_a)
        # Visual encoder  
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v,
                                embd_method=opt.embd_method_v)
        # Lexical encoder
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        
        # Classifier
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        self.netC = FcClassifier(cls_input_size, cls_layers, 
                                 output_dim=opt.output_dim, 
                                 dropout=opt.dropout_rate, 
                                 use_bn=opt.bn)
        
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=opt.lr, 
                                             betas=(opt.beta1, 0.999),
                                             weight_decay=opt.weight_decay)
            self.optimizers = [self.optimizer]
        
        self.output_dim = opt.output_dim

    def set_input(self, input):
        self.acoustic = input['acoustic'].float().to(self.device)
        self.visual = input['visual'].float().to(self.device)
        self.lexical = input['lexical'].float().to(self.device)
        self.label = input['label'].to(self.device)
        self.missing_index = input['missing_index'].long().to(self.device)

    def forward(self):
        # Encode each modality
        self.feat_A = self.netA(self.acoustic)
        self.feat_V = self.netV(self.visual)
        self.feat_L = self.netL(self.lexical)
        
        # Concatenate features
        self.feat_fusion = torch.cat([self.feat_A, self.feat_V, self.feat_L], dim=-1)
        
        # Classification
        self.logits, self.ef_fusion_feat = self.netC(self.feat_fusion)
        self.pred = F.softmax(self.logits, dim=-1)

    def backward(self):
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        self.loss_CE.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
