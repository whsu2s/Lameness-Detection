import torch
import torch.nn as nn


class HRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim=4):
        super(HRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        ''' Hierarchy '''
        # BRNN 1 ========================================================================
        # Head to neck: 0-2
        self.rnn1 = nn.RNN(input_size=3*2, hidden_size=hidden_size, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        # Back: 3-8
        self.rnn2 = nn.RNN(input_size=6*2, hidden_size=hidden_size, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        # FL leg: 9-12
        self.rnn3 = nn.RNN(input_size=4*2, hidden_size=hidden_size, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        # FR leg: 13-16
        self.rnn4 = nn.RNN(input_size=4*2, hidden_size=hidden_size, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        # RL leg: 17-20
        self.rnn5 = nn.RNN(input_size=4*2, hidden_size=hidden_size, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        # RR leg: 21-24
        self.rnn6 = nn.RNN(input_size=4*2, hidden_size=hidden_size, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        # BRNN 2 ========================================================================
        self.rnn_l2 = nn.RNN(input_size=200, hidden_size=100, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        # BRNN 3 ========================================================================
        self.rnn_l41 = nn.RNN(input_size=200, hidden_size=200, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        self.rnn_l42 = nn.RNN(input_size=800, hidden_size=200, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        # BRNN 4 ========================================================================
        self.rnn_l6 = nn.LSTM(input_size=800, hidden_size=200, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)

        self.classify_layer = nn.Linear(2*200, output_dim)
        self.softmax = nn.Softmax()
        
    def forward(self, x):

        out1, h1 = self.rnn1(x[:,:,:3*2])
        out2, h2 = self.rnn2(x[:,:,3*2:9*2])
        out3, h3 = self.rnn3(x[:,:,9*2:13*2])
        out4, h4 = self.rnn4(x[:,:,13*2:17*2])
        out5, h5 = self.rnn5(x[:,:,17*2:21*2])
        out6, h6 = self.rnn6(x[:,:,21*2:])
        # Layer 1 =======================================================================
        out_layer11 = torch.cat((out1, out2), dim=2)
        out_layer12 = torch.cat((out2, out3), dim=2)
        out_layer13 = torch.cat((out2, out4), dim=2)
        out_layer14 = torch.cat((out2, out5), dim=2)
        out_layer15 = torch.cat((out2, out6), dim=2)
        #print('L1: ', out_layer11.shape)
        # Layer 2 =======================================================================
        out_layer21, _ = self.rnn_l2(out_layer11)
        out_layer22, _ = self.rnn_l2(out_layer12)
        out_layer23, _ = self.rnn_l2(out_layer13)
        out_layer24, _ = self.rnn_l2(out_layer14)
        out_layer25, _ = self.rnn_l2(out_layer15)
        #print('L2: ', out_layer21.shape) #torch.Size([16, 20, 200])
        # Layer 3 =======================================================================
        out_layer31 = out_layer21
        out_layer32 = torch.cat((torch.cat((torch.cat((out_layer22, out_layer23), dim=2), 
                                            out_layer24), dim=2), out_layer25), dim=2)
        #print('L3: ', out_layer31.shape) #torch.Size([16, 20, 200])
        # Layer 4 =======================================================================
        out_layer41, _ = self.rnn_l41(out_layer31) #torch.Size([16, 20, 400])
        out_layer42, _ = self.rnn_l42(out_layer32) #torch.Size([16, 20, 400])
        #print('L4: ', out_layer42.shape)
        # Layer 5 =======================================================================
        out_layer5 = torch.cat((out_layer41, out_layer42), dim=2)
        #print('L5: ', out_layer5.shape) #torch.Size([16, 20, 800])
        # Layer 6 =======================================================================
        out_layer6, _ = self.rnn_l6(out_layer5)
        #print('L6: ', out_layer6.shape) #torch.Size([16, 20, 100])
        # Layer 7 =======================================================================       
        #print('Output: ', out1.shape, out_layer6.shape) # torch.Size([16, 20, 100])
        output = self.classify_layer(out_layer6[:, -1, :])
        output = self.softmax(output)
        
        return output
