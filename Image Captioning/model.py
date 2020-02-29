import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # Variables definition
        self.embed_size = embed_size    
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Layer definition
        
        # Embeddings Layer
        self.Wemb = nn.Embedding(num_embeddings = vocab_size, # size of the dictionary of embeddings
                                 embedding_dim = embed_size)  # the size of each embedding vector 
        # LSTM Layer
        self.lstm = nn.LSTM(input_size = embed_size,   # The number of expected features in the input x 
                            hidden_size = hidden_size, # The number of features in the hidden state h
                            num_layers = num_layers,   # Number of recurrent layers 
                            batch_first = True)        # True = the input and output tensors are provided
        
        # Linear Layer
        self.linear = nn.Linear(in_features = hidden_size, # size of each input sample
                                out_features = vocab_size) # size of each output sample
    
    def forward(self, features, captions):
        
        # Size variables definition 
        batch_size = features.shape[0]
        inputs = captions[:, :-1]
        
        # Initialize hidden state to 0
        self.hidden = (torch.zeros(1, 
                                   batch_size, 
                                   self.hidden_size).to(self.device),
                       torch.zeros(1, 
                                   batch_size, 
                                   self.hidden_size).to(self.device))
        
        # Embedding Layer
        x = self.Wemb(inputs)
        
        # Join images and captions 
        x = torch.cat((features.unsqueeze(1), x),
                       dim = 1)
        
        # LSTM Layer
        x, self.hidden = self.lstm(x, self.hidden)
        
        # Linear Layer
        x = self.linear(x)
        
        return x
        
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # Initialize output vector
        subtitle = []
        
        for i in range(max_len):
            
            # Pass the example through the network
            x, states = self.lstm(inputs,
                                  states)
            x = self.linear(x.squeeze(1))
            x = x.squeeze(0).max(0)[1]
            
            # Add prediction like variable (convert tensor to variable)
            subtitle.append(x.item())
        
            # Then put the new input (word) for the new iteration (if there is)
            inputs = self.Wemb(x).unsqueeze(0).unsqueeze(0)
        
        return subtitle 
    
    
    
    
    
    
    
    