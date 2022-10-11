import torch

#have potential bug that occurs when batch_size>1

class FALoss(torch.nn.Module):
    def __init__(self,subscale=0.0625):
        super(FALoss,self).__init__()
        self.subscale=int(1/subscale)

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0,2,1),feature1) #[N,W*H,W*H]

        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature2,2,1,keepdim=True).repeat(1,C,1)
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        # feature2=torch.div(feature2,L2norm)
        mat2 = torch.bmm(feature2.permute(0,2,1),feature2) #[N,W*H,W*H]

        L1norm=torch.norm(mat2-mat1,1)

        return L1norm/((height*width)**2) 
