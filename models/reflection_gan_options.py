#Define options for training ReflectionGAN

#lr_t : Translator learning rate
#lr_d : Discriminator learning rate
#beta1_t : Translator beta1 for Adam
#beta1_d : Discriminator beta1 for Adam
#e_loss_ratio : Embedding loss ratio
#a_loss_ratio : Adversarial loss ratio
#c_loss_ratio : Consistency loss ratio

class Options:
    
    def __init__(self, 
        lr_t,
        lr_d,
        beta1_t,
        beta1_d,
        e_loss_ratio,
        a_loss_ratio,
        c_loss_ratio
    ):

        self.lr_t = lr_t
        self.lr_d = lr_d
        self.beta1_t = beta1_t
        self.beta1_d = beta1_d
        self.e_loss_ratio = e_loss_ratio
        self.a_loss_ratio = a_loss_ratio
        self.c_loss_ratio = c_loss_ratio