import numpy as np
import torch

from torch.autograd import Variable

from .reflection_gan_options import Options

#Here, we implement an object that coordinates the three models
#(ie. FECnet, Translator/Generator, and the Discriminator)
#This object will handle the training schema and inference at the highest level

class ReflectionGAN:

    def __init__(self, fecnet, translator, discriminator, options):
        self.options = options

        #Models
        self.fecnet = fecnet
        self.translator = translator
        self.discriminator = discriminator

        #Optimizers for translator and discriminator
        self.optim_t = torch.nn.Adam(
            self.translator.parameters(),
            self.options.lr_t,
            (self.options.beta1_t, 0.999)
        )
        self.optim_d = torch.nn.Adam(
            self.discriminator.parameters(),
            self.options.lr_d,
            (self.options.beta1_d, 0.999)
        )

        #Loss functions
        self.embedding_loss = torch.nn.L2Loss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.consistency_loss = torch.nn.L1Loss()

        #Loss ratios
        self.e_loss_ratio = self.options.e_loss_ratio
        self.a_loss_ratio = self.options.a_loss_ratio
        self.c_loss_ratio = self.options.c_loss_ratio

        # #Keep record of the losses
        # self.embedding_loss_hist = np.ndarray([])
        # self.adversarial_loss_d_hist = np.ndarray([])
        # self.adversarial_loss_t_hist = np.ndarray([])
        # self.consistency_loss_hist = np.ndarray([])
        # self.loss_t_hist = np.ndarray([])

    def __call__(self, img_trg, img_src):
        self.fecnet.eval()
        self.translator.eval()

        e2 = self.fecnet(img_src)
        out = self.translator(x, e2)

        return out

    #Take in target image and source expression embedding batches as parameters
    #This trains for a single batch, assume epoch training done elsewhere
    def train_batch(self, img_trg, img_src):
        loss_d = 0
        adversarial_loss_t = 0
        embedding_loss_t = 0
        consistency_loss_t = 0
        total_loss_t = 0

        self.translator.train()
        self.discriminator.train()
        self.fecnet.eval()

        e1 = self.fecnet(img_trg) #original expression embedding
        e2 = self.fecnet(img_src)

        #######################################################################
        #ADVERSARIAL LOSS
        #######################################################################

        #Train discriminator
        self.optim_d.zero_grad()

        #Train with real images
        #Set all the labels to true (real)
        label = Variable(torch.full((x.size(0),), 1).cuda())

        d_pred = self.discriminator(img_trg)

        #discriminator loss on real images
        loss_d_real = self.adversarial_loss(d_pred, label)
        loss_d_real.backward()

        #Train with fake images
        #Project expression e2 onto img_trg
        img_gen = self.translator(img_trg, e2)

        #Set label to all fake
        label.fill_(0)

        d_pred = self.discriminator(img_gen)

        #discrim loss on fake images
        loss_d_gen = self.adversarial_loss(d_pred, label)
        loss_d_gen.backward()

        loss_d += loss_d_real.mean().item() + loss_d_gen.mean().item()

        self.optim_d.step()

        #Get adversarial loss

        #Want to fool discriminator
        label.fill_(1)

        self.optim_t.zero_grad()

        #Reassign d_pred, dont want gradients from before
        d_pred = self.discriminator(img_gen)

        #adversarial loss, we'll use this later for final aggregated loss
        a_loss_t = self.adversarial_loss(d_pred, label)

        adversarial_loss_t += a_loss_t.mean().item()

        #######################################################################
        #EMBEDDING LOSS
        #######################################################################

        #Find expression embeddings of generated images
        e_gen = self.fecnet(img_gen)

        #embedding expression distance loss
        e_loss_t = self.embedding_loss(e_gen, e2)

        embedding_loss_t += e_loss_t.mean().item()

        #######################################################################
        #CONSISTENCY LOSS
        #######################################################################

        #Try to obtain original image from transformed image
        #try to see if consistent
        img_con = self.translator(img_gen, e1)

        c_loss_t = self.consistency_loss(img_con, img_trg)

        consistency_loss_t += c_loss_t.mean().item()

        
        #Incorporate all the losses
        loss_t = self.a_loss_ratio * a_loss_t + self.e_loss_ratio * e_loss_t \
               + self.c_loss_ratio * c_loss_t
        loss_t.backward()

        self.optim_t.step()

        total_loss_t += loss_t.mean().item()

        return e_loss_t, a_loss_t, c_loss_t, total_loss_t




        

