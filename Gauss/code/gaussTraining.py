
import os
import numpy as np
import tensorflow as tf
import argparse



from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, Conv2D, Conv2DTranspose


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import losses
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow.keras.backend as K



distorConstraint=0.75



def mergeChannel(args):
    codein0,codein1,codein2,codein3= args
    return K.concatenate([codein0,codein1,codein2,codein3],axis=-1)

def claIn(args):
    outputEnc, yinput = args
    dataCla = K.concatenate((outputEnc,yinput),axis=-1)
    return dataCla




def maxMSEloss(y_true, y_pred):
    global distorConstraint
    meanSError = K.mean(K.square(y_pred- y_true), axis=-1)
    return K.maximum(K.mean(meanSError - distorConstraint, axis=-1), 0.0 )
    
 
class gGAN():
    def __init__(self,testDataSize,model_path,output_path,answer_dimension):
    
        
        self.model_path = model_path
        self.output_path = output_path
        self.ansDim = answer_dimension
        self.channels = 3
        self.latent_dim = 8
        self.num_classes = 4
        
        self.testImgN = testDataSize
        self.n_critic = 1 #number of times in training critic
        Doptimizer = RMSprop(learning_rate=0.00003)

        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.genBasisNet = self.build_genBasis()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.quantizer = self.build_AnsQuantiz()

        self.genBasisNet.summary()
        self.encoder.summary()
        
        real_Q = Input(shape=(4,))
        ansRet = Input(shape=(self.ansDim,))
        realF = Input(shape=(3,4))
        DoutfromR = self.critic(real_Q)
        self.critic_model = Model(inputs=real_Q,outputs=DoutfromR)
        self.critic_model.compile(loss='categorical_crossentropy',optimizer=Doptimizer, metrics=['accuracy'])

        self.critic.trainable = False
        for layer in self.critic.layers:
            layer.trainable = False

        z_gen = Input(shape=(self.latent_dim,))
        
        Qout = self.generator(z_gen)


        
        img_input0 = Input(shape=(3,1))
        img_input1 = Input(shape=(3,1))
        img_input2 = Input(shape=(3,1))
        img_input3 = Input(shape=(3,1))

        lossWeightIn = Input(shape=(1,)) 
        targetImg = Input(shape=(3,))
        targetLabel = Input(shape=(4,)) 

        
        encodedF = self.encoder([img_input0,img_input1,img_input2,img_input3])

        
        Aout = self.genBasisNet([Qout, encodedF])
        
        unoise = Input(shape=(self.ansDim,))
        Aquantized = self.quantizer([unoise,Aout])


        DlayerforGen = self.critic(Qout)
        

        imgOut = self.decoder([Aquantized,z_gen])
        self.generator_model = Model([lossWeightIn,unoise,z_gen,img_input0,img_input1,img_input2,img_input3,targetImg, targetLabel], [imgOut,DlayerforGen])
        GenLoss = lossWeightIn * (maxMSEloss(targetImg, imgOut))  - K.mean(losses.categorical_crossentropy(targetLabel, DlayerforGen))

        self.generator_model.add_loss(GenLoss)
        
        Goptimizer = RMSprop(learning_rate=0.00003)

        self.generator_model.compile(loss=None, optimizer=Goptimizer)



    def build_AnsSelectQ(self):
        Qcompress = Input(shape=(4,))
        Q0 = Dense(4, activation='selu')(Qcompress)
        Q1 = Dense(4, activation='selu')(Q0)
        Q2 = Dense(4, activation='selu')(Q1)
        Qsel = Dense(4, activation='softmax')(Q2)

        AnsSelectQ = Model(inputs = Qcompress, outputs=Qsel)
        Qsel = AnsSelectQ(Qcompress)

        return AnsSelectQ

    def build_AnsQuantiz(self):
        uniNoise = Input(shape=(self.ansDim,))
        ansIn = Input(shape=(self.ansDim,))
        qbar = tf.keras.layers.Add()([ansIn, uniNoise])
        
        ansQuantizer = Model(inputs = [uniNoise,ansIn], outputs=qbar)
        qbar = ansQuantizer([uniNoise,ansIn])
        
        return ansQuantizer

    def build_encoder(self):
        img_input0 = Input(shape=(3,1))
        imgFlat0 = Flatten()(img_input0)
    

        img_input1 = Input(shape=(3,1))
        imgFlat1 = Flatten()(img_input1)
    

        img_input2 = Input(shape=(3,1))
        imgFlat2 = Flatten()(img_input2)
    
        
        img_input3 = Input(shape=(3,1))
        imgFlat3 = Flatten()(img_input3)
        

        outputEnc0 = Reshape((3,1))(imgFlat0)
        outputEnc1 = Reshape((3,1))(imgFlat1)
        outputEnc2 = Reshape((3,1))(imgFlat2)
        outputEnc3 = Reshape((3,1))(imgFlat3)

        composChannel = Lambda(mergeChannel,name='compositeCh')([outputEnc0,outputEnc1,outputEnc2,outputEnc3])
        encoder = Model(inputs = [img_input0,img_input1,img_input2,img_input3], outputs=composChannel)
        composChannel = encoder([img_input0,img_input1,img_input2,img_input3])
        return encoder
    def build_decoder(self):
        featureIn = Input(shape=(self.ansDim,), name='featureExt')
        y_inputN = Input(shape=(self.latent_dim,), name='yin')
        yhot = Lambda(claIn, output_shape=(self.ansDim+self.latent_dim,))([featureIn,y_inputN])


        decodeda = Dense(256, activation='selu',name='latentCode')(yhot)
        decodedb = Dense(256, activation='selu')(decodeda)
        decodedh = Dense(256, activation='selu')(decodedb)
        decodedg = Dense(256, activation='selu')(decodedh)
        decodedz = Dense(256, activation='selu')(decodedg)
        decodedy = Dense(256, activation='selu')(decodedz)
        decodedx = Dense(256, activation='selu')(decodedy)

        decoded = Dense(256, activation='selu')(decodedx)
        outputImg = Dense(3, activation='linear')(decoded)
        decoder = Model(inputs=[featureIn, y_inputN], outputs=outputImg)
        outputImg = decoder( [featureIn, y_inputN] )
        return decoder

    def build_generator(self):
        inMix = Input(shape=(self.latent_dim,), name='inputandrand')
        
        inMixExt = Dense(8)(inMix)
                

        actSplit = Activation('selu')(inMixExt)
        gen00 = Dense(8,activation='selu')(actSplit)
        mid00 = Dense(8,activation='selu')(gen00)
        mid01 = Dense(8,activation='selu')(mid00)

        genPreAct00 = keras.layers.Add()([inMixExt,mid01])

    
        img = Dense(4, activation='selu')(genPreAct00) 
        generator = Model(inputs=inMix, outputs=img)
        img = generator(inMix)
        return generator

    def build_critic(self):

        Qcode = Input(shape=(4,),name='inputQ')
        revAns0 = Dense(64, activation='selu')(Qcode)
        revAns1 = Dense(64, activation='selu')(revAns0)
        revAns2 = Dense(64, activation='selu')(revAns1)
        revAns3 = Dense(64, activation='selu')(revAns2)
        revAns3 = Dense(64, activation='selu')(revAns3)
        revAns33 = Dense(64, activation='selu')(revAns3)
        req = Dense(4, name='densePred')(revAns33)
        classifyOut = keras.layers.Softmax(axis=-1)(req)
        predCla = Model(inputs=Qcode,outputs=classifyOut)
        classifyOut = predCla(Qcode)
        return predCla
        
    def build_genBasis(self):
        img_input = Input(shape=(4,))
        dataIn = Input(shape=(3,4))
       
        dataFlat = Flatten()(dataIn)
        concatD = keras.layers.Concatenate(axis=-1)([img_input, dataFlat])
        gin00 = Dense(256,activation='selu')(concatD)
        genAns0 = Dense(256, activation='selu',name='layerDense00')(gin00)
        genAns1 = Dense(256, activation='selu',name='layerDense01')(genAns0)
        genAns2 = Dense(256,activation='selu')(genAns1)
        genAns3 = Dense(256,activation='selu')(genAns2)
        genAns4 = Dense(256,activation='selu')(genAns3)
        genAns5 = Dense(256,activation='selu')(genAns4)
        genAns6 = Dense(256,activation='selu')(genAns5)
        
        genAnspreAct1 = Dense(256,activation='selu')(genAns6)
        AnsOut = Dense(self.ansDim, activation='sigmoid',name='outputAnswer')(genAnspreAct1)

        genBasisNet = Model(inputs=[img_input,dataIn],outputs=AnsOut)
        AnsOut = genBasisNet([img_input,dataIn])
        return genBasisNet



    def train(self, epochs, batch_size=128, sample_interval=10000, initTuneParam=1.0, addTuneParam=0.0001):
        
        yLabel = np.arange(self.num_classes, dtype='int32')
        yLabel = np.expand_dims(yLabel, axis=0)
        yLabel = np.repeat(yLabel, (batch_size//self.num_classes), axis=0)
        yLabel = yLabel.flatten()
        intClass = np.copy(yLabel)
        yLabel = keras.utils.to_categorical(yLabel, self.num_classes)
        classBat = batch_size // self.num_classes
        lossScale = np.array([initTuneParam for _ in range(batch_size)])
        for epoch in range(epochs):
            inClass = yLabel
            for criticN in range(self.n_critic):
                
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim-self.num_classes))
                genQ = self.generator.predict(np.concatenate((inClass,noise),axis=-1))
                d_loss = self.critic_model.train_on_batch( genQ, inClass)
    

            lossScale += addTuneParam
            #generate data batch
            d1nn = np.random.normal(np.tile([3.0,-3.0,-3.0],(classBat,1,1)),np.tile([3,3,3],(classBat,1,1)),size=(classBat,1,3))
            datare1nn = np.transpose(d1nn, (0,2,1))
            d111 = np.random.normal(np.tile([3.0,3.0,3.0],(classBat,1,1)),np.tile([3,3,3],(classBat,1,1)),size=(classBat,1,3))
            datare111 = np.transpose(d111, (0,2,1))
            dn1n = np.random.normal(np.tile([-3.0,3.0,-3.0],(classBat,1,1)),np.tile([3,3,3.0],(classBat,1,1)),size=(classBat,1,3))
            dataren1n = np.transpose(dn1n, (0,2,1))
            dnn1 = np.random.normal(np.tile([-3.0,-3.0,3.0],(classBat,1,1)),np.tile([3.0,3.0,3.0],(classBat,1,1)),size=(classBat,1,3))
            datarenn1 = np.transpose(dnn1, (0,2,1))
            xBulkIn = np.zeros(((batch_size//self.num_classes), 3,1, self.num_classes ))
            xBulkIn[:,:,:,0] = datare111
            xBulkIn[:,:,:,1] = datare1nn
            xBulkIn[:,:,:,2] = dataren1n
            xBulkIn[:,:,:,3] = datarenn1
            inD = np.transpose(xBulkIn,(0,3,1,2))
            inD = inD.reshape((batch_size,3))
            xBulkIn = np.repeat(xBulkIn,self.num_classes, axis=0)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim-self.num_classes))
            unifNoi = np.random.uniform(-0.5, 0.5, size=(batch_size, self.ansDim))
            g_loss_fake = self.generator_model.train_on_batch([lossScale,unifNoi,np.concatenate((inClass,noise),axis=-1),xBulkIn[:,:,:,0],xBulkIn[:,:,:,1],xBulkIn[:,:,:,2],xBulkIn[:,:,:,3],inD,inClass],y=None)
            if (epoch+1) % 10 == 0:
                print("%d [D loss: %f acc %f] [G loss: %f MSE %f] " % (epoch, d_loss[0], d_loss[1], g_loss_fake+d_loss[0], (g_loss_fake+d_loss[0])/lossScale[0]))
            if (epoch+1) % sample_interval == 0:
                #generate validation data
                d1nn = np.random.normal(np.tile([3.0,-3.0,-3.0],(classBat,1,1)),np.tile([3,3,3],(classBat,1,1)),size=(classBat,1,3))
                datare1nn = np.transpose(d1nn, (0,2,1))
                d111 = np.random.normal(np.tile([3.0,3.0,3.0],(classBat,1,1)),np.tile([3,3,3],(classBat,1,1)),size=(classBat,1,3))
                datare111 = np.transpose(d111, (0,2,1))
                dn1n = np.random.normal(np.tile([-3.0,3.0,-3.0],(classBat,1,1)),np.tile([3,3,3.0],(classBat,1,1)),size=(classBat,1,3))
                dataren1n = np.transpose(dn1n, (0,2,1))
                dnn1 = np.random.normal(np.tile([-3.0,-3.0,3.0],(classBat,1,1)),np.tile([3.0,3.0,3.0],(classBat,1,1)),size=(classBat,1,3))
                datarenn1 = np.transpose(dnn1, (0,2,1))
                xBulkIn = np.zeros(((batch_size//self.num_classes), 3,1, self.num_classes ))
                xBulkIn[:,:,:,0] = datare111
                xBulkIn[:,:,:,1] = datare1nn
                xBulkIn[:,:,:,2] = dataren1n
                xBulkIn[:,:,:,3] = datarenn1
                inD = np.transpose(xBulkIn,(0,3,1,2))
                inD = inD.reshape((batch_size,3))
                xBulkIn = np.repeat(xBulkIn,self.num_classes, axis=0)
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim-self.num_classes))
                Qout = self.generator.predict(np.concatenate((inClass,noise),axis=-1))
                encodedF = self.encoder.predict([xBulkIn[:,:,:,0],xBulkIn[:,:,:,1],xBulkIn[:,:,:,2],xBulkIn[:,:,:,3]])
                Acont = self.genBasisNet.predict([Qout, encodedF])
                deData = self.decoder.predict([(Acont >= 0.5).astype('float32'),np.concatenate((inClass,noise),axis=-1)])
                advPred = self.critic_model.predict(Qout)
                pred = np.array(np.argmax(advPred, axis=1), dtype=int)
                accur = np.count_nonzero(pred == intClass) / batch_size
                print("adversary accuracy %f" % accur)


                #[deData, predClass] =self.generator_model.predict([lossScale,uuNoise,np.concatenate((inClass,noise),axis=-1),xBulkIn[:,:,:,0],xBulkIn[:,:,:,1],xBulkIn[:,:,:,2],xBulkIn[:,:,:,3],inD,inClass])
                valMSE =  K.get_value(K.mean(K.mean(K.square(deData-inD),axis=-1)))
                print("distotion %f" % valMSE)
                
                self.generator_model.save_weights(os.path.join(self.output_path,"gaussD%fA%f.h5" % (valMSE,accur) ))

    def test(self):
        
        self.generator_model.load_weights(self.model_path)
        
        yLabel = np.arange(self.num_classes, dtype='int32')
        yLabel = np.expand_dims(yLabel, axis=0)
        yLabel = np.repeat(yLabel, (self.testImgN//self.num_classes), axis=0)
        yLabel = yLabel.flatten()
        intClass = np.copy(yLabel)
        yLabel = keras.utils.to_categorical(yLabel, self.num_classes)
        #generate test data
        classBat = self.testImgN // self.num_classes
        d1nn = np.random.normal(np.tile([3.0,-3.0,-3.0],(classBat,1,1)),np.tile([3,3,3],(classBat,1,1)),size=(classBat,1,3))
        datare1nn = np.transpose(d1nn, (0,2,1))
        d111 = np.random.normal(np.tile([3.0,3.0,3.0],(classBat,1,1)),np.tile([3,3,3],(classBat,1,1)),size=(classBat,1,3))
        datare111 = np.transpose(d111, (0,2,1))
        dn1n = np.random.normal(np.tile([-3.0,3.0,-3.0],(classBat,1,1)),np.tile([3,3,3.0],(classBat,1,1)),size=(classBat,1,3))
        dataren1n = np.transpose(dn1n, (0,2,1))
        dnn1 = np.random.normal(np.tile([-3.0,-3.0,3.0],(classBat,1,1)),np.tile([3.0,3.0,3.0],(classBat,1,1)),size=(classBat,1,3))
        datarenn1 = np.transpose(dnn1, (0,2,1))
        xBulkIn = np.zeros(((self.testImgN //self.num_classes), 3,1, self.num_classes ))
        xBulkIn[:,:,:,0] = datare111
        xBulkIn[:,:,:,1] = datare1nn
        xBulkIn[:,:,:,2] = dataren1n
        xBulkIn[:,:,:,3] = datarenn1
        inD = np.transpose(xBulkIn,(0,3,1,2))
        inD = inD.reshape((self.testImgN,3))
        xBulkIn = np.repeat(xBulkIn,self.num_classes, axis=0)
        noise = np.random.normal(0, 1, (self.testImgN, self.latent_dim-self.num_classes))
        Qout = self.generator.predict(np.concatenate((yLabel,noise),axis=-1))
        advPred = self.critic_model.predict(Qout)
        pred = np.array(np.argmax(advPred, axis=1), dtype=int)
        accur = np.count_nonzero(pred == intClass) / self.testImgN
        print("adv accuracy %f" % accur)


        encodedF = self.encoder.predict([xBulkIn[:,:,:,0],xBulkIn[:,:,:,1],xBulkIn[:,:,:,2],xBulkIn[:,:,:,3]])
        Acont = self.genBasisNet.predict([Qout, encodedF])
        deData = self.decoder.predict([(Acont >= 0.5).astype('float32'),np.concatenate((yLabel,noise),axis=-1)])
        testMSE =  K.get_value(K.mean(K.mean(K.square(deData-inD),axis=-1)))
        print("test distortion %f" % testMSE)
        
        
    
     

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--distorConstraint', help='distortion constraint', default=1.0)
    parser.add_argument('--initTuneParam', help='initial turning parameter', default=1.0)
    parser.add_argument('--addTuneParam', help='increment turning parameter', default=0.0001)
    parser.add_argument('--testdataSize', help='Number of test data', default=2048)
    parser.add_argument('--model_path', help='Path to existing model weights file', default="../models/gaussModel.h5")
    parser.add_argument('--answer_dimension', help='Number of bits of answer', default=6)
    parser.add_argument('--batch_size', help='Number of data in each batch', default=2048)
    parser.add_argument('--epochs', help='Number of epochs in training', default=40000)
    parser.add_argument('--save_interval', help='Save model for save_interval epochs', default=5000)
    parser.add_argument('--output_path', help="Directory used to save weights", default="../outputs/models")

    args = parser.parse_args()

    distorConstraint = float(args.distorConstraint)
    ggan = gGAN(int(args.testdataSize),args.model_path, args.output_path,int(args.answer_dimension))
    ggan.train(epochs=int(args.epochs), batch_size=int(args.batch_size),sample_interval=int(args.save_interval),initTuneParam=float(args.initTuneParam), addTuneParam=float(args.addTuneParam))
    
    
