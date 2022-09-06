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
from tensorflow.keras.datasets import mnist
import tensorflow.keras.backend as K


distorConstraint=0.75



ansDim = 28*28//4 #answer dimension
Qdim = 5




def repeatClass(args):
    global ansDim
    maskClass = args
    return K.repeat(maskClass, ansDim) #need to match answer dimension

def maxMSEloss(y_true, y_pred):
    global distorConstraint
    meanSError = K.mean(K.mean(K.square(y_pred- y_true), axis=-1), axis=-1)
    return K.maximum(K.mean(meanSError,axis=-1)-distorConstraint, 0.0)
    



def mergeChannel(args):
    codein0,codein1,codein2,codein3,codein4,codein5,codein6,codein7,codein8,codein9 = args
    return K.concatenate([codein0,codein1,codein2,codein3,codein4,codein5,codein6,codein7,codein8,codein9],axis=-1)

def mergCompress(args):
    featSelected, Qinput = args
    featAndQ = K.concatenate((featSelected,Qinput),axis=-1)
    return featAndQ


def claIn(args):
    outputEnc, yinput = args
    dataCla = K.concatenate((outputEnc,yinput),axis=-1)
    return dataCla

class MGAN():
    def __init__(self,model_path,output_path,train_data_path,test_data_path):
        global ansDim
        self.model_path = model_path
        self.output_path = output_path
        

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        
        self.latent_dim = 20
        self.num_classes = 10
        
        self.n_critic = 1
        self.totalImgN = 6000
        self.testImgN = 1000
        self.img_rows = 28
        self.img_cols = 28

       

        # Build the generator and critic
        self.generator = self.build_generator()
        #self.generator.summary()
        self.critic = self.build_critic()
        
        
        self.genBasisNet = self.build_genBasis()
        #self.genBasisNet.summary()
     
        self.encoder = self.build_encoder()
        
        
        self.decoder = self.build_decoder()
        self.decoder.summary()
        self.QansGen = self.build_AnsSelectQ()

        real_Q = Input(shape=(self.num_classes // 2,))
        
        
        DoutfromR = self.critic(real_Q)
        self.critic_model = Model(inputs=real_Q,outputs=DoutfromR)
        Doptimizer = RMSprop(learning_rate=0.00001)
        self.critic_model.compile(loss='categorical_crossentropy',optimizer=Doptimizer, metrics=['accuracy'])

        self.critic.trainable = False
        for layer in self.critic.layers:
            layer.trainable = False

        z_gen = Input(shape=(self.latent_dim,))

        mQ = self.generator(z_gen)

        
        


        img_input0 = Input(shape=(self.img_rows,self.img_cols,1))
        img_input1 = Input(shape=(self.img_rows,self.img_cols,1))
        img_input2 = Input(shape=(self.img_rows,self.img_cols,1))
        img_input3 = Input(shape=(self.img_rows,self.img_cols,1))
        img_input4 = Input(shape=(self.img_rows,self.img_cols,1))
        img_input5 = Input(shape=(self.img_rows,self.img_cols,1))
        img_input6 = Input(shape=(self.img_rows,self.img_cols,1))
        img_input7 = Input(shape=(self.img_rows,self.img_cols,1))
        img_input8 = Input(shape=(self.img_rows,self.img_cols,1))
        img_input9 = Input(shape=(self.img_rows,self.img_cols,1))
        

        
        
        

        encodedF = self.encoder([img_input0,img_input1,img_input2,img_input3,img_input4,img_input5,img_input6,img_input7,img_input8,img_input9])
        
        DlayerforGen = self.critic(mQ)
        QselV = self.QansGen(mQ)
        Aout = self.genBasisNet([QselV,mQ, encodedF])
        
        lossWeightIn = Input(shape=(1,)) 
        targetImg = Input(shape=(self.img_rows,self.img_cols,1))
        targetLabel = Input(shape=(10,)) 

        
        
        unoise = Input(shape=(ansDim,))
        imgOut = self.decoder([unoise,Aout,z_gen ])
        
        self.generator_model = Model([lossWeightIn,unoise,z_gen , img_input0,img_input1,img_input2,img_input3,img_input4,img_input5,img_input6,img_input7,img_input8,img_input9,targetImg, targetLabel], [imgOut,DlayerforGen])
        GenLoss = lossWeightIn * K.mean(maxMSEloss(targetImg, imgOut),axis=-1)  - K.mean(losses.categorical_crossentropy(targetLabel, DlayerforGen))
        self.generator_model.add_loss(GenLoss)
        
        Goptimizer = RMSprop(learning_rate=0.00001)

        self.generator_model.compile(loss=None, optimizer=Goptimizer)
        #self.generator_model.summary()
        
    def build_AnsSelectQ(self):
        Qcompress = Input(shape=(Qdim,))
        Q0 = Dense(5, activation='selu')(Qcompress)
        Q1 = Dense(7, activation='selu')(Q0)
        Q2 = Dense(9, activation='selu')(Q1)
        Qsel = Dense(10, activation='softmax')(Q2)
        
        AnsSelectQ = Model(inputs = Qcompress, outputs=Qsel)
        Qsel = AnsSelectQ(Qcompress)
        
        return AnsSelectQ


    def build_encoder(self):
        img_input0 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded0 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input0)
        encoded0 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded0)
        encoded0 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded0)
        enc0 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded0)
        eFlat0 = Reshape((4*4*16,1))(enc0)
       
        img_input1 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded1 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input1)
        encoded1 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded1)
        encoded1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded1)
        enc1 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded1)

        eFlat1 = Reshape((4*4*16,1))(enc1)
        

        img_input2 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded2 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input2)
        encoded2 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded2)
        encoded2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded2)
        enc2 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded2)

        eFlat2 = Reshape((4*4*16,1))(enc2)
        
        img_input3 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded3 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input3)
        encoded3 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded3)
        encoded3 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded3)
        enc3 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded3)

        eFlat3 = Reshape((4*4*16,1))(enc3)
        

        img_input4 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded4 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input4)
        encoded4 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded4)
        encoded4 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded4)
        enc4 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded4)

        eFlat4 = Reshape((4*4*16,1))(enc4)
        

        img_input5 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded5 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input5)
        encoded5 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded5)
        encoded5 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded5)
        enc5 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded5)

        eFlat5 = Reshape((4*4*16,1))(enc5)
        

        img_input6 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded6 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input6)
        encoded6 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded6)
        encoded6 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded6)
        enc6 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded6)
        eFlat6 = Reshape((4*4*16,1))(enc6)
        

        img_input7 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded7 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input7)
        encoded7 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded7)
        encoded7 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded7)
        enc7 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded7)
        eFlat7 = Reshape((4*4*16,1))(enc7)
        

        img_input8 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded8 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input8)
        encoded8 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded8)
        encoded8 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded8)
        enc8 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded8)
        eFlat8 = Reshape((4*4*16,1))(enc8)
        

        img_input9 = Input(shape=(self.img_rows,self.img_cols,1))
        encoded9 = Conv2D(8, kernel_size=(3, 3),activation='selu')(img_input9)
        encoded9 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded9)
        encoded9 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='selu')(encoded9)
        enc9 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='selu')(encoded9)
        eFlat9 = Reshape((4*4*16,1))(enc9)
        

        
        

        composChannel = Lambda(mergeChannel,name='compositeCh')([eFlat0,eFlat1,eFlat2,eFlat3,eFlat4,eFlat5,eFlat6,eFlat7,eFlat8,eFlat9])
        encoder = Model(inputs = [img_input0,img_input1,img_input2,img_input3,img_input4,img_input5,img_input6,img_input7,img_input8,img_input9], outputs=composChannel)
        composChannel = encoder([img_input0,img_input1,img_input2,img_input3,img_input4,img_input5,img_input6,img_input7,img_input8,img_input9])
        return encoder
    
    def build_decoder(self):
        global ansDim
        uniNoise = Input(shape=(ansDim,))
        featureIn = Input(shape=(ansDim,), name='featureExt')
        y_inputN = Input(shape=(self.latent_dim,), name='yin')
        noiseF = tf.keras.layers.Add()([featureIn, uniNoise])
        yhot = Lambda(claIn, output_shape=(ansDim+self.latent_dim,))([noiseF,y_inputN])
        decodeda = Dense(512, activation='selu',name='latentCode')(yhot)
        decodedb = Dense(512, activation='selu')(decodeda)
        decodedc = Dense(512, activation='selu')(decodedb)
        decubic  = Reshape((4,4,32))(decodedc)
        convt8 = Conv2DTranspose(64, (5,5), activation='selu', strides = (2,2), padding='valid')(decubic)
        convt16 = Conv2DTranspose(128, (5,5), activation='selu', strides = (2,2), padding='valid')(convt8)
        outputImg = Conv2DTranspose(1, (4,4), activation='tanh', strides=(1,1),  padding='valid')(convt16)
        
        decoder = Model(inputs=[uniNoise,featureIn, y_inputN], outputs=outputImg)
        outputImg = decoder( [uniNoise,featureIn, y_inputN] )
        return decoder

    def build_generator(self):
        inMix = Input(shape=(self.latent_dim,), name='inputandrand')
        inMixExt = Dense(20)(inMix)
                

        actSplit = Activation('selu')(inMixExt)
        gen00 = Dense(20,activation='selu')(actSplit)
        mid00 = Dense(20,activation='selu')(gen00)
        

        genPreAct00 = Add()([inMixExt,mid00])
        dimReduct0 = Dense(9,activation='selu')(genPreAct00)
        dimReduct1 = Dense(7,activation='selu')(dimReduct0)

       
        img = Dense(Qdim, activation='selu')(dimReduct1) 
        generator = Model(inputs=inMix, outputs=img)
        img = generator(inMix)
        return generator

    def build_critic(self):

        Qcode = Input(shape=(self.num_classes // 2 ,),name='inputQ')
        revAns0 = Dense(64, activation='selu')(Qcode)
        revAns1 = Dense(64, activation='selu')(revAns0)
        revAns2 = Dense(64, activation='selu')(revAns1)
        revAns3 = Dense(64, activation='selu')(revAns2)
        revAns4 = Dense(64, activation='selu')(revAns3)
        revAns33 = Dense(64, activation='selu')(revAns4)
        req = Dense(self.num_classes, name='densePred')(revAns33)
        

        
        classifyOut = keras.layers.Softmax(axis=-1)(req)
        predCla = Model(inputs=Qcode,outputs=classifyOut)
        classifyOut = predCla(Qcode)
        return predCla
    
    def build_genBasis(self):
        global ansDim
        Q_input = Input(shape=(self.num_classes,))
        compQ = Input(shape=(Qdim,))
        
        dataIn = Input(shape=(4*4*16,self.num_classes))
        reQ = Lambda(repeatClass)(Q_input)
        featSel = tf.keras.layers.Multiply()([reQ, dataIn])
        featFlat = Flatten()(featSel)

        featcompQ = Lambda(mergCompress, output_shape=(4*4*16*self.num_classes + 5,))([featFlat,compQ])
        
        encodeA = Dense(2048, activation='selu')(featcompQ)
        encodeB = Dense(1024, activation='selu')(encodeA)
        AnsOut = Dense(ansDim, activation='sigmoid')(encodeB)

        genBasisNet = Model(inputs=[Q_input,compQ, dataIn],outputs=AnsOut)
        AnsOut = genBasisNet([Q_input,compQ, dataIn])
        return genBasisNet

    
    
    
   


    def train(self, epochs, batch_size=128, sample_interval=10000, initTuneParam=1.0, addTuneParam=0.0001):
        global ansDim
        
        inputCompound = np.zeros((self.totalImgN,self.img_rows,self.img_cols,1,self.num_classes))
        testSet = np.zeros((self.testImgN,self.img_rows,self.img_cols,1,self.num_classes))
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        assert y_train.shape == (60000,)
        assert y_test.shape == (10000,)
        #https://keras.io/api/datasets/mnist/
        for imgClass in range(0,self.num_classes):
            testFeatN = self.totalImgN - np.count_nonzero(y_train==imgClass) #number of images in class imgClass of training set

            if testFeatN > 0: #if number of images in class imgClass of training set smaller than 6000
                imgTemp = x_test[np.nonzero(y_test==imgClass)]
                testTemp = imgTemp[testFeatN:]
                inputCompound[:,:,:,:,imgClass] = np.expand_dims(np.concatenate((x_train[np.nonzero(y_train==imgClass)],imgTemp[:testFeatN]),axis=0),axis=3)
                randInd = np.random.choice(testTemp.shape[0],size=self.testImgN, replace=True)
                testSet[:,:,:,:,imgClass] = np.expand_dims(testTemp[randInd],axis=3)

            else:
                exT = x_train[np.nonzero(y_train==imgClass)]
                inputCompound[:,:,:,:,imgClass] = np.expand_dims(exT[:self.totalImgN],axis=3)
                testTemp = np.expand_dims(np.concatenate((x_test[np.nonzero(y_test==imgClass)],exT[self.totalImgN:]),axis=0),axis=3)
                randInd = np.random.choice(testTemp.shape[0],size=self.testImgN, replace=False)
                testSet[:,:,:,:,imgClass] = testTemp[randInd]
        
        np.save(self.train_data_path,inputCompound)
        np.save(self.test_data_path,testSet)
        
        #normalize
        inputCompound = inputCompound.astype('float32')
        inputCompound /= 127.5
        inputCompound -= 1.0
        testSet = testSet.astype('float32')
        testSet /= 127.5
        testSet -= 1.0

        gtTest = np.transpose(inputCompound,(0,4,1,2,3))
        gtTest = gtTest.reshape((self.totalImgN*self.num_classes,self.img_rows,self.img_cols,1))
        testSet = np.repeat(inputCompound,self.num_classes, axis=0)

        yLabel = np.arange(self.num_classes, dtype='int32')
        yLabel = np.expand_dims(yLabel, axis=0)
        yLabel = np.repeat(yLabel, (batch_size//self.num_classes), axis=0)
        yLabel = yLabel.flatten()
        yLabel = keras.utils.to_categorical(yLabel, self.num_classes)

        classBat = batch_size // self.num_classes
        lossScale = np.array([initTuneParam for _ in range(batch_size)])
        for epoch in range(epochs):


            inClass = yLabel
            for criticN in range(self.n_critic):
                   
                noise = np.random.normal(0, 1, (batch_size, self.num_classes))
                genQ = self.generator.predict(np.concatenate((inClass,noise),axis=-1))
                d_loss = self.critic_model.train_on_batch( genQ, inClass)
            
  
            xBulkIn = np.zeros((classBat,self.img_rows,self.img_cols,1,self.num_classes))
            
            for catN in range(self.num_classes):
                xBulkIn[:,:,:,:,catN]= inputCompound[np.random.choice(self.totalImgN,size=classBat, replace=False),:,:,:,catN]
            
            
            
            inD = np.transpose(xBulkIn,(0,4,1,2,3))
            inD = inD.reshape((batch_size,self.img_rows,self.img_cols,1))
            xBulkIn = np.repeat(xBulkIn,self.num_classes, axis=0)
            noise = np.random.normal(0, 1, (batch_size, self.num_classes))
            unifNoi = np.random.uniform(-0.5, 0.5, size=(batch_size, ansDim))
            g_loss_fake = self.generator_model.train_on_batch([lossScale,unifNoi,np.concatenate((inClass,noise),axis=-1),xBulkIn[:,:,:,:,0],xBulkIn[:,:,:,:,1],xBulkIn[:,:,:,:,2],xBulkIn[:,:,:,:,3],xBulkIn[:,:,:,:,4],xBulkIn[:,:,:,:,5],xBulkIn[:,:,:,:,6],xBulkIn[:,:,:,:,7],xBulkIn[:,:,:,:,8],xBulkIn[:,:,:,:,9],inD,inClass],y=None)
            if (epoch+1) % 50 == 0:
                print ("%d [D loss: %f acc %f] [G loss: %f MSE-th %f] " % (epoch, d_loss[0], d_loss[1], d_loss[0]+g_loss_fake, (d_loss[0]+g_loss_fake)/lossScale[0]))
            
            lossScale = lossScale + addTuneParam
            

            if (epoch+1) % sample_interval == 0:
                intClass = np.arange(self.num_classes, dtype='int32')
                intClass = np.expand_dims(intClass, axis=0)
                intClass = np.repeat(intClass, 1000, axis=0)
                intClass = intClass.flatten()
                tLabel = keras.utils.to_categorical(intClass, self.num_classes)
                
                tNoise = np.random.normal(0, 1, (10000, self.num_classes))
                uuNoise = np.zeros((10000, ansDim))

                compressQ = self.generator.predict(np.concatenate((tLabel,tNoise ),axis=-1))
                Qsel = self.QansGen.predict(compressQ)
                advPred = self.critic_model.predict(compressQ)
                pred = np.array(np.argmax(advPred, axis=1), dtype=int)
                accur = np.count_nonzero(pred == intClass) / 10000
                print("adv accuracy %f" % accur)

                
                #print(Qsel[0:10])
                encLatent =self.encoder.predict([inputCompound[:,:,:,:,0],inputCompound[:,:,:,:,1],inputCompound[:,:,:,:,2],inputCompound[:,:,:,:,3],inputCompound[:,:,:,:,4],inputCompound[:,:,:,:,5],inputCompound[:,:,:,:,6],inputCompound[:,:,:,:,7],inputCompound[:,:,:,:,8],inputCompound[:,:,:,:,9]])
                Acont = self.genBasisNet.predict([Qsel,compressQ, encLatent])
                del encLatent
                del Qsel
                del compressQ
                gen_imgs = self.decoder.predict([uuNoise,(Acont >= 0.5).astype('float32'),np.concatenate((tLabel,tNoise ),axis=-1)])

                
                del Acont

                MSEv = K.get_value(K.mean(K.mean(K.square(gen_imgs-gtTest[0:10000]),axis=-1)))
                print(MSEv)
                del gen_imgs
                
                self.generator_model.save_weights(os.path.join(self.output_path,"MNISTD%fA%f.h5" % (MSEv,accur) ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distorConstraint', help='distortion constraint', default=1.0)
    parser.add_argument('--initTuneParam', help='initial turning parameter', default=1.0)
    parser.add_argument('--addTuneParam', help='increment turning parameter', default=0.0001)
    parser.add_argument('--model_path', help='Path to existing model weights file', default="../models/MNISTModel.h5")
    parser.add_argument('--answer_dimension', help='Number of bits of answer', default=196)
    parser.add_argument('--batch_size', help='Number of data in each batch', default=2048)
    parser.add_argument('--epochs', help='Number of epochs in training', default=40000)
    parser.add_argument('--save_interval', help='Save model for save_interval epochs', default=5000)
    parser.add_argument('--output_path', help="Directoy for where save weights", default="../outputs/models")
    parser.add_argument('--train_data_path', help="train data path", default="../data/train60000.npy")
    parser.add_argument('--test_data_path', help="test data path", default="../data/test10000.npy")

    args = parser.parse_args()
    distorConstraint = float(args.distorConstraint)
    ansDim = int(args.answer_dimension)
    Mgan = MGAN(args.model_path, args.output_path,args.train_data_path,args.test_data_path)
    Mgan.train(epochs=int(args.epochs), batch_size=int(args.batch_size),sample_interval=int(args.save_interval),initTuneParam=float(args.initTuneParam), addTuneParam=float(args.addTuneParam))     
