from __future__ import print_function, division
import scipy

#from keras.datasets import mnist
from keras_contrib.layers import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

class CycleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'patient1_2_6'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                #    self.predict_full_dataset_at_the_end()
                #if (d_loss[1]>0.92 and g_loss[0]<3):
                #    self.predict_many_images_B_to_A()
                #    return
                    #self.sample_images(epoch, batch_i)
                    #self.predict_full_dataset_at_the_end()
                    #self.translated_full_tif_image("./../../top_mosaic_09cm_area1.tif")
                #if batch_i % sample_interval == 0:
                    #self.translated_full_tif_image("Generated_images/top_mosaic_09cm_area1.tif", epoch, epochs, batch_i,self.data_loader.n_batches, d_loss[0], 100*d_loss[1], g_loss[0] )
                    #self.predict_many_images()
        return

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

    def predict_full_dataset_at_the_end(self):
        os.makedirs('converted_datasets/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=300, is_testing=True)
        j=0
        for i in imgs_A:
            fake_B = self.g_AB.predict(imgs_A[j][np.newaxis,:])
            # Rescale images 0 - 1
            fake_B = 0.5 * fake_B + 0.5
            imsave("converted_datasets/%s/%d.png" % (self.dataset_name, j), fake_B[0])
            j+=1


    def translated_full_tif_image(self, image_path, epoch, epochs, batch_i, batches, d_loss_0, d_loss_1, g_loss_0 ):
        crop_height = 512
        crop_width = 512
        full_tif_image = cv2.cvtColor(cv2.imread(image_path,-1), cv2.COLOR_BGR2RGB)
        translated_full_image = np.zeros((3076, 2048,3), np.float)
        for x_increment in range(3):
            for y_increment in range(5):
                cropped_img = full_tif_image[crop_height*y_increment:crop_height*(y_increment+1), crop_width*x_increment:crop_width*(x_increment+1) ]
                #input_image = np.expand_dims(np.float32(cropped_img[:crop_height, :crop_width]),axis=0)/255.0
                input_image = np.array(cropped_img)/127.5 - 1.
                fake_B = self.g_AB.predict(input_image[np.newaxis,:])
                fake_B = 0.5 * fake_B + 0.5
                #fake_B = np.array(fake_B[0,:])
                translated_full_image[crop_height*y_increment:crop_height*(y_increment+1), crop_width*x_increment:crop_width*(x_increment+1)] = fake_B[0]
        translated_full_image = translated_full_image[0:2569, 0:1919,:]
        file_name = "./Generated_images/translated_full_image_[Epoch_%d_%d]_[Batch_%d_%d]_[D_loss:_%f_acc:_%d]_[G_loss:_%f].png" %(epoch, epochs, batch_i, batches, d_loss_0, d_loss_1, g_loss_0 )
        imsave(file_name, translated_full_image)


    def predict_many_images(self):
        crop_height = 512
        crop_width = 512
        path_list = ['/home/bilel/Dropbox/Urban_areas_segmentation_from_uav_journal_paper/Code/Vaihingen512_t/train',
        '/home/bilel/Dropbox/Urban_areas_segmentation_from_uav_journal_paper/Code/Vaihingen512_t/val'
        ]
        for p in path_list:
            for f in  glob.glob(os.path.join(p, '*.png')):
                image = cv2.cvtColor(cv2.imread(f,-1), cv2.COLOR_BGR2RGB)
                translated_full_image = np.zeros((crop_height, crop_width,3), np.float)
                input_image = np.array(image)/127.5 - 1.
                fake_B = self.g_AB.predict(input_image[np.newaxis,:])
                fake_B = 0.5 * fake_B + 0.5
                translated_full_image = fake_B[0]
                os.remove(f)
                imsave("%s" %f, translated_full_image)


    def predict_many_images_B_to_A(self):
        crop_height = 512
        crop_width = 512
        path_list = ['/home/bilel/Dropbox/Urban_areas_segmentation_from_uav_journal_paper/Code/Postdam512_t/train',
        '/home/bilel/Dropbox/Urban_areas_segmentation_from_uav_journal_paper/Code/Postdam512_t/val'
        ]
        for p in path_list:
            for f in  glob.glob(os.path.join(p, '*.png')):
                image = cv2.cvtColor(cv2.imread(f,-1), cv2.COLOR_BGR2RGB)
                translated_full_image = np.zeros((crop_height, crop_width,3), np.float)
                input_image = np.array(image)/127.5 - 1.
                fake_A = self.g_BA.predict(input_image[np.newaxis,:])
                fake_A = 0.5 * fake_A + 0.5
                translated_full_image = fake_A[0]
                os.remove(f)
                imsave("%s" %f, translated_full_image)


if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=20000, batch_size=5, sample_interval=50)
