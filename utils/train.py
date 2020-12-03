import os
import math
import tqdm
import torch
import itertools
import traceback
from model.generator import Generator
from model.multiscale import MultiScaleDiscriminator
from .utils import get_commit_hash
from .validation import validate
from utils.stft_loss import MultiResolutionSTFTLoss
from utils.timeloss import TimeDomainLoss_v1
from model.freq_discriminator import FrequencyDiscriminator

def train(args, pt_dir, chkpt_path, trainloader, valloader, writer, logger, hp, hp_str):
    model_g = Generator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
    #print("Generator : \n",model_g)

    model_d = MultiScaleDiscriminator(hp.model.num_D, hp.model.ndf, hp.model.n_layers,
                                      hp.model.downsampling_factor, hp.model.disc_out).cuda()
    model_f = FrequencyDiscriminator().cuda()

    #print("Discriminator : \n", model_d)
    optim_g = torch.optim.Adam(model_g.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    optim_d = torch.optim.Adam(itertools.chain(model_d.parameters(), model_f.parameters()),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))

    githash = get_commit_hash()


    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        model_f.load_state_dict(checkpoint['model_f'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        logger.info("Starting new training run.")

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    try:
        model_g.train()
        model_d.train()
        stft_loss = MultiResolutionSTFTLoss()
        criterion = torch.nn.MSELoss().cuda()
        time_loss = TimeDomainLoss_v1(hp.train.batch_size, hp.audio.segment_length, hp.time_loss_params.win_lengths,
                                      hp.time_loss_params.hop_sizes)
        time_loss_valid = TimeDomainLoss_v1(1, hp.audio.segment_length, hp.time_loss_params.win_lengths,
                                      hp.time_loss_params.hop_sizes)

        for epoch in itertools.count(init_epoch+1):
            if epoch % hp.log.validation_interval == 0:
                with torch.no_grad():
                    validate(hp, model_g, model_d, model_f, valloader, stft_loss, time_loss_valid, criterion, writer, step)

            trainloader.dataset.shuffle_mapping()
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
            avg_g_loss = []
            avg_d_loss = []
            avg_adv_loss = []
            for (melG, audioG), (melD, audioD) in loader:
                melG = melG.cuda()      # torch.Size([16, 80, 64])
                audioG = audioG.cuda()  # torch.Size([16, 1, 16000])
                melD = melD.cuda()      # torch.Size([16, 80, 64])
                audioD = audioD.cuda()  #torch.Size([16, 1, 16000]

                # generator
                optim_g.zero_grad()
                fake_audio = model_g(melG)[:, :, :hp.audio.segment_length]  # torch.Size([16, 1, 12800])

                
                
                loss_g = 0.0

                # reconstruct the signal from multi-band signal


                sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audioG.size(2)].squeeze(1), audioG.squeeze(1))
                loss_g = sc_loss + mag_loss

                # Time Domain loss
                loss_g += hp.model.lambda_time_loss * time_loss(audioG.squeeze(1), fake_audio[:, :, :audioG.size(2)].squeeze(1))

                adv_loss = 0.0
                if step > hp.train.discriminator_train_start_steps:

                    disc_fake_g = model_d(fake_audio)
                    # for multi-scale Time discriminator

                    for feats_fake, score_fake in disc_fake_g:
                        # adv_loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                        adv_loss += criterion(score_fake, torch.ones_like(score_fake))
                    adv_loss = adv_loss / len(disc_fake_g) # len(disc_fake) = 3

                    # For Frequency Discriminator
                    disc_fake_freq_g = model_f(fake_audio.squeeze(1))
                    adv_loss += criterion(disc_fake_freq_g, torch.ones_like(disc_fake_freq_g))



                    if hp.model.feat_loss :
                        disc_real = model_d(audioG)
                        for (feats_fake, score_fake), (feats_real, _) in zip(disc_fake, disc_real):
                            for feat_f, feat_r in zip(feats_fake, feats_real):
                                adv_loss += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))

                    loss_g += hp.model.lambda_adv * adv_loss
            

                loss_g.backward()
                optim_g.step()

                # discriminator
                loss_d_avg = 0.0
                if step > hp.train.discriminator_train_start_steps:
                    fake_audio = model_g(melD)[:, :, :hp.audio.segment_length]

                    fake_audio = fake_audio.detach()
                    loss_d_sum = 0.0
                    for _ in range(hp.train.rep_discriminator):
                        optim_d.zero_grad()

                        # Time Domain Discriminator
                        disc_fake = model_d(fake_audio)
                        disc_real = model_d(audioD)
                        loss_d = 0.0
                        loss_d_real = 0.0
                        loss_d_fake = 0.0
                        for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                            loss_d_real += criterion(score_real, torch.ones_like(score_real))
                            loss_d_fake += criterion(score_fake, torch.zeros_like(score_fake))
                        loss_d_real = loss_d_real / len(disc_real) # len(disc_real) = 3
                        loss_d_fake = loss_d_fake / len(disc_fake) # len(disc_fake) = 3

                        # Frequency Discriminator
                        disc_fake_freq = model_f(fake_audio.squeeze(1))
                        loss_d_fake_freq = criterion(disc_fake_freq, torch.zeros_like(disc_fake_freq))

                        disc_real_freq = model_f(audioD.squeeze(1))
                        loss_d_real_freq = criterion(disc_real_freq, torch.ones_like(disc_real_freq))

                        loss_d = loss_d_real + loss_d_fake + loss_d_fake_freq + loss_d_real_freq
                        loss_d.backward()
                        optim_d.step()
                        loss_d_sum += loss_d
                    loss_d_avg = loss_d_sum / hp.train.rep_discriminator
                    loss_d_avg = loss_d_avg.item()

                step += 1
                # logging
                loss_g = loss_g.item()
                avg_g_loss.append(loss_g)
                avg_d_loss.append(loss_d_avg)
                avg_adv_loss.append(adv_loss)

                if any([loss_g > 1e8, math.isnan(loss_g), loss_d_avg > 1e8, math.isnan(loss_d_avg)]):
                    logger.error("loss_g %.01f loss_d_avg %.01f at step %d!" % (loss_g, loss_d_avg, step))
                    raise Exception("Loss exploded")

                if step % hp.log.summary_interval == 0:
                    writer.log_training(loss_g, loss_d_avg, adv_loss, step)
                    loader.set_description("Avg : g %.04f d %.04f ad %.04f| step %d" % (sum(avg_g_loss) / len(avg_g_loss),
                                                                                sum(avg_d_loss) / len(avg_d_loss),
                                                                                sum(avg_adv_loss) / len(avg_adv_loss),
                                                                                step))
            if epoch % hp.log.save_interval == 0:
                save_path = os.path.join(pt_dir, '%s_%s_%04d.pt'
                    % (args.name, githash, epoch))
                torch.save({
                    'model_g': model_g.state_dict(),
                    'model_d': model_d.state_dict(),
                    'model_f': model_f.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
