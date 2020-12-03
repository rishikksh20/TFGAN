import tqdm
import torch


def validate(hp, generator, discriminator, disc_f, valloader, stft_loss, time_loss, criterion, writer, step):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    loss_g_sum = 0.0
    loss_d_sum = 0.0
    for mel, audio in loader:
        mel = mel.cuda()
        audio = audio.cuda()    # B, 1, T torch.Size([1, 1, 212893])

        # generator
        fake_audio = generator(mel) # B, 1, T' torch.Size([1, 1, 212992])

        disc_fake = discriminator(fake_audio[:, :, :audio.size(2)]) # B, 1, T torch.Size([1, 1, 212893])
        disc_real = discriminator(audio)

        # Frequency Discriminator
        disc_fake_freq = disc_f(fake_audio[:, :, :audio.size(2)].squeeze(1))
        disc_real_freq = disc_f(audio.squeeze(1))

        adv_loss =0.0
        loss_d_real = 0.0
        loss_d_fake = 0.0
        sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audio.size(2)].squeeze(1), audio.squeeze(1))
        loss_g = sc_loss + mag_loss
        # Time Domain loss
        loss_g += hp.model.lambda_time_loss * time_loss(audio.squeeze(1), fake_audio[:, :, :audio.size(2)].squeeze(1))


        for (feats_fake, score_fake), (feats_real, score_real) in zip(disc_fake, disc_real):
            adv_loss += criterion(score_fake, torch.ones_like(score_fake))

            if hp.model.feat_loss :
                for feat_f, feat_r in zip(feats_fake, feats_real):
                    adv_loss += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))
            loss_d_real += criterion(score_real, torch.ones_like(score_real))
            loss_d_fake += criterion(score_fake, torch.zeros_like(score_fake))

        adv_loss = adv_loss / len(disc_fake)

        # Frequency Discriminator
        adv_loss += criterion(disc_fake_freq, torch.ones_like(disc_fake_freq))
        loss_d_fake_freq = criterion(disc_fake_freq, torch.zeros_like(disc_fake_freq))
        loss_d_real_freq = criterion(disc_real_freq, torch.ones_like(disc_real_freq))


        loss_d_real = loss_d_real/len(score_real)
        loss_d_fake = loss_d_fake/len(disc_fake)
        loss_g += hp.model.lambda_adv * adv_loss
        loss_d = loss_d_real + loss_d_fake + loss_d_fake_freq + loss_d_real_freq
        loss_g_sum += loss_g.item()
        loss_d_sum += loss_d.item()

        loader.set_description("g %.04f d %.04f ad %.04f| step %d" % (loss_g, loss_d, adv_loss, step))

    loss_g_avg = loss_g_sum / len(valloader.dataset)
    loss_d_avg = loss_d_sum / len(valloader.dataset)

    audio = audio[0][0].cpu().detach().numpy()
    fake_audio = fake_audio[0][0].cpu().detach().numpy()

    writer.log_validation(loss_g_avg, loss_d_avg, adv_loss, generator, discriminator, audio, fake_audio, step)

    torch.backends.cudnn.benchmark = True
    generator.train()
    discriminator.train()
