from tqdm.notebook import tqdm
from statistics import mean
from EffiColorizer.utility import *


# -----------------------------------------------------------------------------------------------
class TrainGAN:
    def __init__(self, model, data_loader, eval_data, running_avg_window):
        if len(eval_data['L']) < 5:
            raise Exception("len of eval_data must be at least 5")
        self.model = model
        self.data_loader = data_loader
        self.running_avg_window = running_avg_window
        self.gray_eval_imgs = batch_lab_to_rgb(eval_data['L'][:5], torch.zeros_like(eval_data['ab'][:5]))
        self.original_eval_imgs = batch_lab_to_rgb(eval_data['L'][:5], eval_data['ab'][:5])
        self.L_eval_imgs = eval_data['L'][:5]

    def pretrain_generator(self, epochs):
        self._plot_eval_imgs(gray_imgs=True, original_imgs=True, painted_imgs=True)

        loss_history = []
        for i in range(epochs):
            print(f"Epoch {i + 1}/{epochs}")

            record = 0
            count = 0
            for data in tqdm(self.data_loader):
                loss = self.model.pretrain_G(data)
                record += loss
                count += 1

            loss_history.append(record / count)
            print(f'L1 loss: {loss_history[-1]:0.5e}')
            self._plot_eval_imgs(painted_imgs=True)

        plot_cost(loss_history)
        self._plot_eval_imgs(gray_imgs=True, original_imgs=True, painted_imgs=True)

    def train_main_model(self, epochs):
        self._plot_eval_imgs(gray_imgs=True, original_imgs=True, painted_imgs=True)

        loss_meter = GANLossMeters(self.running_avg_window)
        for i in range(epochs):
            print(f"Epoch {i + 1}/{epochs}")

            for data in tqdm(self.data_loader):
                losses = self.model.train_GAN(data)
                loss_meter.record(losses)

            loss_meter.print()
            self._plot_eval_imgs(painted_imgs=True)

        loss_meter.plot()
        self._plot_eval_imgs(gray_imgs=True, original_imgs=True, painted_imgs=True)

    def _plot_eval_imgs(self, gray_imgs=None, original_imgs=None, painted_imgs=None):
        if gray_imgs:
            plot_imgs(self.gray_eval_imgs, ncols=5, figsize=(20, 4), title="gray imgs")
        if original_imgs:
            plot_imgs(self.original_eval_imgs, ncols=5, figsize=(20, 4), title="original imgs")
        if painted_imgs:
            painted_eval_imgs = self.model(self.L_eval_imgs)
            plot_imgs(painted_eval_imgs, ncols=5, figsize=(20, 4), title="painted imgs")


# -----------------------------------------------------------------------------------------------
def init_weights(model, init='norm', gain=0.02):

    def init_func(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if init == 'norm':
                nn.init.normal_(m.weight, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init == 'he':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif init == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                raise NotImplementedError(f"initialization method [{init}] is not implemented")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, mean=1.0, std=gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    model.apply(init_func)  # it applies init_func to all submodules


# -----------------------------------------------------------------------------------------------
class GANLossMeters:
    def __init__(self, running_avg_window):
        self.loss_history = {'loss_G_GAN': [],
                             'loss_G_L1': [],
                             'loss_G': [],
                             'loss_D_fake': [],
                             'loss_D_real': [],
                             'loss_D': []
                             }
        self.count = 0
        self.running_avg_window = running_avg_window

    def record(self, losses: dict):
        self.count += 1
        for loss_name, loss_value in losses.items():
            self.loss_history[loss_name].append(loss_value)
            # running average
            if self.count >= self.running_avg_window:
                avg = mean(self.loss_history[loss_name][self.count-self.running_avg_window: self.count])
                self.loss_history[loss_name][self.count-1] = avg

    def print(self):
        print("Generator loss:")
        print("   GAN:  {0:0.5e},   L1: {1:0.5e}, total: {2:0.5e}".format(
            self.loss_history['loss_G_GAN'][-1], self.loss_history['loss_G_L1'][-1], self.loss_history['loss_G'][-1]))
        print("discriminator loss:")
        print("   fake: {0:0.5e}, real: {1:0.5e}, total: {2:0.5e}".format(
            self.loss_history['loss_D_fake'][-1], self.loss_history['loss_D_real'][-1],
            self.loss_history['loss_D'][-1]))

    def plot(self):
        _, axs = plt.subplots(2, 3, figsize=(20, 10))
        X_plot = np.arange(0, self.count)
        axs[0, 0].plot(X_plot, self.loss_history['loss_G_GAN'], color='b', label='loss_G_GAN')
        axs[0, 1].plot(X_plot, self.loss_history['loss_G_L1'], color='g', label='loss_G_L1')
        axs[0, 2].plot(X_plot, self.loss_history['loss_G'], color='r', label='loss_G')

        axs[1, 0].plot(X_plot, self.loss_history['loss_D_fake'], color='b', label='loss_D_fake')
        axs[1, 1].plot(X_plot, self.loss_history['loss_D_real'], color='g', label='loss_D_real')
        axs[1, 2].plot(X_plot, self.loss_history['loss_D'], color='r', label='loss_D')

        for ax in axs.flat:
            ax.set_ylabel("Cost")
            ax.set_xlabel("# updates")
            ax.legend()


# -----------------------------------------------------------------------------------------------
def plot_cost(history, interval=None):
    if interval is None:
        interval = [0, len(history)]
    plt.plot(np.arange(interval[0], interval[1]), history[interval[0]:interval[1]])
    plt.ylabel("Cost")
    plt.xlabel("# Epochs")
    plt.show()
