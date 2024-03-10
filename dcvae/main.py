# %%
from src.libraries import *
from time import time


# Set GPU memory to dynamic allocation instead of using all the GPU memory
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    for i, d in enumerate(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
except:
    print('No GPU detected. Use CPU instead')


def parse_arguments():
    parser = argparse.ArgumentParser(description='autoencoder experiments')
    parser.add_argument(
        '--model', type=str, default='dcvae',
        help="one of the following: 'vae', 'beta-vae', 'cvae', 'dcvae', 'RFVAE', 'FactorVAE'")
    parser.add_argument(
        '--task', type=str, default='kmnist',
        help="one of the following: 'mnist', 'fmnist','cifar10','kmnist','emnist'")
    parser.add_argument(
        '--beta', type=float, default=1.0,
        help='coefficient for KL divergence')
    parser.add_argument(
        '--num_epochs', type=int, default=50, dest='num_epochs',
        help='maximum number of epochs used during training (default: 25)')
    parser.add_argument(
        '--batch_size', type=int, default=64, dest='batch_size',
        help='batch_size used during training (default: 64)')
    parser.add_argument(
        '--train_size', type=int, default=6400, dest='train_size',
        help='number of samples used during training (default: 6400)')
    parser.add_argument(
        '--num_generated_image', type=int, default=100, dest='test_size',
        help='number of generated images (default: 100)')
    parser.add_argument(
        '--latent_dim', type=int, default=2, dest='latent_dim',
        help='number of latent embeddings (default: 2)')
    parser.add_argument(
        '--prefix', type=str, required=True, dest='prefix',
        help='directory (default: dcvae)')
    parser.add_argument(
        '--outdir', type=str, default='./results/', dest='outdir',
        help='directory (default: ./results/)')

    FLAGS, unparsed = parser.parse_known_args()

    args = parser.parse_args()
    args.test_size = int(np.sqrt(args.test_size)) ** 2

    if args.model.lower() in ['betaVAE', 'beta-vae']:
        args.latent_dim = max(2, args.latent_dim - 2)

    return args


def main(model_name, task, beta, num_epochs, train_size, batch_size, latent_dim, test_size, outdir, prefix,
         show_images):
    experiment_name = '_' + task
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'logging'), exist_ok=True)

    optimizers = {
        'primary': tf.keras.optimizers.Adam(1e-3),
        'secondary': tf.keras.optimizers.Adam(1e-3)
    }

    train_dataset, test_dataset = get_dataset(task, train_size, test_size, batch_size)
    input_dims, kernel_size, strides = get_dataset_params(task)

    model = VAE.create_model(
        model_name,
        {
            'latent_dim': latent_dim,
            'prefix': prefix,
            'input_dims': input_dims,
            'kernel_size': kernel_size,
            'strides': strides
        }
    )

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        #print('tracking uri:', mlflow.get_tracking_uri())
        #print('artifact uri:', mlflow.get_artifact_uri())
        mlflow.log_params({
            'latent_dim': latent_dim,
            'prefix': prefix,
            'input_dims': input_dims,
            'kernel_size': kernel_size,
            'strides': strides
        })
        for epoch in range(1, num_epochs + 1):
            start_time = time()
            for train_x in train_dataset:
                elbo, logpx_z, kl_divergence = model.train_step(
                    train_x, optimizers, beta=beta, epoch=epoch
                )

            mlflow.log_metrics({
                'train_elbo': elbo.numpy(),
                'train_logpx_z': logpx_z.numpy(),
                'train_kl_divergence': kl_divergence.numpy()
            }, step=epoch)

            end_time = time()
            display.clear_output(wait=False)
            for test_x in test_dataset:
                elbo, logpx_z, kl_divergence = model.elbo(test_x, beta=beta)

            mlflow.log_metrics({
                'test_elbo': elbo.numpy(),
                'test_logpx_z': logpx_z.numpy(),
                'test_kl_divergence': kl_divergence.numpy()
            }, step=epoch)

            message = ''.join(
                f'Epoch: {epoch:>5}\tTest ELBO: {elbo:>.2f}\t'
                f'Test Reconstructed Loss: {logpx_z:>.5f}\tTest KL-Divergence: '
                f'{kl_divergence:>.2f}\tTime Elapse: {end_time - start_time:.3f}'
            )
            print(message)
            save_images(model, outdir, epoch, test_x, show_images)
            plot_latent_images(model, train_x, outdir, 10, epoch, show_images=show_images)

            if epoch % 20 == 1:
                model.save_weights(os.path.join(outdir, f'{model.prefix}-{run.info.run_id}-epoch-{epoch}.ckpt'))

        image_generation(model, outdir)
        model.save_weights(os.path.join(outdir, f'{model.prefix}-{run.info.run_id}.ckpt'))
        return model


# %%
if __name__ == "__main__":
    try:
        args = parse_arguments()
        model = main(
            args.model,
            args.task,
            args.beta,
            args.num_epochs,
            args.train_size,
            args.batch_size,
            args.latent_dim,
            args.test_size,
            args.outdir,
            args.prefix,
            show_images=False
        )
    except:
        warnings.warn('Error in parameters, use default parameters instead')
        model = main(
            model_name='dcvae',
            task='kmnist',
            beta=1.0,
            num_epochs=50,
            train_size=60000,  # 12800 / 60000
            batch_size=64,  # 64 / 128
            latent_dim=2,
            test_size=25,
            outdir='results/' + str('dcvae') + str(' kmnist'),
            prefix="dcvae",
            show_images=False
        )
