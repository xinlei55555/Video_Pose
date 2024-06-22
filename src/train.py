import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.optim as optim

import wandb
import argparse
from tqdm import tqdm

from loss.PoseLoss import PoseEstimationLoss
from data_format.DataFormat import JHMDBLoad
from data_format.CocoVideoLoader import COCOVideoLoader
from import_config import open_config

import os


def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)  # this depends on how I saved the model
    return model


def training_loop(config, n_epochs, optimizer, scheduler, model, loss_fn, train_set, test_set, device, rank, world_size,
                  checkpoint_directory, checkpoint_name,  follow_up=(False, 1, None)):
    # os.chdir(os.path.join(os.getcwd(), checkpoint_directory))
    os.makedirs(os.path.join(checkpoint_directory, checkpoint_name), exist_ok=True)
    best_val_loss = float('inf')

    # logging the gradients in the model
    wandb.watch(model, log_freq=10)

    start_epoch = 1
    # then load checkpoint to follow up on the model
    if follow_up[0]:
        checkpoint_path = follow_up[2]
        model = load_checkpoint(checkpoint_path, model)
        start_epoch = follow_up[1]

    for epoch in range(start_epoch, n_epochs + start_epoch):
        print(f'[==>] Epoch {epoch} started ')

        # telling the data loader which epoch we are at
        if torch.cuda.device_count() > 1 and config['parallelize']:
            train_set.sampler.set_epoch(epoch)

        model.train()  # so that the model keeps updating its weights.
        train_loss = 0.0
        # print('train batch for epoch # ', epoch)

        if epoch == start_epoch:
            # Prints GPU memory summary
            print('\t Memory before (in MB)', torch.cuda.memory_allocated()/1e6)
            print(
                f'The number of batches in the train_set is {len(train_set)}')
            print(f'The number of batches in the test_set is {len(test_set)}')

        print('[=============>] train batch for epoch # ', epoch )
        for i, data in enumerate(train_set):
            train_inputs, train_labels = data

            # should load individual batches to GPU
            train_inputs, train_labels = train_inputs.to(
                device), train_labels.to(device)

            # first make an initial guess as to the weights (Note: training is done in parallel)
            train_outputs = model(train_inputs)

            if epoch == start_epoch and i == 0:
                print('The shape of the inputs is ', train_inputs.shape)
                print('The shape of the outputs is ', train_outputs.shape)
                print('The shape of the labels are ', train_labels.shape)

            # determine the loss using the loss_fn which is passed into the training loop.
            # Note: Need to pass float! (not double)
            loss_train = loss_fn(train_outputs.float(), train_labels.float())
    
            # checking for vanishing gradient.
            if config['show_gradients'] and i == 0:
                for name, param in model.named_parameters():
                    print(f'Parameter: {name}')
                    print(f'Values: {param.data}')
                    if param.grad is not None:
                        print(f'Gradients: {param.grad}')
                    else:
                        print('No gradient computed for this parameter')

            # optimizer changes the weight and biases to zero, before starting the training again.
            optimizer.zero_grad()

            # this is what computes the derivative of the loss
            loss_train.backward()  # !this will accumulate the gradients at the leaf nodes

            # Gradient clipping
            if config['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])  # Clip gradients with a maximum norm of 1.0

            # then, the optimizer will update the values of the weights based on all the derivatives of the losses computed by loss_train.backward()
            optimizer.step()

            # .item transforms loss from pytorch tensor to python value
            train_loss += loss_train.item()

            if epoch == start_epoch and i == 0:
                # Prints GPU memory summary
                print('\t Memory after train_batch (in MB)',
                      torch.cuda.memory_allocated()/1e6)

            torch.cuda.empty_cache()  # Clear cache to save memory

        model.eval()  # so that the model does not change the values of the parameters
        test_loss = 0.0
        with torch.no_grad():  # reduce memory while torch is using evaluation mode
            print('[======================>] test batch for epoch # ', epoch)
            for i, data in enumerate(test_set):
                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(
                    device), test_labels.to(device)

                # repeat for the validation
                test_outputs = model(test_inputs)

                # get the loss again for the validation
                loss_val = loss_fn(test_outputs.float(), test_labels.float())

                test_loss += loss_val.item()

                if epoch == start_epoch and i == 0:
                    # Prints GPU memory summary
                    print('\t Memory after test_batch (in MB)',
                          torch.cuda.memory_allocated()/1e6)

                torch.cuda.empty_cache()  # Clear cache to save memory

        # update scheduler
        if config['scheduler']:
            print('[==========================>] Registering the learning rate.')
            scheduler.step(test_loss)

        # the shown loss should be for individual elements in the batch size
        show_loss_train, show_loss_test = train_loss / \
            len(train_set), test_loss / len(test_set)

        if rank == 0:
            print(f'[===================================>] Completed Epoch {epoch}')
            print(f'[************************************************************]')
            print('Information')
            wandb.log({"Pointwise training loss": show_loss_train})
            wandb.log({"Pointwise testing loss": show_loss_test})
            wandb.log({"Training loss": train_loss})
            wandb.log({"Testing loss": test_loss})

            print(f"Epoch {epoch},\n \t Pointwise Training loss {float(show_loss_train)}, \n"
                  f" \t Pointwise Validation loss {float(show_loss_test)}")
            print(
                f"\t Full training loss: {float(train_loss)}, \n \t Full test loss: {float(test_loss)}")

            # if scheduler defined:
            if config['scheduler']:
                lr = optimizer.param_groups[0]['lr']
                print(f'The current learning rate is: {lr}')

            # I use the full loss when comparing, to avoid having too small numbers.
            if test_loss < best_val_loss or epoch % 50 == 0:
                if test_loss < best_val_loss:
                    best_val_loss = test_loss
                # save model locally
                checkpoint_path = os.path.join(
                    checkpoint_directory, checkpoint_name, f"{config['model_type']}_{checkpoint_name}_{test_loss:.4f}_epoch_{epoch}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Best model saved at {checkpoint_path}')
                print("\t Model parameters are of the following size",
                      len(list(model.parameters())))
            print(f'[************************************************************]')


    if rank == 0:
        wandb.finish()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, config, config_file_name):
    wandb.init(
        project=config['model_name'],
        name=config['checkpoint_name'],
        notes=str(config),
        # entity=config_file_name,  # this will be the new name
        config={
            "dataset": config['dataset_name'],
            "epochs": config['epoch_number'],
        }
    )
    num_epochs = config['epoch_number']
    batch_size = config['batch_size']
    normalize = config['normalized']
    default = config['default']  # custom normalization.
    follow_up = (config['follow_up'], config['previous_training_epoch'],
                 config['previous_checkpoint'])
    jump = config['jump']
    real_job = config['real_job']
    checkpoint_dir = config['checkpoint_directory']
    checkpoint_name = config['checkpoint_name']
    dataset_name = config['dataset_name']
    # loading the data initially:
    if dataset_name == 'JHMDB':
        train_set = JHMDBLoad(config, train_set=True, real_job=real_job,
                            jump=jump, normalize=(normalize, default))
        test_set = JHMDBLoad(config, train_set=False, real_job=real_job,
                            jump=jump, normalize=(normalize, default))
    if dataset_name == 'COCO':
        train_set = COCOVideoLoader(config, train_set = True, real_job=real_job)
        test_set = COCOVideoLoader(config, train_set = False, real_job=real_job)


    if torch.cuda.device_count() == 1 or not config['parallelize']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # configuration
        pin_memory = True  # if only 1 GPU
        num_cpu_cores = os.cpu_count()
        num_workers = config['num_cpus'] * (num_cpu_cores) - 1
        print(f'num_workers is: {num_workers}, for {num_cpu_cores} cores')

        # data loader
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # Initialize the model
        # choosing the right model:
        if config['model_type'] == 'heatmap':
            model = HeatMapVideoMambaPose(config).to(device)

        elif config['model_type'] == 'HMR_decoder':
            model = HMRVideoMambaPose(config).to(device)
        
        elif config['model_type'] == 'MLP_only_decoder':
            model = MLPVideoMambaPose(config).to(device)
        
        elif config['model_type'] == 'HMR_decoder_coco_pretrain':
            model = HMRVideoMambaPoseCOCO(config).to(rank)

        else:
            print('Your selected model does not exist! (Yet)')
            return

        model = model.to(device)  # to unique GPU
        print('Model loaded successfully as follows: ', model)

        # loss
        loss_fn = PoseEstimationLoss(config)

        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['learning_rate'])
        
        # learning rate scheduler
        # I will leave the rest of the parameters as the default
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5) # half the learning rate each time

        # Training loop
        print(f"The model has started training, with the following characteristics:")

        training_loop(config, num_epochs, optimizer, scheduler, model, loss_fn,
                      train_loader, test_loader, device, rank, world_size, checkpoint_dir, checkpoint_name, follow_up)

    elif torch.cuda.device_count() == 0:
        print("ERROR! No GPU detected...")
        return

    else:
        # When parallel, set = 0 and pin_memory = False
        num_workers = 0
        pin_memory = False

        # Initialize process group
        setup(rank=rank, world_size=world_size)

        # Fixing data loader for parallelization
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False,)

        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=train_sampler, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False,)

        # use normal test loader for the test set
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # loading the model
        # sending the model to the correct rank
        if config['model_type'] == 'heatmap':
            model = HeatMapVideoMambaPose(config).to(rank)

        elif config['model_type'] == 'HMR_decoder':
            model = HMRVideoMambaPose(config).to(rank)
        
        elif config['model_type'] == 'MLP_only_decoder':
            model = MLPVideoMambaPose(config).to(rank)
        
        elif config['model_type'] == 'HMR_decoder_coco_pretrain':
            model = mHMRVideoMambaPoseCOCO(config).to(rank)

        else:
            print('Your selected model does not exist! (Yet)')
            return
        
        model = DDP(model, device_ids=[
                    rank], output_device=rank, find_unused_parameters=True)

        # loss
        loss_fn = PoseEstimationLoss(config)

        # optimizer
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
        if config['optimizer'] == 'adamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        else:
            print("No optimizers selected!")
            raise NotImplementedError

        # learning rate scheduler
        # I will leave the rest of the parameters as the default
        scheduler = RLR(optimizer=optimizer, factor=config['scheduler_factor']) # half the learning rate each time

        # Training loop
        print(f"The model has started training, with the following characteristics:")
        training_loop(config, num_epochs, optimizer, scheduler, model, loss_fn,
                      train_loader, test_loader, device, rank, world_size, checkpoint_dir, checkpoint_name, follow_up)

        # Cleanup
        dist.destroy_process_group()


if __name__ == '__main__':
    # importing all possible models:
    from models.heatmap.HeatVideoMamba import HeatMapVideoMambaPose
    from models.HMR_decoder.HMRMambaPose import HMRVideoMambaPose
    from models.MLP_only_decoder.MLPMambaPose import MLPVideoMambaPose
    from models.HMR_decoder_coco_pretrain.HMRMambaPose import HMRVideoMambaPoseCOCO

    # argparse to get the file path of the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='heatmap/heatmap_beluga.yaml',
                        help='Name of the configuration file')
    args = parser.parse_args()
    config_file = args.config

    # import configurations:
    config = open_config(config_file)
    if not config['use_last_frame_only'] and config['jump'] != config['num_frames']:
        print("The Jump does not match the parameter for last frame!")

    # a few sanity checks before running the model

    # since not parallel, I set the rank = 0, and the world size = 1 (by default)
    if torch.cuda.device_count() <= 1 or not config['parallelize']:
        main(0, 1, config, config_file)

    else:
        # world size determines the number of GPUs running together.
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, config, config_file),
                 nprocs=world_size, join=True)
