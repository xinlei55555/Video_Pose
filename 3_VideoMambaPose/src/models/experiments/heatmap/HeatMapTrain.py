import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

from HeatMapLoss import PoseEstimationLoss
from HeatVideoMamba import HeatMapVideoMambaPose
from DataFormat import JHMDBLoad
from import_config import open_config

import os


def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)  # this depends on how I saved the model
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    return model


def training_loop(n_epochs, optimizer, model, loss_fn, train_set, test_set, device, checkpoint_directory, checkpoint_name, follow_up=(False, 1, None)):
    os.chdir(checkpoint_directory)
    os.makedirs(checkpoints_name, exist_ok=True)
    best_val_loss = float('inf')

    start_epoch = 1
    # then load checkpoint to follow up on the model
    if follow_up[0]:
        checkpoint_path = follow_up[2]
        model = load_checkpoint(checkpoint_path, model)
        start_epoch = follow_up[1]

    for epoch in range(start_epoch, n_epochs + start_epoch):
        print(f'Epoch {epoch} started ======>')
        model.train()  # so that the model keeps updating its weights.
        train_loss = 0.0
        # print('train batch for epoch # ', epoch)

        if epoch == start_epoch:
            # Prints GPU memory summary
            print('Memory before (in MB)', torch.cuda.memory_allocated()/1e6)
            print(f'The length of the train_set is {len(train_set)}')
            print(f'The length of the test_set is {len(test_set)}')

        print('test batch for epoch # ', epoch, '==============>')
        for i, data in enumerate(train_set):
            train_inputs, train_labels = data

            # should load individual batches to GPU
            train_inputs, train_labels = train_inputs.to(
                device), train_labels.to(device)

            # first make an initial guess as to the weights (Note: training is done in parallel)
            train_outputs = model(train_inputs)

            if epoch == start_epoch and i == 1:
                print('The shape of the outputs is ', train_outputs.shape)
                print('The shape of the labels are ', train_labels.shape)

            # determine the loss using the loss_fn which is passed into the training loop.
            # Note: Need to pass float! (not double)
            loss_train = loss_fn(train_outputs.float(), train_labels.float())

            # optimizer changes the weight and biases to zero, before starting the training again.
            optimizer.zero_grad()

            # this is what computes the derivative of the loss
            loss_train.backward()  # !this will accumulate the gradients at the leaf nodes

            # then, the optimizer will update the values of the weights based on all the derivatives of the losses computed by loss_train.backward()
            optimizer.step()

            # .item transforms loss from pytorch tensor to python value
            train_loss += loss_train.item()

            if epoch == start_epoch and i == 1:
                # Prints GPU memory summary
                print('Memory after train_batch (in MB)',
                      torch.cuda.memory_allocated()/1e6)

            torch.cuda.empty_cache()  # Clear cache to save memory

        model.eval()  # so that the model does not change the values of the parameters
        test_loss = 0.0
        with torch.no_grad():  # reduce memory while torch is using evaluation mode
            print('test batch for epoch # ', epoch, '======================>')
            for i, data in enumerate(test_set):
                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(
                    device), test_labels.to(device)

                # repeat for the validation
                test_outputs = model(test_inputs)

                # get the loss again for the validation
                loss_val = loss_fn(test_outputs.float(), test_labels.float())

                test_loss += loss_val.item()

                if epoch == start_epoch and i == 1:
                    # Prints GPU memory summary
                    print('Memory after test_batch (in MB)',
                          torch.cuda.memory_allocated()/1e6)

                torch.cuda.empty_cache()  # Clear cache to save memory

        # the shown loss should be for individual elements in the batch size
        show_loss_train, show_loss_test = train_loss / \
            len(train_set), test_loss / len(test_set)

        wandb.log({"Pointwise training loss": show_loss_train})
        wandb.log({"Pointwise testing loss": show_loss_train})

        print(f"Epoch {epoch}, Pointwise Training loss {float(show_loss_train)},"
              f" Pointwise Validation loss {float(show_loss_test)}")
        print(
            f"Full training loss: {float(train_loss)}, Full test loss: {float(test_loss)}")

        # I use the full loss when comparing, to avoid having too small numbers.
        if test_loss < best_val_loss:
            best_val_loss = test_loss

            # save model locally
            checkpoint_path = os.path.join(
                checkpoints_dir, f'heatmap_{best_val_loss:.4f}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Best model saved at {checkpoint_path}')
            print("Model parameters are of the following size",
                  len(list(model.parameters())))
    wandb.finish()


def main():
    # import configurations:
    config = open_config()

    wandb.init(
        project="1heatmap_video_mamba",

        config={
            "model_name": config['model_name'],
            "dataset": config['dataset_name'],
            "epochs": config['epoch_number'],
        }
    )

    # currently, only taking the first GPU, but later will use DDP will need to change.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and loss function
    model = HeatMapVideoMambaPose().to(device)

    # making sure to employ parallelization!!!
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # move the data to the GPU
    model = model.to(device)
    print('Model loaded successfully as follows: ', model)

    loss_fn = PoseEstimationLoss()

    # configuration
    pin_memory = False
    if torch.cuda.device_count() >= 1:
        pin_memory = True

    num_epochs = config['epoch_number']
    batch_size = config['batch_size']
    num_workers = config['num_cpus'] - 1
    normalize = config['normalized']
    default = config['default']  # custom normalization.
    follow_up = (config['follow_up'], config['previous_training_epoch'],
                 config['previous_checkpoint'])
    jump = config['jump']
    real_job = config['real_job']
    checkpoint_dir = config['checkpoint_directory']
    checkpoint_name = config['checkpoint_name']

    train_set = JHMDBLoad(train_set=True, real_job=real_job,
                          jump=jump, normalize=(normalize, default))
    test_set = JHMDBLbetteroad(train_set=False, real_job=real_job,
                               jump=jump, normalize=(normalize, default))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    print(f"The model has started training, with the following characteristics:")
    training_loop(num_epochs, optimizer, model, loss_fn,
                  train_loader, test_loader, device, checkpoint_dir, checkpoint_name, follow_up)


if __name__ == '__main__':
    main()
