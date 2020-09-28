import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def get_loss_acc(model, train_loader, valid_loader):
    """
    Get loss and validation accuracy of example neural network
    :param model:
    :param train_loader:
    :param valid_loader:
    :return:
    """
    n_epochs = 2
    learning_rate = 0.001

    # training loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # measurements used for graphing loss
    batch_loss = []

    for e in range(1, n_epochs + 1):
        # initialize variable to monitor training loss
        train_loss = 0.0
        # train the model
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass
            output = model(data)
            # calculate loss
            loss = criterion(output, target)
            # backward pass
            loss.backward()
            # optimizer step
            optimizer.step()
            # record the average batch loss
            batch_loss.append(loss.item())

    # after training for 2 epochs check the validation accuracy
    correct = 0
    total = 0
    for data, target in valid_loader:
        # forward pass
        output = model(data)
        # get the predicted class from the max probability value
        _, pred = torch.max(output, 1)
        # count the total number of correct labels for which the predicted and true labels are equal
        total += target.size(0)
        correct += (pred == target).sum()

    # calculate the accuracy
    # to convert 'correct' from  a tensor to a scalar use '.item()'
    valid_acc = correct.item() / total

    # return model stats
    return batch_loss, valid_acc


def compare_init_weights(model_list, plot_title, train_loader, valid_loader, plot_n_batches=100):
    """
    Plot loss and print stats of weights using example neural network
    :param model_list:
    :param plot_title:
    :param train_loader:
    :param valid_loader:
    :param plot_n_batches:
    :return:
    """
    colors = ['r', 'b', 'g', 'c', 'y', 'k']
    label_accs = []
    label_loss = []

    assert len(model_list) <= len(colors), 'Too many initial weights to plot'

    for i, (model, label) in enumerate(model_list):
        loss, val_acc = get_loss_acc(model, train_loader, valid_loader)

        plt.plot(loss[:plot_n_batches], colors[i], label=label)
        label_accs.append((label, val_acc))
        label_loss.append((label, loss[-1]))

    plt.title(plot_title)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.show()

    print('After 2 epochs:')
    print('Validation accuracy')
    for label, val_acc in label_accs:
        print('  {:7.3f}% -- {}'.format(val_acc * 100, label))
    print('Training loss')
    for label, loss in label_loss:
        print('  {:7.3f}% -- {}'.format(loss, label))


def hist_dist(title, distribution_tensor, hist_range=(-4, 4)):
    """
    Display histogram values in a given distribution tensor
    :param title:
    :param distribution_tensor:
    :param hist_range:
    :return:
    """
    plt.title(title)
    plt.hist(distribution_tensor, np.linspace(*hist_range, num=int(len(distribution_tensor) / 2)))
    plt.show()
