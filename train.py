from tqdm.auto import tqdm


def network_train(network, data_loader, optimizer, loss_function, device):
    network.train()
    total_correct = 0
    total_loss = 0

    for labels, images in tqdm(data_loader):
        labels, images = labels.to(device), images.to(device)

        preds = network(images)
        loss = loss_function(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        _, preds = preds.max(1)
        total_correct += preds.eq(labels).sum()

    avg_train_loss = total_loss / 50000
    train_acc = total_correct.float() / 50000
    print(f'Train set: Average loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')
    return avg_train_loss, train_acc


def network_eval(network, data_loader, loss_function, device):
    network.eval()
    test_correct = 0
    test_loss = 0

    for labels, images in data_loader:
        labels, images = labels.to(device), images.to(device)

        preds = network(images)
        loss = loss_function(preds, labels)
        test_loss += loss.item() * len(labels)
        _, preds = preds.max(1)
        test_correct += preds.eq(labels).sum()

    avg_test_loss = test_loss / 10000
    test_acc = test_correct.float() / 10000
    print(f'Test set: Average loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.4f}')
    return avg_test_loss, test_acc
