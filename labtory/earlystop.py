def train(device, model, epochs, optimizer, loss_function, train_loader, valid_loader):
    # Early stopping
    the_last_loss = 100
    patience = 2
    trigger_times = 0

    for epoch in range(1, epochs+1):
        model.train()

        for times, data in enumerate(train_loader, 1):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            outputs = model(inputs.view(inputs.shape[0], -1))
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # Show progress
            if times % 100 == 0 or times == len(train_loader):
                print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, epochs, times, len(train_loader), loss.item()))

        # Early stopping
        the_current_loss = validation(model, device, valid_loader, loss_function)
        print('The current loss:', the_current_loss)

        if the_current_loss > the_last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                return model
        else:
            trigger_times = 0

        the_last_loss = the_current_loss

    return model