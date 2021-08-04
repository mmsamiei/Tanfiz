def train(model, loader, device, max_epoch, optimizer, criterion):
  model.train()
  for epoch in range(max_epoch):  # loop over the dataset multiple times
      running_loss = 0.0
      for i, data in tqdm(enumerate(loader)):

          inputs, labels = data
          inputs = inputs.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
      print('[%d] loss: %.3f' %
            (epoch + 1, running_loss / 100))
      running_loss = 0.0

  print('Finished Training')
