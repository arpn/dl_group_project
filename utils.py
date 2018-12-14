from sklearn import metrics
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Measure model accuracy and F1-score
def measure(model, data_loader, device=torch.device('cuda')):
    model.eval()
    accuracy = 0
    prediction = torch.tensor([], device=device)
    true_labels = torch.tensor([], device=device)
    with torch.no_grad():
        for (X, y) in data_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            pred = output.sigmoid().round()
            prediction = torch.cat((prediction, pred))
            true_labels = torch.cat((true_labels, y))
            accuracy += (y*pred).sum().item()
        accuracy = accuracy/len(data_loader.dataset)
        f1_score = metrics.f1_score(true_labels.cpu().numpy(),
                                    prediction.cpu().numpy(),
                                    average='micro')
    return accuracy, f1_score

# Extract predictions from model
def predict(model, data_loader, device=torch.device('cuda')):
    model.eval()
    prediction = torch.tensor([], device=device)
    for (X, _) in data_loader:
        X = X.to(device)
        output = model(X)
        prediction = torch.cat((prediction, output.sigmoid().round()))
    return prediction.cpu().numpy()

#def f1Score(data_loader, k):
#    model.eval()
#    true_index_vectors = []
#    index_vectors = []
#    for (X, y) in data_loader:
#        X = X.to(device)
#        y = y.to(device)
#        output = model(X)
#        # Iterate over predicted labels
#        for batch_idx in range(len(y)):
#            k_highest_idx = torch.topk(output[batch_idx], k)[1].cpu().numpy()
#            idx_vector = np.zeros(len(output[batch_idx]))
#            for k_high in k_highest_idx:
#                idx_vector[k_high] = 1
#            index_vectors.append(idx_vector)
#            true_index_vectors.append(y[batch_idx].cpu().numpy())
#    true_index_vectors = np.array(true_index_vectors)
#    index_vectors = np.array(index_vectors)
#    f1_score = metrics.f1_score(
#        true_index_vectors,
#        index_vectors,
#        average='micro'
#    )
#    return f1_score
#
#def predict(data_loader):
#    result = np.zeros((len(data_loader.dataset),output_dim))
#    model.eval()
#    pos = 0
#    for (batch_idx,(X,y)) in enumerate(data_loader):
#        X = X.to(device)
#        output = model(X)
#        output = torch.round(torch.sigmoid(output))
#        result[pos:pos+len(X),] = output.cpu().detach().numpy()
#        pos += len(X)
#    return result
#
#def predictF1(data_loader):
#    result = predict(data_loader)
#    true_result = np.zeros((len(data_loader.dataset),output_dim))
#    for i in range(len(data_loader.dataset)):
#        true_result[i] = data_loader.dataset[i][1].numpy()
#    f1_score = metrics.f1_score(
#        true_result,
#        result,
#        average='micro'
#    )
#    return f1_score

def train(model, train_loader, criterion, optimizer, epoch, epochs, train_vector, logs_per_epoch=7, device=torch.device('cuda')):
    model.train()
    train_loss = 0
    num_batches = len(train_loader)
    start = time.time()
    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad() 
        output = model(X)
        loss = criterion(output, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if batch_idx % (num_batches//logs_per_epoch) == 0 and batch_idx > 0:
            now = time.time()
            batch_size = len(y)
            inputs_per_sec = ((batch_idx+1)*batch_size)/(now-start)
            eta_min = (epochs*num_batches-(epoch-1)*num_batches-(batch_idx+1))*batch_size/inputs_per_sec/60
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tInputs/s: {:.1f}\tRemaining: {:.1f} min'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), inputs_per_sec, eta_min))

    train_loss /= len(train_loader)
    train_vector.append(train_loss)

def validate(model, test_loader, criterion, loss_vector, f1_vector=[], device=torch.device('cuda')):
    model.eval()
    val_loss = 0
    print('\nValidating...')
    with torch.no_grad():
        for (X, y) in test_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            val_loss += criterion(output, y).data.item()

    val_loss /= len(test_loader)
    loss_vector.append(val_loss)
    f1_score = measure(model, test_loader, device)[1]
    f1_vector.append(f1_score)

    print('Validation set: Average loss: {:.4f}\t F1-score: {:.4f}\n'.format(val_loss, f1_score))
