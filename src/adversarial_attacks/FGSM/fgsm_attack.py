def fgsm_attack(model, X, y, epsilon):
    X.requires_grad = True

    outputs = model(X)
    loss = nn.CrossEntropyLoss()(outputs, y)
    model.zero_grad()
    loss.backward()

    data_grad = X.grad.data
    perturbed_data = X + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data


def elbow_method_epsilon(model, X, y):
    epsilons = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    accuracies = []

    for epsilon in epsilons:
        perturbed_X = fgsm_attack(model, X, y, epsilon)
        outputs = model(perturbed_X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean()
        accuracies.append(accuracy.item())

    # Elbow 방법을 사용하여 최적의 epsilon 선택
    # (이 부분은 실제 구현에서 더 정교하게 작성해야 합니다)
    optimal_epsilon = epsilons[accuracies.index(min(accuracies))]

    return optimal_epsilon