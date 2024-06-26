import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

# GPU를 사용할 수 있으면 사용. 아니라면 CPU를 사용.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 코드를 실행할때마다 동일한 결과를 얻기 위하여 랜덤 시드를 고정.
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

# 데이터셋을 불러와서 텐서로 변환.
# 이미지 데이터를 파이토치가 처리할 수 있도록 만듬.
mnist_train = dsets.MNIST(root='MNIST_data/', 
                          train=True, 
                          transform=transforms.ToTensor(), 
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', 
                         train=False, 
                         transform=transforms.ToTensor(), 
                         download=True)

# 데이터로더를 설정해서 배치 크기를 조절.
# 데이터를 주어진 배치 크기로 나눔.
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# 모델 정의
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # 입력 1, 출력 32, 커널 크기 3*3
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        # 두번째층
        # 입력 32, 출력 64, 커널 크기 3*3
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        # 전결합층
        # 입력 7*7*64, 출력 10 
        self.fc = torch.nn.Linear(7 * 7 * 64, 10)
        
        # 전결합층 가중치 초기화.
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # 합성곱과 풀링 적용.
        out = self.layer1(x)
        out = self.layer2(out)
        # 출력 값 평탄화.
        # 전결합층에 넣기 위해 높은 차원의 데이터를 1차원으로 변환.
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 모델과 비용 함수, 옵티마이저 정의
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 총 배치 수 출력
total_batch = len(data_loader)
print("총 배치 수: {}".format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device) # 데이터
        Y = Y.to(device) # 레이블

        optimizer.zero_grad() # 기울기 초기화
        hypothesis = model(X) # 예측값
        cost = criterion(hypothesis, Y) # 예측값과 실제 값의 차이
        cost.backward()
        optimizer.step() # 기울기를 이용해 가중치 업데이트

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# 평가 시에는 기울기를 계산하지 않음.
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
