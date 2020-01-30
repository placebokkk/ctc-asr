import torch

from model import Net


def print_mat(t):
    m, n = t.size()
    for i in range(m):
        a = ''
        for j in range(n):
            a = a + ' ' + str(t[i][j].data[0])
        print(a)


def print_lstm(m):
    weight_list = {}
    for name in m.named_parameters():
        if 'weight' in name[0]:
            print(name[0])
            print(name[1][0])
        if 'bias' in name[0]:
            print(name[0])
            print(name[1])


if __name__ == '__main__':
    model_paras = {
        'input_dim': 120,
        'hidden_dim': 640,
        'num_layers': 4,
        'output_dim': 74  # 73 phone + 1 blank
    }

    model_path = '/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/model2/init.pt'
    checkpoint = torch.load(model_path)

    model = Net(model_paras)
    model.load_state_dict(checkpoint['model'])
    for name in model.named_parameters():
        print(name[0])
        print(name[1].size())
    torch.set_printoptions(profile="full")
    print_lstm(model.lstm)
    print('model.fc.weight')
    print(model.fc.weight)
    print('model.fc.weight.norm')
    print(torch.norm(model.fc.weight, dim=1))
    print('model.fc.weight.sum')
    print(model.fc.weight.sum(dim=1))
