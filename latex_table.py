# DeepGOld
# Create the main latex table of results for the paper
# copyright 2022 moshe sipper  
# www.moshesipper.com 
import operator

ds = {'fashionmnist,': 'Fashion-MNIST',
      'cifar10,': 'CIFAR10',
      'cifar100,': 'CIFAR100',
      'tinyimagenet,': 'Tiny ImageNet'}

algs = {'RidgeClassifier': {'name': 'RG', 'count': 0},
        'KNeighborsClassifier': {'name': 'KN', 'count': 0},
        'SGDClassifier': {'name': 'SG', 'count': 0},
        'PassiveAggressiveClassifier': {'name': 'PA', 'count': 0},
        'LogisticRegression': {'name': 'LR', 'count': 0},
        'RandomForestClassifier': {'name': 'RF', 'count': 0},
        'MLPClassifier': {'name': 'MP', 'count': 0},
        'LGBMClassifier': {'name': 'LG', 'count': 0},
        'XGBClassifier': {'name': 'XG', 'count': 0},
        'CatBoostClassifier': {'name': 'CB', 'count': 0}}

with open('res.txt', 'r') as f:
    lines = f.readlines()
assert len(lines)%9==0

f = open('res.txt', 'r')

fashion,cf10,cf100,tiny=False,False,False,False
i=0
while True:
    j = i%9
    if j==0:
        col1 = ''
    line = f.readline().split()
    if not line:
        break
    dataset = ds[line[0]]
    if dataset=='Fashion-MNIST' and not fashion:
        fashion=True
        col1 = f'\\hline\n\multirow{{10}}{{*}}{{{dataset}}}'
    elif dataset=='CIFAR10' and not cf10:
        cf10=True
        col1 = f'\\hline\n\multirow{{10}}{{*}}{{{dataset}}}'
    elif dataset=='CIFAR100' and not cf100:
        cf100=True
        col1 = f'\\hline\n\multirow{{10}}{{*}}{{{dataset}}}'
    elif dataset=='Tiny ImageNet' and not tiny:
        tiny=True
        col1 = f'\\hline\n\multirow{{10}}{{*}}{{{dataset}}}'
    
    if j==0:
        net, maj, ml = [], [], [] # strings
        netf, majf, mlf = [], [], [] # just the numbers
    
    if j in [0,1,2]:
        netf.append(line[3].strip('%').strip('\n'))
        net.append(f'{netf[-1]}\%')
    elif j in [3,4,5]:
        majf.append(line[5].strip('%').strip('\n'))
        maj.append(f'{majf[-1]}\%')
    elif j in [6,7,8]:
        mlf.append(line[6].strip('%').strip('\n'))
        algname = line[1].replace(',','').replace('-optuna','')
        algs[algname]['count'] += 1
        ml.append(f'{mlf[-1]}\% ({algs[algname]["name"]})')
    if j==8:
        for k in range(3):
            l = [float(netf[k]), float(majf[k]), float(mlf[k])]
            mx = l.index(max(l))
            if mx==0:
                net[k] = '\\bb{' + net[k] + '}'
            elif mx==1:
                maj[k] = '\\bb{' + maj[k] + '}'
            elif mx==2:
                ml[k] = '\\bb{' + ml[k] + '}'     
        print(f'{col1} & {net[0]} & {maj[0]} & {ml[0]} & {net[1]} & {maj[1]} & {ml[1]} & {net[2]} & {maj[2]} & {ml[2]} \\\\')
    i += 1
    
print()
lst = list(algs.values())
srt = sorted(lst, key=lambda d: d['count'], reverse=True) 
for alg in srt:
    print(alg['name'], alg['count'])


#Dataset & \multicolumn{3}{|c|}{3 networks} & \multicolumn{3}{c}{7 networks}  & \multicolumn{3}{|c}{11 networks}  \\ \hline
# dataset & Net & Maj & ML & Net & Maj & ML & Net & Maj & ML

# \multirow{10}{*}{Fashion-MNIST} 
#       & xx.xx\% & xx.xx\% & xx.xx\% (XX) & xx.xx\% & xx.xx\% & xx.xx\% (LR) & xx.xx\% & xx.xx\% & xx.xx\% (LR) \\
#       & xx.xx\% & xx.xx\% & xx.xx\% (XX) & xx.xx\% & xx.xx\% & xx.xx\% (XX) & xx.xx\% & xx.xx\% & xx.xx\% (XX) \\


# fashionmnist, 3, resnext101_32x8d, 91.97%
# fashionmnist, 7, regnet_x_3_2gf, 92.23%
# fashionmnist, 11, vgg16_bn, 94.22%
# fashionmnist, test, 3, majority-vote accuracy, 92.38%
# fashionmnist, test, 7, majority-vote accuracy, 93.12%
# fashionmnist, test, 11, majority-vote accuracy, 93.79%
# fashionmnist, SGDClassifier-optuna, 3, test accuracy score: 91.99%
# fashionmnist, LGBMClassifier-optuna, 7, test accuracy score: 93.25%
# fashionmnist, RidgeClassifier-optuna, 11, test accuracy score: 94.40%
# fashionmnist, 3, resnext101_32x8d, 91.97%
# fashionmnist, 7, regnet_x_3_2gf, 92.23%
# fashionmnist, 11, vgg16_bn, 94.22%
# fashionmnist, test, 3, majority-vote accuracy, 92.38%
# fashionmnist, test, 7, majority-vote accuracy, 93.12%
# fashionmnist, test, 11, majority-vote accuracy, 93.79%
# fashionmnist, SGDClassifier-optuna, 3, test accuracy score: 91.99%
# fashionmnist, LGBMClassifier-optuna, 7, test accuracy score: 93.25%
# fashionmnist, RidgeClassifier-optuna, 11, test accuracy score: 94.40%
