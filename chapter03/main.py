from MachineLearningInAction.chapter03 import trees
from MachineLearningInAction.chapter03 import treePlotter


if __name__ == '__main__':
    myDat, labels = trees.createDataSet()
    print(myDat)
    print(trees.calcShannonEnt(myDat))

    # 增加maybe分类,观察熵的变化
    myDat[0][-1] = 'maybe'
    print(myDat)
    print(trees.calcShannonEnt(myDat))

    myTree = trees.createTree(myDat, labels)
    print(myTree)

    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree=trees.createTree(lenses,lensesLabels)
    treePlotter.createPlot(lensesTree)
