import collections

# Define a tree
class Node:
    def __init__(self,Data):
        self.leftChild= None
        self.rightChild=None
        self.data= Data

    #adding/insering a Node
    def insertNode(this,DataToInsert):
        if this.data == None:
            this.data=DataToInsert
        else:
            if DataToInsert < this.data:
                if this.leftChild== None:
                    this.leftChild = Node(DataToInsert)
                else: this.leftChild.insertNode(DataToInsert)
            elif DataToInsert > this.data:
                if this.rightChild== None:
                    this.rightChild = Node(DataToInsert)
                else: this.rightChild.insertNode(DataToInsert)

#Function to print Tree in order
def PrintNodesInOrder(rootNode):
    if rootNode==None:
        return
    else:
        PrintNodesInOrder(rootNode.leftChild)
        print(rootNode.data, end=" ")
        PrintNodesInOrder(rootNode.rightChild)

#nodes pre-order printing
def PrintNodesPreOrder(rootNode):
    if rootNode==None:
        return
    else:
        print(rootNode.data, end=" ")
        PrintNodesPreOrder(rootNode.leftChild)
        PrintNodesPreOrder(rootNode.rightChild)

#nodes post-order printing
def PrintNodesPostOrder(rootNode):
    if rootNode==None:
        return
    else:
        print(rootNode.data, end=" ")
        PrintNodesPostOrder(rootNode.rightChild)
        PrintNodesPostOrder(rootNode.leftChild)

#create Adjucency list, a Dictionary contain multiple [key, value] pairs, where each [Key] is the current Rootnode
#... data and each [value] will be the list of it's two children's data [leftchild Data, RightChild Data]
# Input: myDict = {}
#            myDict["key1"] = [1, 2]                          # Adding list as value
#            myDict["key2"] = ["Geeks", "For", "Geeks"]
# Output: {'key2': ['Geeks', 'For', 'Geeks'], 'key1': [1, 2]}
# Explanation: In the output, we have a dictionary of lists.
def makeAdjacencyListDictionary(rootNode):
    if rootNode==None:
        return
    else:
        dict[rootNode.data]=[]
        makeAdjacencyListDictionary(rootNode.leftChild)

        if rootNode.leftChild:
            dict[rootNode.data].append(rootNode.leftChild.data)
        if rootNode.rightChild:
            dict[rootNode.data].append(rootNode.rightChild.data)

        makeAdjacencyListDictionary(rootNode.rightChild)
        return dict
def makeAdjacencyListDictionaryReverse(rootNode):
    if rootNode==None:
        return
    else:
        dictRev[rootNode.data]=[]
        makeAdjacencyListDictionaryReverse(rootNode.rightChild)

        if rootNode.rightChild:
            dictRev[rootNode.data].append(rootNode.rightChild.data)
        if rootNode.leftChild:
            dictRev[rootNode.data].append(rootNode.leftChild.data)

        makeAdjacencyListDictionaryReverse(rootNode.leftChild)
        return dictRev

def BFS(AdjcListDict):
    Q= collections.deque('g')
    explored=[]

    while Q:
        PoppedNode=Q.popleft()
        explored.append(PoppedNode)
        for li in AdjcListDict[PoppedNode]:   #taking the list-elements foreach Dictionary item
            Q.append(li)

    print(explored)  # traversal: sequence of visiting nodes
def DFS(AdjcListDict):
    #Stack = collections.deque('g')   #also works the same
    Stack = ['g']
    explored = []

    while Stack:
        PoppedNode = Stack.pop()
        if PoppedNode not in explored:
            explored.append(PoppedNode)
        for li in AdjcListDict[PoppedNode]:  # taking the list-elements foreach Dictionary item
            Stack.append(li)

    print(explored)  # traversal: sequence of visiting nodes

def Search(AdLiDic,NodeToSearch):
    Sta=['g']
    visitd=[]
    found= False
    while Sta:
        popdnode=Sta.pop()
        if popdnode == NodeToSearch:
            return "found"
        if popdnode not in visitd:
            visitd.append(popdnode)
            for li in AdLiDic[popdnode]:
                    Sta.append(li)
    return "not found"



# adding real time nodes data to tree
if __name__=='__main__':
    root = Node('g')
    root.insertNode('c')
    root.insertNode('b')
    root.insertNode('a')
    root.insertNode('e')
    root.insertNode('d')
    root.insertNode('f')
    root.insertNode('i')
    root.insertNode('h')
    root.insertNode('j')
    root.insertNode('k')

    dict={}
    dictRev={}

    filledupAdjListDict = makeAdjacencyListDictionary(root)
    filledupAdjListDictReverse = makeAdjacencyListDictionaryReverse(root)
    print("--------------BFS-------------")
    BFS(filledupAdjListDict)
    print("------------DFS (right to left)---------------")
    DFS(filledupAdjListDict)
    print("------------DFS (left to right)---------------")
    DFS(filledupAdjListDictReverse)


    print(Search(filledupAdjListDict,'c'))


    # for key in filledupDict:
    #     print(f'{eachPair}:{dict[key]}')

    for key,value in filledupAdjListDict.items():
        print(key,":",value)

# Printing the nodes in (any particular) Order
print("\n IN-Order")
PrintNodesInOrder(root)
print("\n PRE-Order")
PrintNodesPreOrder(root)
print("\n POST-Order")
PrintNodesPostOrder(root)