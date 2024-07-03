import math
class BinarySearch:

    def __init__(self, AList):
        AList.sort(reverse=False)
        print(AList)


    def BS(self,NumberToFind, ListofNumberstoPass):
        firstIndex=0
        midIndex =0
        lastIndex= len(ListofNumberstoPass)-1
        found= 'false'

        while firstIndex<=lastIndex:
            midIndex =((firstIndex+lastIndex) // 2)
            if NumberToFind> int(ListofNumberstoPass[midIndex]):
                firstIndex=midIndex+1
            elif NumberToFind< int(ListofNumberstoPass[midIndex]):
                lastIndex=midIndex-1
            elif NumberToFind==int(ListofNumberstoPass[midIndex]):
                print(" midPos="+ str(midIndex) + " firstPos="+str(firstIndex)+ " lastPos=" + str(lastIndex))
                found='true'
                break

        if found=='false':
            print("not Found")






print("enter numbers with space inbetween")
Li = str(input()).split(" ")
print(Li)

B= BinarySearch(Li)
print("enter the number to Search")
N=int(input())
B.BS(N,Li)