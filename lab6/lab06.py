import random
import numpy as np

DNALettersDict = {0: "A", 1: "T", 2: "C", 3: "G"}

#===============================================================================================
#Generate DNA sequence for the test
#===============================================================================================
OriginalDNALength = 1000

OriginalDNASeq = [DNALettersDict[random.randint(0, 3)] for i in range(OriginalDNALength)]

#===============================================================================================
#Generate Fragments
#===============================================================================================
NumberOfFragments = 10000
MinLength = 50
MaxLength = 400

def SelFragment(DNASeq):
  startPos = random.randint(0, OriginalDNALength-1-MinLength)
  length = random.randint(MinLength, MaxLength)
  return DNASeq[startPos:startPos+length]

ListOfFragments = [ SelFragment(OriginalDNASeq) for i in range(NumberOfFragments)]

def Compare(ListFrag, n, A, B, tempCnt):
    if A >= len(ListFrag[0]) or B >= len(ListFrag[n]):
        return tempCnt
    if ListFrag[0][A] != ListFrag[n][B]:
        return tempCnt
    tempCnt += 1
    tempCnt = Compare(ListFrag, n, (A+1), (B+1), tempCnt)
    return tempCnt

def FindFragment(ListFrag):
    n = 1
    address = 0
    addressA = 0
    addressB = 0
    Cnt = 0
    while n < len(ListFrag) and n < 3:
        i = 0
        while i <(len(ListFrag[0])):
            j = 0
            while j <(len(ListFrag[n])):
                tempCnt = 0
                tempCnt = Compare(ListFrag, n, i, j, 0)
                if tempCnt > Cnt:
                    address = n
                    addressA = i
                    addressB = j
                    Cnt = tempCnt
                j += 1
            i += 1
        n += 1

    Parameter={}
    Parameter[0] = address
    Parameter[1] = addressA
    Parameter[2] = addressB
    Parameter[3] = Cnt
    return Parameter

def MergeDna(ListFrag, Param):
    if Param[3] == len(ListFrag[Param[0]]): 
        del ListFrag[Param[0]]
        #print("Pierwszy")
    elif (Param[1] + Param[3]) == len(ListFrag[0]) and Param[2] == 0:
        for i in range(Param[3],Param[3]+(len(ListFrag[Param[0]])-Param[3])):
            ListFrag[0].append(ListFrag[Param[0]][i])
            #print("Drugi")
        del ListFrag[Param[0]]
    elif (Param[2] + Param[3]) == len(ListFrag[Param[0]]) and Param[1] == 0:
        for i in range(Param[3],Param[3]+(len(ListFrag[0])-Param[3])):
            ListFrag[Param[0]].append(ListFrag[0][i])
            #print("Trzeci")
        del ListFrag[0]

        
Parameter = np.zeros(4)
ListOfFragments.sort(key=len, reverse=True)
while NumberOfFragments > 1:
    Parameter = FindFragment(ListOfFragments)  
    MergeDna(ListOfFragments, Parameter)
    NumberOfFragments = len(ListOfFragments)

#===============================================================================================
#Reconstruct DNA from fragments
#===============================================================================================
print(OriginalDNASeq)
print(len(OriginalDNASeq))
print(ListOfFragments[0])
print(len(ListOfFragments[0]))